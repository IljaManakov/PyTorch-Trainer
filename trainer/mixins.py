#!/usr/bin/env python
# encoding: utf-8
"""
mixins.py

several mixins that provide additional functionality to the Trainer class

Copyright (c) 2019, I. Manakov

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
to permit persons to whom the Software is furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial portions
of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import os
from collections import namedtuple
from functools import wraps
from time import time, ctime

import h5py
import torch as pt
from trainer.utils import to_numpy


class SaveMixin(object):
    """used to automatically collect objects of specified class and save them using pt.save"""

    def save(self, *args,  save_config=None, checkpoint_name=None, directory='./', **kwargs):
        """
        create a checkpoint with all objects of the types specified in the save config
        :param save_config: dict that specifies which attributes to save. Keys should be classes and values conversion methods.
        :param checkpoint_name: name of the checkpoint that will be created
        :param directory: folder where checkpoints will be saved
        :return: filename of the saved checkpoint
        """

        if save_config is None:
            save_config = self.default_save_config.copy()

        # find all objects that need to be saved
        objects = {}

        for key, value in self.__dict__.items():

            # apply conversion method specified in save_config for that class
            if value in save_config.keys():

                # find conversion method for present value
                for level in (value, self, object, type):
                    conversion = getattr(level, save_config[value], None)
                    if conversion is not None:
                        break

                # convert the value for storage
                objects[key] = conversion() if level is value else conversion(value)

        # set default save path if it is None
        if checkpoint_name is not None:
            checkpoint = checkpoint_name
        else:
            checkpoint = f'checkpoint_{ctime().replace(" ", "-")}.pt'

        # save the objects
        pt.save(objects, os.path.join(directory, checkpoint))
        return checkpoint

    @staticmethod
    def to_numpy(sample):
        """
        convert individual and sequences of Tensors to numpy, while leaving strings, integers and floats unchanged.
        set up as bound method because of the way conversion method lookup works in save
        :param sample: individual or sequence of Tensors, ints, floats or strings
        :return: same as sample but all Tensors are converted to numpy.ndarray
        """
        return to_numpy(sample)

    @property
    def default_save_config(self):
        """initialize the default save config that saves pytorch Modules, Tensors and Optimizers
         as well as strings, floats and ints. If apex is installed FP16_Optimizers are also saved"""
        save_config = {pt.nn.Module: 'state_dict',
                       pt.optim.Optimizer: 'state_dict',
                       pt.Tensor: '_to_numpy',
                       int: '__int__',
                       str: '__str__',
                       float: '__float__'}

        try:
            from apex.fp16_utils import FP16_Optimizer
            save_config[FP16_Optimizer] = 'state_dict'
        except ImportError:
            pass

        return save_config


class ValidationMixin(object):
    """
    get losses over a validation set
    """

    @staticmethod
    def validate(*args, dataloader, forward_pass, **kwargs):
        """
        evaluate model on validation set
        :param dataloader: Dataloader with validation set
        :param forward_pass: callable that returns (prediction, loss)
        :return: ndarray of losses
        """

        losses = []
        for sample in dataloader:

            with pt.no_grad():
                try:
                    prediction, loss = forward_pass(sample)
                except ValueError:
                    continue
                losses.append(to_numpy(loss))

        return losses


class TestSampleMixin(object):
    """used to perform inference on a single sample"""

    @staticmethod
    def test_on_sample(*args, sample, model, criterion, **kwargs):
        """
        perform inference on the test sample
        :return: result of inference as namedtuple('prediction', 'loss')
        """
        input, target = sample
        with pt.no_grad():
            prediction = model(input)
            loss = criterion(prediction, target)

        prediction, loss = to_numpy((prediction, loss))

        return namedtuple('test_on_sample', ('prediction', 'loss'))(prediction, loss)


class MonitorMixin(object):
    """
    wraps public methods to intercept outputs and store them in a hdf5 file
    """

    def __del__(self):
        """make sure that storage is closed on deconstruction"""
        if hasattr(self, 'storage') and hasattr(self.storage, 'close'):
            self.storage.close()

    @staticmethod
    def open_storage(*, filename):
        """
        creates or opens the hdf5 storage and appends it to instance attributes
        :param filename: name of the hdf5 storage
        :return: h5py.File instance
        """

        # create directory if necessary
        if not os.path.isdir(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))

        # open hdf5 file
        if os.path.isfile(filename):
            mode = 'w'
        else:
            mode = 'a'

        storage = h5py.File(filename, mode, libver='latest', swmr=True)

        return storage

    def close_storage(self, *args, **kwargs):
        """
        generic method for closing storage
        :param args: not used
        :param kwargs: not used
        :return: None
        """

        storage = getattr(self, 'storage', None)
        if storage is not None:
            storage.close()

    def monitor(self, *, name, method=None):
        """
        wrap a funciton to intercept its results. If only a name is specified, method is retrieved from self and
        will be replaced with wrapped version. If name and method are specified wrapped function will be returned instead.
        :param name: name of the function / method
        :param method: callable to wrap. If this argument is specified, the passed callable will be wrapped and returned.
        Otherwilse, the method will be retrieved from self via getattr and replaced with its wrapped version.
        :return: wrapped callable if method is not None else None
        """
        set_method = False
        if method is None:
            method = getattr(self, name)
            set_method = True
        method = self._wrap(method, name)

        if set_method:
            setattr(self, name, method)
        else:
            return method

    def _wrap(self, func, name):
        """
        intercept return values of func and store them in the hdf5 file before returning them
        :param func: method to wrap
        :return: wrapped method
        """

        @wraps(func)
        def monitored(*args, key=None, **kwargs):

            results = func(*args, **kwargs)

            if key is None:
                key = time()

            # only write to storage if method returns something
            if results is not None:
                self._store(results, name, key)

            return results

        return monitored

    def _store(self, results, group_name, key):
        """
        writes results to specified group in the hdf5 file
        :param results: results as returned by some method
        :param group_name: name of hdf5 group object in the hdf5 file
        """

        storage = getattr(self, 'storage', None)

        # check if storage is initialized
        if storage is None:
            return None

        # check if storage is open
        if storage.name is None:
            return None

        key = f'{key}'
        group = storage.require_group(group_name)

        # case where the result has internal structure (i.e. is an object)
        if hasattr(results, '__dict__'):

            # obtain public fields from the object
            fields = [field for field in results.__dict__ if not field.startswith('_')]
            for field in fields:
                key = '/'.join([key, field])
                group[key] = getattr(results, field)

        # case where the result is a namedtuple
        elif hasattr(results, '_fields'):

            # obtain fields from namedtuple
            for field in getattr(results, '_fields'):
                field_key = '/'.join([key, field])
                group[field_key] = getattr(results, field)

        # default case
        else:
            group[key] = results

        storage.flush()


class CheckpointMixin(object):

    def load(self, *args, checkpoint):
        """
        load a checkpoint that can contain model and optimizer state
        :param checkpoint: filename of the checkpoint
        :return: None
        """

        checkpoint = pt.load(checkpoint)

        # load model state
        if hasattr(self, 'model') and 'model' in checkpoint.keys():
            print('loading model checkpoint ...', end='')
            model = getattr(self, 'model')
            model.load_state_dict(checkpoint['model'])
            print('done!')

        # load optimizer state
        if hasattr(self, 'optimizer') and 'optimizer' in checkpoint.keys():
            print('loading optimizer checkpoint...', end='')
            optimizer = getattr(self, 'optimizer')
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('done!')
