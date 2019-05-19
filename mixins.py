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
import sys
from collections import Sequence, namedtuple
from functools import wraps, partial
from time import ctime

import h5py
import torch as pt
sys.path.extend(['/home/kazuki/Documents/Promotion/Project_Helpers/trainer'])
import events


def to_numpy(sample):
    """
    convert individual and sequences of Tensors to numpy, while leaving strings, integers and floats unchanged
    :param sample: individual or sequence of Tensors, ints, floats or strings
    :return: same as sample but all Tensors are converted to numpy.ndarray
    """
    if isinstance(sample, int) or isinstance(sample, float) or isinstance(sample, str):
        return sample
    elif isinstance(sample, pt.Tensor):
        return sample.detach().cpu().numpy()
    elif isinstance(sample, tuple) or isinstance(sample, list):
        return [to_numpy(s) for s in sample]
    elif isinstance(sample, Sequence):
        collection = sample.__class__
        return collection(*[to_numpy(s) for s in sample])


class IntervalBased(object):
    """
    used for decorating functions.
    the function will then only be executed at set intervals
    """
    def __init__(self, interval):
        """
        :param interval: interval at which the function will be called
        """
        self.interval = interval
        self.counter = 0

    def __call__(self, func):
        """
        wraps function so that it is executed only at set interval
        :param func: function to wrap
        :return: wrapped function
        """
        @wraps(func)
        def wrapped(*args, **kwargs):
            self.counter += 1
            if self.counter % self.interval == 0:
                return func(*args, **kwargs)

        return wrapped


class SaveMixin(object):
    """used to automatically collect objects of specified class and save them using pt.save"""

    def save(self, save_config, checkpoint_name=None, directory='./', *args, **kwargs):
        """
        create a checkpoint with all objects of the types specified in the save config
        :param save_config: dict that specifies which attributes to save. Keys should be classes and values conversion methods.
        :param checkpoint_name: name of the checkpoint that will be created
        :param directory: folder where checkpoints will be saved
        :return: filename of the saved checkpoint
        """

        # find all objects that need to be saved
        objects = {}

        for key, value in self.__dict__.items():

            # check if value is instance of classes in save_config
            cls = None
            for obj in save_config.keys():
                if isinstance(value, obj):
                    cls = obj
                    break

            # apply conversion method specified in save_config for that class
            if cls is not None:

                # find conversion method for present value
                for level in (value, self, object, type):
                    conversion = getattr(level, save_config[cls], None)
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
    def _to_numpy(sample):
        """
        convert individual and sequences of Tensors to numpy, while leaving strings, integers and floats unchanged.
        set up as bound method because of the way
        :param sample: individual or sequence of Tensors, ints, floats or strings
        :return: same as sample but all Tensors are converted to numpy.ndarray
        """
        return to_numpy(sample)


class EventSaveMixin(SaveMixin):

    @staticmethod
    def setup_saving(*, directory='./', save_config=None, event=events.EACH_EPOCH, interval=1):
        """
        register an event for interval based saving
        :param directory: directory in which to save
        :param save_config: config with the classes that should be saved, default=None results in default config
        :param event: event type that will be registered, default=events.EACH_EPOCH
        :param interval: interval at which the testing should occur, default=1
        :return: None
        """

        save_config = save_config if save_config is not None else EventSaveMixin.default_save_config()

        # modify save method to include interval, directory and save_config
        new_save = IntervalBased(interval)(partial(SaveMixin.save, save_config=save_config, directory=directory))

        # create event method
        setattr(EventSaveMixin, event, new_save)

    @staticmethod
    def default_save_config():
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
    def validate(validation_loader, forward_pass, *args, **kwargs):
        """
        evaluate model on validation set
        :param validation_loader:
        :param forward_pass:
        :return: ndarray of losses
        """

        losses = []
        for sample in validation_loader:

            with pt.no_grad():
                try:
                    prediction, loss = forward_pass(sample)
                except ValueError:
                    continue
                losses.append(to_numpy(loss))

        return losses


class EventValidationMixin(ValidationMixin):

    @staticmethod
    def setup_validation(*, dataloader, forward_pass, event=events.EACH_EPOCH, interval=1):
        """
        register an event for interval based validation
        :param dataloader: pytorch dataloader over the validation set
        :param interval: frequency of validation
        """
        # modify validate method to include interval, dataloader and forward_pass
        new_validate = partial(ValidationMixin.validate, validation_loader=dataloader, forward_pass=forward_pass)
        new_validate = IntervalBased(interval)(new_validate)

        # create event method
        setattr(EventValidationMixin, event, new_validate)


class TestSampleMixin(object):
    """used to perform inference on a single fixed sample at regular intervals for tracking training progress"""

    def test_on_sample(self, sample, model, criterion, *args, **kwargs):
        """
        perform inference on the test sample
        :return: result of inference as namedtuple('prediction', 'loss')
        """

        with pt.no_grad():
            prediction = model(sample)
            loss = criterion(prediction, sample)

        prediction, loss = to_numpy((prediction, loss))

        return namedtuple('test_on_sample', ('prediction', 'loss'))(prediction, loss)


class EventTestSampleMixin(TestSampleMixin):

    @staticmethod
    def setup_test_sample(*, sample, model, criterion, event=events.EACH_EPOCH, interval=1):
        """
        register an event for periodic tests on a given sample
        :param sample: sample to test on
        :param model: model to test
        :param criterion: criterion for loss calculation
        :param event: which event to register, deflaut=events.EACH_EPOCH
        :param interval: interval at which the testing should occur, default=1
        :return: None
        """

        # modify sample test method to include interval, sample, model and criterion
        new_test_on_sample = partial(EventTestSampleMixin.test_on_sample, sample=sample, model=model, criterion=criterion)
        new_test_on_sample = IntervalBased(interval)(new_test_on_sample)

        setattr(EventTestSampleMixin, event, new_test_on_sample)


class MonitorMixin(object):
    """used to monitor and store outputs of public methods to a hdf5 file"""

    def setup_monitoring(self, filename, exclusions=('setup', 'train')):
        """gather public methods for monitoring"""

        setattr(MonitorMixin, events.BEFORE_TRAINING, lambda: (self._open_storage(filename), self._wrap_methods(exclusions)))
        setattr(MonitorMixin, events.AFTER_TRAINING, lambda: getattr(self, 'storage').close())

    def _open_storage(self, filename):

        # open hdf5 file
        if os.path.isfile(filename):
            mode = 'w'
        else:
            mode = 'a'

        if not hasattr(self, 'storage'):
            setattr(self, 'storage', h5py.File(filename, mode, libver='latest'))
        storage = getattr(self, 'storage')
        storage.swmr_mode = True

    def _wrap_methods(self, exclusions):

        monitored = []
        for entry in self.__dir__():

            # exclude entry if it is protected/private or not callable
            if entry.startswith('_'):
                continue
            if not callable(getattr(self, entry)):
                continue

            # make sure callable is not an attribute of a an attribute (i.e. a stored class)
            if entry in self.__class__.__dict__.keys():
                monitored.append(entry)
            elif hasattr(super(self.__class__, self), entry):
                monitored.append(entry)

        # remove excluded methods from monitoring
        monitored = [method for method in monitored if not any([exclusion in method for exclusion in exclusions])]

        # replace method with wrapped version of itself
        for method in monitored:
            setattr(self, method, self._monitor(getattr(self, method)))

        print('\nmonitoring following methods:')
        print('-----------------------------')
        for method in monitored:
            print(method)
        print()

    def _monitor(self, func):
        """
        wraps a method and stores return values in the hdf5 file before returning them
        :param func: method to wrap
        :return: wrapped method
        """
        if type(func) is partial:
            name = func.func.__name__
        else:
            name = func.__name__

        @wraps(func)
        def monitored(*args, step=None, **kwargs):

            results = func(*args, **kwargs)

            if step is None:
                step = f'{ctime().replace(" ", "-")}'

            # only write to storage if method returns something
            if results is not None:
                self._store(results, name, step)

            return results

        return monitored

    def _store(self, results, group_name, step):
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

        key = f'{step}'
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

    def load(self, checkpoint):
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


class EventCheckpointMixin(CheckpointMixin):

    @staticmethod
    def setup_loading_chechpoint(*, checkpoint):
        """
        register an event to load a checkpoint before training
        :param checkpoint: filename of the checkpoint
        :return: None
        """

        load_checkpoint = partial(CheckpointMixin.load, checkpoint=checkpoint)
        setattr(EventTestSampleMixin, events.BEFORE_TRAINING, load_checkpoint)
