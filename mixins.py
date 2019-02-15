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
from collections import Sequence, namedtuple
from functools import wraps

import h5py
import torch as pt
from torch.utils.data import DataLoader


class ToNumpyMixin(object):
    """provides capability to convert individual and sequences of Tensors to numpy"""

    def __init__(self):
        super().__init__()

    def _to_numpy(self, sample):
        """
        Tensors are converted to ndarrays while ints and floats are unaltered
        :param sample: individual or sequence of Tensor, int and float
        :return: converted sample, data structures such as namedtuple are retained
        """

        if isinstance(sample, int) or isinstance(sample, float):
            return sample
        elif isinstance(sample, pt.Tensor):
            return sample.detach().cpu().numpy()
        elif isinstance(sample, tuple) or isinstance(sample, list):
            return [self._to_numpy(s) for s in sample]
        elif isinstance(sample, Sequence):
            collection = sample.__class__
            return collection(*[self._to_numpy(s) for s in sample])


class TrainerMixin(object):
    """base for Mixins that are aimed at the Trainer class"""

    def __init__(self):
        self._assertion()
        super().__init__()

    def _assertion(self):
        """asserts that Mixin is inherited by a class that has Trainer structure"""
        assert isinstance(getattr(self, 'model', None), pt.nn.Module), 'model attribute is absent or not an instance ' \
                                                                       'of pt.nn.Module '
        assert callable(getattr(self, '_transform', None)), 'transform attribute is absent or not callable'
        assert callable(getattr(self, 'criterion', None)), 'criterion attribute is absent or not callable'
        assert hasattr(self, 'steps'), 'steps attribute is absent'

    @staticmethod
    def _convert(input, conversion, name):
        """
        attempts to cast input to specified type
        :param input: value to cast
        :param conversion: cast function
        :param name: string representation of the type
        :return: converted input
        """

        try:
            return conversion(input)
        except ValueError:
            raise ValueError(f'cannot convert {input} to {name}')


class SaveMixin(TrainerMixin, ToNumpyMixin):
    """used to automatically collect objects of specified class and save them using pt.save"""

    save_interval = None
    save_counter_field = None
    save_config = None
    save_directory = None
    last_counter_status = 0

    def __init__(self):
        super().__init__()
        if isinstance(getattr(self, 'call_after_single_step'), list):
            self.call_after_single_step.append('save')

    def save(self, *, force=False):
        """
        create a checkpoint with all objects of the types specified in the save config
        :param force: ignores counter and forces saving, default: False
        :return: filename of the saved checkpoint
        """
        # get current counter value
        counter = getattr(self, self.save_counter_field, None)

        if not force:
            if self.save_counter_field is None or self.save_interval is None:
                raise UserWarning('saving has not been set up')

            if counter is None:
                raise UserWarning(f'invalid counter field {self.save_counter_field}')

        if counter % self.save_interval == 0 and counter != self.last_counter_status or force:

            # store counter value (necessary if saving only once every epoch)
            if not force:
                self.last_counter_status = counter

            # find all objects that need to be saved
            objects = {}
            model_class = None
            for key, value in self.__dict__.items():

                # check if value is instance of classes in save_config
                cls = None
                for obj in self.save_config.keys():
                    if isinstance(value, obj):
                        cls = obj
                        break

                # apply conversion method specified in save_config for that class
                if cls is not None:

                    # find conversion method for present value
                    for level in (value, self, object, type):
                        conversion = getattr(level, self.save_config[cls], None)
                        if conversion is not None:
                            break

                    # convert the value for storage
                    objects[key] = conversion() if level is value else conversion(value)

                # remember class of the model for save path
                if cls == pt.nn.Module:
                    model_class = str(value.__class__).split('\'')[1].split('.')[-1]

            epochs = getattr(self, 'epochs', 'NaN')
            steps = getattr(self, 'steps', 'NaN')
            # set default save path if it is None
            if model_class is not None:
                checkpoint = os.path.join(self.save_directory, f'{model_class}_epochs-{epochs}_steps-{steps}.pt')
            else:
                checkpoint = os.path.join(self.save_directory, f'checkpoint_epochs-{epochs}_steps-{steps}.pt')

            # save the objects
            pt.save(objects, checkpoint)
            return checkpoint

    def default_save_config(self):
        """initialize the default save config"""
        self.save_config = {pt.nn.Module: 'state_dict',
                            pt.optim.Optimizer: 'state_dict',
                            pt.Tensor: '_to_numpy',
                            int: '__int__',
                            str: '__str__',
                            float: '__float__'}

        try:
            from apex.fp16_utils import FP16_Optimizer
            self.save_config[FP16_Optimizer] = 'state_dict'
        except ImportError:
            pass

    # noinspection PyProtectedMember
    def setup_saving(self, *, interval, counter_field='steps', directory='./', save_config=None):
        """
        initialize the saving mixin
        :param interval: integer that represents the frequency of saving
        :param counter_field: name of the field that will be used as a counter, default: 'steps'
        :param directory: directory where the checkpoint files will be created, default: './'
        :param save_config: dict that specifies which objects to save,
                            keys are classes / types and values are the names of the conversion methods,
                            e.g. pt.nn.Module: 'state_dict'
        """
        self.save_interval = TrainerMixin._convert(interval, int, 'integer')
        self.save_counter_field = TrainerMixin._convert(counter_field, str, 'string')
        self.save_directory = TrainerMixin._convert(directory, str, 'string')
        self.save_config = save_config
        if save_config is None:
            self.default_save_config()


class ValidationMixin(TrainerMixin, ToNumpyMixin):
    """
    used to perform validation over a validation set at regular intervals
    """

    validation_loader = None
    validation_interval = None

    def __init__(self):
        super().__init__()
        if isinstance(getattr(self, 'call_after_single_step'), list):
            self.call_after_single_step.append('validate')

    def validate(self):
        """
        perform validation
        :return: validation loss
        """
        if self.validation_loader is None or self.validation_interval is None:
            return

        if (self.steps + 1) % self.validation_interval == 0:

            losses = []
            for sample in self.validation_loader:

                with pt.no_grad():
                    try:
                        prediction, loss = self._forward(sample)
                    except ValueError:
                        continue
                    losses.append(self._to_numpy(loss))

            return losses

    def setup_validation(self, *, dataloader, interval):
        """
        initialize validation mixin
        :param dataloader: pytorch dataloader over the validation set
        :param interval: frequency of validation
        """
        assert isinstance(dataloader, DataLoader), \
            f'expected instance of pt.utils.DataLoader, got {type(dataloader)} instead'
        self.validation_loader = dataloader
        self.validation_interval = self._convert(interval, int, 'integer')


class TestSampleMixin(TrainerMixin, ToNumpyMixin):
    """used to perform inference on a single fixed sample at regular intervals for tracking training progress"""

    test_input = None
    test_target = None
    test_sample_interval = None

    def __init__(self):
        super().__init__()
        if isinstance(getattr(self, 'call_after_single_step'), list):
            self.call_after_single_step.append('test_on_sample')

    def test_on_sample(self):
        """
        perform inference on the test sample
        :return: result of inference as namedtuple('prediction', 'loss')
        """

        if self.test_input is None or self.test_sample_interval is None:
            return

        if self.steps % self.test_sample_interval == 0:
            with pt.no_grad():
                prediction = self.model(self.test_input)
                loss = self.criterion(prediction, self.test_target)

            prediction, loss = self._to_numpy((prediction, loss))

            return namedtuple('test_on_sample', ('prediction', 'loss'))(prediction, loss)

    def setup_test_sample(self, *, sample, interval):
        """
        initialize test sample mixin
        :param sample: sample or mini-batch to be used for tracking progress
        :param interval: frequency of inference
        """

        self.test_input, self.test_target = self._transform(sample)
        self.test_sample_interval = self._convert(interval, int, 'integer')


class MonitorMixin(TrainerMixin):
    """used to monitor and store outputs of public methods to a hdf5 file"""

    monitored = None
    storage = None

    def __init__(self):
        """gather public methods for monitoring"""

        super(MonitorMixin, self).__init__()

        monitored = []
        for entry in self.__dir__():

            # exclude entry if it is protected/private or not callable
            if entry.startswith('_'):
                continue
            if not callable(getattr(self, entry)):
                continue

            # make sure callable is not an attribute of a an attribute (i.e. a stored class)
            if entry in Trainer.__dict__.keys():
                monitored.append(entry)
            elif hasattr(super(self.__class__, self), entry):
                monitored.append(entry)

        self.monitored = monitored

    def setup_monitoring(self, filename, exclusions=('setup', 'train')):
        """
        initialize monitoring
        :param filename: filename of the hdf5 file
        :param exclusions: sequence of (parts of) method names that should be excluded from monitoring
        """
        # open hdf5 file
        if os.path.isfile(filename):
            mode = 'w'
        else:
            mode = 'a'
        self.storage = h5py.File(filename, mode, libver='latest')
        self.storage.swmr_mode = True

        # remove excluded methods from monitoring
        self.monitored = [method for method in self.monitored
                          if not any([exclusion in method for exclusion in exclusions])]

        # replace method with wrapped version of itself
        for method in self.monitored:
            setattr(self, method, self._monitor(getattr(self, method)))

    def _monitor(self, func):
        """
        wraps a method and stores return values in the hdf5 file before returning them
        :param func: method to wrap
        :return: wrapped method
        """

        group = self.storage.require_group(func.__name__)

        @wraps(func)
        def monitored(*args, **kwargs):

            results = func(*args, **kwargs)

            # only write to storage if method returns something
            if results is not None:
                self._store(results, group)

            return results

        return monitored

    def _store(self, results, group):
        """
        writes results to specified group in the hdf5 file
        :param results: results as returned by some method
        :param group: hdf5 group object in the hdf5 file
        """

        key = f'{self.steps}'

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
            for field in results._fields:
                field_key = '/'.join([key, field])
                group[field_key] = getattr(results, field)

        # default case
        else:
            group[key] = results

        self.storage.flush()
