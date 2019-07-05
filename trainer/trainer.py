#!/usr/bin/env python
# encoding: utf-8
"""
trainer.py

Implementation of a Trainer class for easy training of PyTorch models.

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

from collections import Sequence
from functools import partial
import os

import numpy as np
import torch as pt
from torch.utils.data import DataLoader

from trainer.mixins import SaveMixin, TestSampleMixin, ValidationMixin, MonitorMixin, CheckpointMixin
from trainer import events
from trainer.utils import weight_init, IntervalBased, Config


class Trainer(SaveMixin, TestSampleMixin, ValidationMixin, MonitorMixin, CheckpointMixin):
    """class that implements the basic logic of training a model for streamlined training"""

    def __init__(self, *, model, criterion, optimizer, dataloader, logdir='.', storage='storage.hdf5',
                 transformation=lambda x: x, loss_decay=0.95, split_sample=None):
        """
        initializes the Trainer object
        :param model: model implemented as a child of pt.nn.Module
        :param criterion: callable that returns loss which implements backward()
        :param optimizer: optimizer instance
        :param dataloader: dataloader instance
        :param transformation: callable that is applied to samples from dataloader at every training step,
               default: identity
        :param loss_decay: float that represents portion of previous loss that is kept for the loss update,
                           only relevant for printing at each training step,
                           default: 0.95
        """

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.logdir = logdir
        self.storage = self.open_storage(filename=os.path.join(logdir, storage))
        self.transformation = transformation
        self.cuda = next(model.parameters()).is_cuda
        self.dtype = next(model.parameters()).dtype
        self.loss_decay = loss_decay
        self.epochs = 0
        self.steps = 0
        self.call_after_single_step = []
        self.split_sample = split_sample if callable(split_sample) else self._split_sample
        self.events = {'before_training': [],
                       'each_step': [],
                       'each_epoch': [],
                       'after_training': [self.close_storage]}

        super().__init__()

        # unify backward call for all types of optimizer
        if not hasattr(self.optimizer, 'backward'):
            setattr(self.optimizer, 'backward', self._backward)

    def train(self, *, n_epochs=None, n_steps=None):
        """
        train the model
        :param n_epochs: number of epochs to train
        :param n_steps: number of steps to train, overrides n_epochs
        """

        try:
            for event in self.events[events.BEFORE_TRAINING]:
                event(self)

            if n_epochs is None and n_steps is None:
                raise ValueError('either n_epochs or n_steps need to be specified')

            # n_steps overrides n_epochs
            if n_steps is not None:
                n_epochs = 2**32

            cumulative_loss = None
            for epoch in range(n_epochs):
                for step, sample in enumerate(self.dataloader):

                    prediction, loss = self.forward_pass(sample)
                    self.backward_pass(loss)
                    loss = self.to_numpy(loss)

                    # update cumulative loss and print current progress
                    if cumulative_loss is None:
                        cumulative_loss = np.sum([loss])
                    cumulative_loss = round(self.loss_decay*cumulative_loss + (1-self.loss_decay) * np.sum([loss]), 6)
                    self.show_progress(epoch, step, n_epochs, n_steps, cumulative_loss)

                    # end training after n_steps if n_steps is set
                    self.steps += 1
                    if step == n_steps:
                        return

                    for event in self.events[events.EACH_STEP]:
                        event(self, key=(epoch, step))

                for event in self.events[events.EACH_EPOCH]:
                    event(self, key=(epoch, step))

                self.epochs += 1

        finally:
            if hasattr(self.dataloader.dataset, 'close'):
                self.dataloader.dataset.close()
            for event in self.events[events.AFTER_TRAINING]:
                event(self)

    def _transform(self, sample):
        """
        applies transformation on a sample and matches its dtype and cuda status to that of the model
        :param sample: sample to transform
        """
        sample = self.transformation(sample)
        inputs, targets = self.split_sample(sample)
        inputs = self._cast(inputs)

        return inputs, targets

    def _cast(self, sample):
        """
        matches dtype and cuda status of all Tensors in the sample to those of the model
        :param sample: sample to cast
        :return: cast sample
        """
        if isinstance(sample, pt.Tensor):
            sample = sample.type(self.dtype)
            sample = sample.cuda() if self.cuda else sample
            return sample
        elif isinstance(sample, str):
            return sample
        elif isinstance(sample, Sequence):
            return sample.__class__(([self._cast(s) for s in sample]))

    def forward_pass(self, sample):
        """
        forward pass including input transformation and splitting, model prediction and loss calculation
        :param sample: sample from the dataloader
        :return: tuple of model prediction and loss
        """

        inputs, targets = self._transform(sample)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        return outputs, loss

    def backward_pass(self, loss):
        """
        backward pass including resetting gradients, gradient calculation and optimizer step
        :param loss: loss tensor
        :return: None
        """

        self.optimizer.zero_grad()
        self.optimizer.backward(loss)
        self.optimizer.step()

    def show_progress(self, epoch, step, n_epochs, n_steps, loss):
        """
        print the current progress and loss
        :param epoch: current epoch
        :param step: current step
        :param n_epochs: number of epoch in training
        :param n_steps: number of steps in training
        :param loss: current loss
        """

        steps_in_epoch = len(self.dataloader)
        n_steps = n_steps if n_steps is not None else n_epochs * steps_in_epoch - 1
        steps_taken = epoch * steps_in_epoch + step
        progress = round(100 * steps_taken / n_steps, 2)

        print(f'progress: {progress}%, epoch: {epoch}, step: {step}, loss: {loss}')

    @staticmethod
    def _split_sample(sample):
        """
        default function for splitting samples from the dataloader into model inputs and targets
        :param sample: sample from dataloader
        :return: sample[0] as inputs, sample[1] as targets
        """

        inputs, targets = sample
        return inputs, targets

    @staticmethod
    def _backward(loss):
        """
        Calls backward on loss tensor. Used to unify the API for PyTorch and apex optimizers
        :param loss: loss tensor
        :return: None
        """

        loss.backward()

    def register_event_handler(self, event, handler, interval=None, monitor=True, name=None, **kwargs):
        """
        register a function that will be called on the specified event
        :param event: event on which the handler is called, should be one of the events in events.py
        :param handler: event handler
        :param interval: interval at which the handler should be run - only useful for the events each epoch or each
        step if you dont want to run the handler every step but e.g. every 10th step instead
        :param monitor: bool indicating whether the return value of the handler should be recorded, default True
        :param name: name under which the return values of the handler should be stored
        :param kwargs: keyword arguments for the handler
        :return:None
        """
        name = handler.__name__ if name is None else name

        # make handler interval based
        if interval:
            handler = IntervalBased(interval)(handler)

        # set defaults for handler
        if kwargs:
            handler = partial(handler, **kwargs)

        # wrap method for monitoring (has to be an attribute to get correct name)
        if monitor:
            handler = self.monitor(name=name, method=handler)

        self.events[event].append(handler)

    @classmethod
    def from_config(cls, config: Config, copy_config=True, altered=False):
        """
        initialize a trainer instance from a config
        :param config: name of the config file (.py file), should contain the variables MODEL, DATASET, LOSS, OPTIMIZER
        and LOGDIR and corresponding dicts of the same name in lower case, which specify keyword arguments.
        Additionally, a dict for dataloader and trainer is also needed.
        :param copy_config: bool indicating whether the config module should be copied to LOGDIR
        :param altered: indicates whether the config module's variable have been altered after import.
        if False, the original file of the module will be copied otherwise the module variables will be dumped.
        :return: trainer instance
        """

        parameters = ['LOGDIR', 'MODEL', 'DATASET', 'OPTIMIZER', 'LOSS',
                      'model', 'dataset', 'dataloader', 'optimizer', 'loss', 'trainer']
        verified = [hasattr(config, parameter) for parameter in parameters]
        assert all(verified), f'config must contain the following parameters: {parameters}'

        # initialize output directory
        if not os.path.isdir(config.LOGDIR):
            os.makedirs(config.LOGDIR)

        # initialize components
        model = config.MODEL(**config.model)
        model.apply(weight_init)
        dataset = config.DATASET(**config.dataset)
        dataloader = DataLoader(dataset, **config.dataloader)
        criterion = config.LOSS(**config.loss)
        optimizer = config.OPTIMIZER(model.parameters(), **config.optimizer)
        if hasattr(config, 'APEX'):
            optimizer = config.APEX(optimizer, **config.apex)
        logdir = config.LOGDIR

        # initialize Trainer instance
        trainer = cls(model=model, criterion=criterion, optimizer=optimizer,
                      dataloader=dataloader, logdir=logdir, **config.trainer)

        # save config
        if copy_config:
            config.save(os.path.join(logdir, 'config.py'), not altered)

        return trainer

    @classmethod
    def from_config_module(cls, config, copy_config=True, altered=False):
        """
        initialize a trainer instance from a config
        :param config: name of the config file (.py file), should contain the variables MODEL, DATASET, LOSS, OPTIMIZER
        and LOGDIR and corresponding dicts of the same name in lower case, which specify keyword arguments.
        Additionally, a dict for dataloader and trainer is also needed.
        :param copy_config: bool indicating whether the config module should be copied to LOGDIR
        :param altered: indicates whether the config module's variable have been altered after import.
        if False, the original file of the module will be copied otherwise the module variables will be dumped.
        :return: trainer instance
        """

        config = Config.from_module(config)
        trainer = cls.from_config(config, copy_config, altered)

        return trainer
        
    @classmethod
    def from_config_file(cls, config, copy_config=True):
        """
        initialize a trainer instance from a config
        :param config: name of the config file (.py file), should contain the variables MODEL, DATASET, LOSS, OPTIMIZER
        and LOGDIR and corresponding dicts of the same name in lower case, which specify keyword arguments.
        Additionally, a dict for dataloader is also needed.
        :param copy_config: bool indicating whether the config module should be copied to LOGDIR
        :return: trainer instance
        """

        config = Config.from_file(config)
        trainer = Trainer.from_config(config, copy_config)

        return trainer
