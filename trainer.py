from collections import Sequence

import numpy as np
import torch as pt

from mixins import SaveMixin, TestSampleMixin, ValidationMixin, MonitorMixin, ToNumpyMixin


class Trainer(SaveMixin, TestSampleMixin, ValidationMixin, MonitorMixin, ToNumpyMixin):
    """class that implements the basic logic of training a model for streamlined training"""

    def __init__(self, model, criterion, optimizer, dataloader,
                 transformation=lambda x: x, loss_decay=0.95,
                 split_sample=None):
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
        self.transformation = transformation
        self.cuda = next(model.parameters()).is_cuda
        self.dtype = next(model.parameters()).dtype
        self.loss_decay = loss_decay
        self.epochs = 0
        self.steps = 0
        self.call_after_single_step = []
        self.split_sample = split_sample if callable(split_sample) else self._split_sample

        super().__init__()

        # unify backward call for all types of optimizer
        if not hasattr(self.optimizer, 'backward'):
            setattr(self.optimizer, 'backward', self._backward)

    def __del__(self):
        """make sure that storage is closed on deconstruction"""
        if hasattr(self, 'storage') and hasattr(self.storage, 'close'):
            self.storage.close()

    def train(self, *, n_epochs=None, n_steps=None):
        """
        train the model
        :param n_epochs: number of epochs to train
        :param n_steps: number of steps to train, overrides n_epochs
        """

        try:
            if n_epochs is None and n_steps is None:
                raise ValueError('either n_epochs or n_steps need to be specified')

            # n_steps overrides n_epochs
            if n_steps is not None:
                n_epochs = 2**32

            cumulative_loss = None
            for epoch in range(n_epochs):
                for step, sample in enumerate(self.dataloader):

                    try:
                        loss = self.single_step(sample)
                    except ValueError:
                        continue

                    # execute additional methods inherited from mixins
                    for method in self.call_after_single_step:
                        getattr(self, method)()

                    # update cumulative loss and print current progress
                    if cumulative_loss is None:
                        cumulative_loss = np.sum([loss])
                    cumulative_loss = round(self.loss_decay*cumulative_loss + (1-self.loss_decay) * np.sum([loss]), 6)
                    self._show_progress(epoch, step, n_epochs, n_steps, cumulative_loss)

                    # end training after n_steps if n_steps is set
                    self.steps += 1
                    if step == n_steps:
                        return

                self.epochs += 1

        finally:
            if hasattr(self.dataloader.dataset, 'close'):
                self.dataloader.dataset.close()
            if hasattr(self, 'save'):
                self.save(force=True)
            if hasattr(self, 'storage') and hasattr(self.storage, 'close'):
                self.storage.close()

    def _transform(self, sample):
        """
        applies transformation on a sample and matches its dtype and cuda status to that of the model
        :param sample: sample to transform
        """
        sample = self.transformation(sample)
        sample = self._cast(sample)
        inputs, targets = self.split_sample(sample)

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

    def _forward(self, sample):

        inputs, targets = self._transform(sample)
        outputs = self.model(inputs)
        loss = self.criterion(outputs, targets)

        return outputs, loss

    def single_step(self, sample):
        """
        implements the logic of a forward and backward pass of a single training step
        :param sample: sample from the dataloader
        :return: loss calculated on the current sample
        """

        outputs, loss = self._forward(sample)
        self.optimizer.zero_grad()
        self.optimizer.backward(loss)
        self.optimizer.step()

        return self._to_numpy(loss)

    def _show_progress(self, epoch, step, n_epochs, n_steps, loss):
        """
        print the current progress and loss
        :param epoch: current epoch
        :param step: current step
        :param n_epochs: number of epoch in training
        :param n_steps: number of steps in training
        :param loss: current loss
        """

        # step += 1
        steps_in_epoch = len(self.dataloader)
        n_steps = n_steps if n_steps is not None else n_epochs * steps_in_epoch - 1
        steps_taken = epoch * steps_in_epoch + step
        progress = round(100 * steps_taken / n_steps, 2)

        print(f'progress: {progress}%, epoch: {epoch}, step: {step}, loss: {loss}')

    @staticmethod
    def _split_sample(self, sample):
        """
        default function for splitting samples from the dataloader into model inputs and targets
        :param sample: sample from dataloader
        :return: sample[0] as inputs, sample[1] as targets
        """

        inputs, targets = sample
        return inputs, targets

    @staticmethod
    def _backward(loss):

        loss.backward()