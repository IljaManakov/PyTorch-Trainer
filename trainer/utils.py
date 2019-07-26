from collections import Sequence, namedtuple
from functools import wraps

import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.init as init


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """

    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def identity(x):
    return x


def to_numpy(sample):
    """
    convert individual and sequences of Tensors to numpy, while leaving strings, integers and floats unchanged
    :param sample: individual or sequence of Tensors, ints, floats or strings
    :return: same as sample but all Tensors are converted to numpy.ndarray
    """
    if isinstance(sample, str):
        return sample
    elif isinstance(sample, pt.Tensor):
        return sample.detach().cpu().numpy()
    elif isinstance(sample, tuple) and hasattr(sample, '_fields'):  # namedtuple
        collection = sample.__class__
        return collection(*[to_numpy(s) for s in sample])
    elif isinstance(sample, Sequence):
        return [to_numpy(s) for s in sample]
    else:
        return sample


def set_seed(seed):
    pt.backends.cudnn.deterministic = True
    pt.backends.cudnn.benchmark = False
    np.random.seed(seed)
    pt.manual_seed(seed)
    pt.cuda.manual_seed_all(seed)


def test_on_sample(*args, sample, forward_pass, **kwargs):
    """
    perform inference on a sample
    :param sample: pt.Tensor on which inference is performed
    :param forward_pass: callable that returns (prediction, loss)
    :return: result of inference as namedtuple('prediction', 'loss')
    """

    with pt.no_grad():
        prediction, loss = to_numpy(forward_pass(sample))

    return namedtuple('test_on_sample', ('prediction', 'loss'))(prediction, loss)


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
            _, loss = forward_pass(sample)
            losses.append(to_numpy(loss))

    return np.stack(losses)


def show_progress(epoch, step, n_epochs, n_steps, steps_in_epoch, loss):
    """
    print the current progress and loss
    :param epoch: current epoch
    :param step: current step
    :param n_epochs: number of epoch in training
    :param n_steps: number of steps in training
    :param steps_in_epoch: number of steps in one epoch
    :param loss: current loss
    """

    n_steps = n_steps if n_steps is not None else n_epochs * steps_in_epoch - 1
    steps_taken = epoch * steps_in_epoch + step
    progress = round(100 * steps_taken / n_steps, 2)

    print(f'progress: {progress}%, epoch: {epoch}, step: {step}, loss: {loss}')


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
