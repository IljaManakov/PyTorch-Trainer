from collections import Sequence
from functools import wraps

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
