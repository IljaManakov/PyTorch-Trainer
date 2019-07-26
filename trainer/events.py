"""
events.py

defines the events that are known to the trainer

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

from abc import ABC, abstractmethod
from functools import partial
from trainer.utils import IntervalBased

EACH_STEP = 'each_step'
EACH_EPOCH = 'each_epoch'
BEFORE_TRAINING = 'before_training'
AFTER_TRAINING = 'after_training'

event_list = [BEFORE_TRAINING, EACH_STEP, EACH_EPOCH, AFTER_TRAINING]


class AbstractEventHandler(ABC):

    def __init__(self, func, *, name=None, interval=None, **kwargs):

        self.__name__ = func.__name__ if name is None else name

        # make handler interval based
        if interval:
            func = IntervalBased(interval)(func)

        # set defaults for handler
        if kwargs:
            handler = partial(func, **kwargs)

        self.func = func

    @abstractmethod
    def __call__(self, trainer, key=None, loss=None, step=None, epoch=None):

        return self.func(trainer, key=None, loss=None, step=None, epoch=None)
