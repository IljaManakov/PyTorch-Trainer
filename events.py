from abc import ABC, abstractmethod

EACH_STEP = 'each_step'
EACH_EPOCH = 'each_epoch'
BEFORE_TRAINING = 'before_training'
AFTER_TRAINING = 'after_training'

events = [BEFORE_TRAINING, EACH_STEP, EACH_EPOCH, AFTER_TRAINING]


class TrainingEventConsumer(ABC):

    @abstractmethod
    def before_training(self):
        pass

    @abstractmethod
    def after_training(self):
        pass

    @abstractmethod
    def each_step(self, step, epoch, loss):
        pass

    @abstractmethod
    def each_epoch(self, epoch, loss):
        pass
