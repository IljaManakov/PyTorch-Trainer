from torch.optim import Optimizer

from trainer.events import AbstractEventHandler
from trainer.mixins import SaveMixin


class EventSave(AbstractEventHandler):
    """
    Adapter for the save method that constructs checkpoint name from epoch and step
    and passes logdir as directory
    """

    def __init__(self, *, func=SaveMixin.save, name=None, interval=None):
        super().__init__(func, name=name, interval=interval)

    def __call__(self, trainer, key=None, loss=None, step=None, epoch=None):
        checkpoint_name = f'epoch-{epoch}' if step is None else f'epoch-{epoch}_step-{step}'
        save_config = SaveMixin.default_save_config().copy()
        save_config.pop(Optimizer)
        self.func(trainer, directory=trainer.logdir, checkpoint_name=checkpoint_name, save_config=save_config)
        return checkpoint_name


class TrainingLoss(AbstractEventHandler):
    """
    Adapter for recording training loss. Simply takes in the loss at each step and returns it.
    If registered as an event handler and monitored the loss will be saved to storage
    """

    def __init__(self, func=lambda x: x, name='training-loss', interval=None):
        super().__init__(func, name=name, interval=interval)

    def __call__(self, trainer, key=None, loss=None, step=None, epoch=None):
        return self.func(loss)

