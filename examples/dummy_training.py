from trainer import Trainer, events, Config
from trainer.handlers import EventSave
from examples.dummy_components import MNIST
from torch.utils.data import DataLoader

config = Config.from_file('dummy_config.py')
trainer = Trainer.from_config(config, altered=True)  # altered is set as a test to see if Config.dump works
sample = next(iter(trainer.dataloader))
validation_loader = DataLoader(MNIST('validation.pt'))

# register handlers for events (see in trainer.events and trainer.handlers)
trainer.register_event_handler(events.EACH_STEP, trainer, name='inference', interval=10, sample=sample)
trainer.register_event_handler(events.EACH_EPOCH, trainer.validate, dataloader=validation_loader)
trainer.register_event_handler(events.EACH_EPOCH, EventSave())

trainer.monitor(name='model.activation')  # register monitoring for parts of the trainer. Return values will be stored

trainer.train(n_steps=4)
