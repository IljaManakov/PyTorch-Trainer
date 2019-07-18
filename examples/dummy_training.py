from trainer import Trainer, events
from examples.dummy_components import MNIST
from torch.utils.data import DataLoader

trainer = Trainer.from_config_file('dummy_config.py')
sample = trainer._transform(next(iter(trainer.dataloader)))
validation_loader = DataLoader(MNIST('validation.pt'))

trainer.register_event_handler(events.EACH_STEP, trainer.test_on_sample, interval=10,
                               sample=sample, model=trainer.model, criterion=trainer.criterion)
trainer.register_event_handler(events.EACH_EPOCH, trainer.validate,
                               dataloader=validation_loader, forward_pass=trainer.forward_pass)
trainer.register_event_handler(events.EACH_EPOCH, trainer.save,
                               directory=trainer.logdir)
trainer.monitor(name='criterion')

trainer.train(n_epochs=1)
