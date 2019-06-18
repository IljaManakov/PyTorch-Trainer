from trainer import Trainer
from examples.dummy_components import MNIST
from torch.utils.data import DataLoader

trainer = Trainer.from_config('dummy_config.py')
trainer.setup_test_sample(sample=next(iter(trainer.dataloader)), model=trainer.model, criterion=trainer.criterion,
                          event='each_step', interval=10)
trainer.setup_validation(dataloader=DataLoader(MNIST('validation.pt')), forward_pass=trainer._forward)
trainer.setup_saving(directory=trainer.logdir)
trainer.setup_monitoring(trainer.logdir + '/storage.hdf5')

trainer.train(n_epochs=1)
