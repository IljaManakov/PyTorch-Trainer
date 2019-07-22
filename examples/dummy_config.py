from examples.dummy_components import FeedForward, MNIST
import torch as pt

MODEL = FeedForward
DATASET = MNIST
LOSS = pt.nn.CrossEntropyLoss
OPTIMIZER = pt.optim.Adam
LOGDIR = './run1'

model = {
    # model keyword arguments go here
}

dataset = {
    'filename': 'train.pt'
}

dataloader = {
    'batch_size': 32
}

loss = {
    # loss keyword arguments go here
}

optimizer = {
    'lr': 0.1
}

trainer = {
    'storage': 'storage.hdf5',
    'split_sample': lambda x: (x[0], x[1]),
    'transformation': lambda x: x,
    'loss_decay': 0.95
}

cuda = True
dtype = pt.float32
seed = 0
