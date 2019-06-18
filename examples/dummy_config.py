from examples.dummy_components import FeedForward, MNIST
import torch as pt

MODEL = FeedForward
DATASET = MNIST
LOSS = pt.nn.CrossEntropyLoss
OPTIMIZER = pt.optim.Adam
LOGDIR = './run1/'

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
    # trainer keyword arguments for split_sample or transformation go here
}

