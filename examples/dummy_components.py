import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class FeedForward(nn.Module):

    def __init__(self):

        super().__init__()

        self.layers = nn.ModuleList([nn.Linear(*args) for args in [(784, 128), (128, 32), (32, 10)]])
        self.activation = F.relu

    def forward(self, x):

        out = x
        for layer in self.layers[:-1]:
            out = self.activation(layer(out))

        out = self.layers[-1](out)

        return out


class MNIST(Dataset):

    def __init__(self, filename):

        self.data, self.labels = pt.load(filename)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, item):

        return self.data[item], self.labels[item].long()
