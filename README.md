# Introduction
This package contains a Trainer class that streamlines the training of models and recording of results.
The Trainer class is designed in a modular way using Mixins.
This approach can be used to extend its capabilities beyond what it currently provides.
Additionally, the class makes use of an eventing pattern that allows users to register event handlers that will be executed at specified points in training.
The Trainer class can be found in trainer.py.
All of the Mixins are stored in mixins.py.
The module events.py contains the definitions of possible events and utils.py contains other miscellaneous code.

# Installation
simply install from PyPi using `pip install pt-trainer`

# Usage
Initialize a Trainer instance by passing a PyTorch model (inherited from nn.Module), PyTorch Dataloader instance, optimizer (PyTorch or apex) and a loss function that accepts the model prediction and targets and returns a loss tensor.
Alternatively, a Trainer can be created from a config file.
The config file should be another python file and contain the following variables:
- MODEL:        class of the model
- DATASET:      class of the dataset
- LOSS:         class of the loss function
- OPTIMIZER:    class of the optimizer
- LOGDIR:       path to the directory in which files generated by the trainer will be written
- model:        dict with kwargs for MODEL
- dataset:      dict with kwargs for DATASET
- dataloader:   dict with kwargs for the dataloader that will wrap DATASET
- loss:         dict with kwargs for LOSS
- optimizer:    dict with kwargs for OPTIMIZER
- trainer:      dict with kwargs for the Trainer class, such as split_sample

Optionally the APEX variable and apex dict can be specified to wrap the OPTIMIZER.

Once initialized, you can register event handlers using the method register_event_handler, specifying the handler and the event on which it will be called.
There are four possible events: before training, each step, each epoch and after training.

Training is then executed using the train method and passing either n_epochs or n_steps.

# Example
In the folder titled 'examples' I have set up a simple case of training a feed-forward neural net on a portion of MNIST.
This examples illustrates how to setup the config and how to use the trainer. Try running dummy_training.py if you want
to train the model.