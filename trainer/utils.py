from collections import Sequence
from functools import wraps
import shutil
import inspect
import importlib
from types import FunctionType, BuiltinFunctionType
import ast

import torch as pt
import torch.nn as nn
import torch.nn.init as init


def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """

    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)


def identity(x):
    return x


def to_numpy(sample):
    """
    convert individual and sequences of Tensors to numpy, while leaving strings, integers and floats unchanged
    :param sample: individual or sequence of Tensors, ints, floats or strings
    :return: same as sample but all Tensors are converted to numpy.ndarray
    """
    if isinstance(sample, int) or isinstance(sample, float) or isinstance(sample, str):
        return sample
    elif isinstance(sample, pt.Tensor):
        return sample.detach().cpu().numpy()
    elif isinstance(sample, tuple) or isinstance(sample, list):
        return [to_numpy(s) for s in sample]
    elif isinstance(sample, Sequence):
        collection = sample.__class__
        return collection(*[to_numpy(s) for s in sample])


class IntervalBased(object):
    """
    used for decorating functions.
    the function will then only be executed at set intervals
    """
    def __init__(self, interval):
        """
        :param interval: interval at which the function will be called
        """
        self.interval = interval
        self.counter = 0

    def __call__(self, func):
        """
        wraps function so that it is executed only at set interval
        :param func: function to wrap
        :return: wrapped function
        """
        @wraps(func)
        def wrapped(*args, **kwargs):
            self.counter += 1
            if self.counter % self.interval == 0:
                return func(*args, **kwargs)

        return wrapped


class Imports(object):
    """
    class for handling imported names in a python module
    """

    def __init__(self, imports):
        """
        extract imported names from source code string or another Imports instance
        :param imports: source code of a module of other instance of Imports
        """

        if isinstance(imports, str):
            self.imports, self.aliases = Imports._parse_source_string(imports)
        elif isinstance(imports, Imports):
            self.imports, self.aliases = imports.imports, imports.aliases
        else:
            raise ValueError('input has to be either string or Imports instance')

    @classmethod
    def from_file(cls, file):
        """
        initialize instance from .py config file
        :param file: filename of the config .py file
        :return: Config instance
        """

        spec = importlib.util.spec_from_file_location("config", file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        return cls.from_module(module)

    @classmethod
    def from_module(cls, module):
        """
        initialize Imports instance from module type
        :param module: python module
        :return: Imports instance
        """
        source = inspect.getsource(module)
        return cls(source)

    @staticmethod
    def _parse_source_string(import_string):
        """
        extract imports from a source code string using python's ast
        :param import_string: source code string
        :return: dict with imports where the names are the keys and their modules are the values
        """

        imports = []
        aliases = []
        for node in ast.iter_child_nodes(ast.parse(import_string)):
            name = Imports._parse_node(node)
            if name is not None:
                imports += [(n[0], n[1]) for n in name]
                aliases += [(n[2], n[0]) for n in name if n[2] is not None]

        return dict(imports), dict(aliases)

    @staticmethod
    def _parse_node(node, module=''):
        """
        extract imports from ast nodes recursively
        :param node: ast nodes or list of nodes
        :param module: module of the imported name, is handled automatically by recursive calls
        :return: tuple of imported name and its module name
        """
        if isinstance(node, Sequence):
            return [Imports._parse_node(n, module) for n in node]
        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            return Imports._parse_node(node.names, getattr(node, 'module', ''))
        elif isinstance(node, ast.alias):
            return node.name, module, node.asname

    def dump(self):
        """
        convert imports back to source code
        :return: python source code of the imports
        """

        text = ''
        for key, value in self.imports.items():
            if value != '':
                text += f'from {value} '
            text += f'import {key}\n'
        return text

    def is_imported(self, name):
        """
        check if name is imported
        :param name: name to check
        :return: bool
        """
        return name in self.imports.keys() or name in self.aliases.keys()


class Config(Imports):
    """
    class for handling .py config files
    """

    def __init__(self, imports='', file=None, **kwargs):
        """
        initialize Config instance, all kwargs become attributes of the instance
        :param imports: source code string or instance of Imports
        :param file: filename of the config module
        :param kwargs: parameters of the config
        """
        self.__dict__.update(kwargs.copy())
        self.parameters = kwargs.keys()
        super(Config, self).__init__(imports)
        self.file = file

    @classmethod
    def from_module(cls, module):
        """
        initialize instance from python module
        :param module: python module
        :return: Config instance
        """

        imports = super(Config, cls).from_module(module)
        varnames = [var for var in dir(module) if not var.startswith('__') and not imports.is_imported(var)]
        variables = {varname: getattr(module, varname) for varname in varnames}

        return cls(imports=imports, file=module.__file__, **variables)

    def _parse_dict(self, dictionary):
        """
        parse a dict to valid python code
        :param dictionary: dict object
        :return: parsed dict as string
        """

        # define conversion function for variable types
        conversion = {str: lambda x: f'\'{x}\'',
                      int: lambda x: x,
                      float: lambda x: x,
                      bool: lambda x: x,
                      type: lambda x: self._parse_type(x),
                      FunctionType: lambda x: self._parse_callable_in_dict(x),
                      BuiltinFunctionType: lambda x: self._parse_callable_in_dict(x)}

        # convert variables in the dictionary
        output = {key: conversion[type(value)](value) for key, value in dictionary.items()}

        # construct string representation of the dictionary
        if not output:
            return str(output)

        text = '{'

        for key, value in output.items():
            text += f'\'{key}\': {value},\n'

        return text[:-2] + '}'

    def _parse_callable_in_dict(self, func):
        """
        parse callable in a dict to valid python code
        :param func: callable
        :return: parsed callable as string
        """

        LAMBDA = lambda: None
        if func.__name__ == LAMBDA.__name__:
            code = inspect.getsource(func).rstrip(' \n,')
            return code[code.find('lambda'):]
        elif isinstance(func, FunctionType) or isinstance(func, BuiltinFunctionType):
            return func.__name__
        elif isinstance(func, type):
            return self._parse_type(func)
        else:
            raise ValueError(f'{type(func)} objects cannot be handled by the conversion function')

    def _parse_type(self, obj):
        """
        parse class instance to valid python code
        :param obj: class instance
        :return: parsed instance as string
        """

        if self.is_imported(obj.__name__):
            return obj.__name__
        else:
            return str(obj).replace('class', '').lstrip('< \'').rstrip('> \'')

    def var_to_source_code(self, varname):
        """
        convert a variable from the config to valid python code
        :param varname: name of the variable
        :return: parsed variable as string
        """

        conversion = {str: lambda varname, var: f'{varname} = \'{var}\'',
                      int: lambda varname, var: f'{varname} = {var}',
                      float: lambda varname, var: f'{varname} = {var}',
                      bool: lambda varname, var: f'{varname} = {var}',
                      dict: lambda varname, var: f'{varname} = {self._parse_dict(var)}',
                      type: lambda varname, var: f'{varname} = {self._parse_type(var)}',
                      FunctionType: lambda varname, var: f'{inspect.getsource(var)}',
                      BuiltinFunctionType: lambda varname, var: f'{inspect.getsource(var)}'}

        var = getattr(self, varname)

        return f'\n{conversion[type(var)](varname, var)}\n'

    def dump(self):
        """
        dump the variables of a config module to a string
        :return: parsed config as a string
        """

        text = super(Config, self).dump() + '\n'

        for parameter in self.parameters:
            text += self.var_to_source_code(parameter)

        return text

    def save(self, filename, copy_file=True):
        """
        save the config module to a file
        :param filename: name under which the config will be saved
        :param copy_file: if True, copy the original file of the module, otherwise dump module's variables to text
        :return: None
        """

        if copy_file and self.file != '':
            shutil.copy(self.file, filename)
        else:
            with open(filename, 'w') as file:
                file.write(self.dump())
                self.file = filename if self.file == '' else self.file
