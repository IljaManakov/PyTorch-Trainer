import shutil
import inspect
import importlib
from types import FunctionType, BuiltinFunctionType
import ast
from collections import Sequence


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
        self.parameters = list(kwargs.keys())
        super(Config, self).__init__(imports)
        self.file = file

    def __repr__(self):
        return self.dump()

    def __setattr__(self, key, value):
        """
        append names of new attrs to parameters
        """
        if key not in ('parameters', 'file', 'imports', 'aliases') and key not in self.parameters:
            self.parameters.append(key)
        return super(Config, self).__setattr__(key, value)

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
                      type: lambda x: self._parse_type(x),
                      FunctionType: lambda x: self._parse_callable_in_dict(x),
                      BuiltinFunctionType: lambda x: self._parse_callable_in_dict(x),
                      dict: lambda x: self._parse_dict(x)}
        default = lambda x: x

        # convert variables in the dictionary
        output = {key: conversion.get(type(value), default)(value) for key, value in dictionary.items()}

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
                      dict: lambda varname, var: f'{varname} = {self._parse_dict(var)}',
                      type: lambda varname, var: f'{varname} = {self._parse_type(var)}',
                      FunctionType: lambda varname, var: f'{inspect.getsource(var)}',
                      BuiltinFunctionType: lambda varname, var: f'{inspect.getsource(var)}'}
        default = lambda varname, var: f'{varname} = {var}'

        var = getattr(self, varname)

        return f'\n{conversion.get(type(var), default)(varname, var)}\n'

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