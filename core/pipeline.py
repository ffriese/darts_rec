import datetime
import logging
import numbers
import os
from argparse import ArgumentError

from termcolor import colored

from core.module import Module, Loggable, deque
from core.datatypes import CVImage


class Parameters(object):
    pass


class Parameter:
    def __init__(self, value_type, default_value, value_range=None, choices=None, comment='', advanced=False):
        self.value_type = value_type
        self.value = default_value
        self.advanced = advanced
        self.comment = comment
        self.choices = choices
        if issubclass(value_type, numbers.Number) and choices is None:
            self.value_range = value_range
        elif value_range is not None:
            raise ArgumentError("value_range can only be set for numeric types, not for %r" % value_type)

    def val(self):
        return self.value


class PipelineMeta(type):
    # noinspection PyProtectedMember
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance._update_modules()
        return instance


class Pipeline(Loggable, metaclass=PipelineMeta):

    IM_SHOWS = dict()

    def __init__(self):
        self.modules = []
        self.params = Parameters()

        self.log(colored('==============================INITIALIZING MODULES==============================',
                         'blue', attrs=['bold']))

    def log(self, msg, *args, level=logging.INFO, simple_time_format=False):
        Loggable._log(msg, *args, level=level,
                      module_name=colored('Pipeline: %s' % self.__class__.__name__, 'green'),
                      module_log_level=None, simple_time_format=simple_time_format)

    def _update_modules(self):
        self.modules = [mod for mod in locals()['self'].__dict__.values() if isinstance(mod, Module)]
        return self.modules

    @staticmethod
    def _connect_submodules():
        instances = sorted(Module.instances, key=lambda obj: obj.__startup_priority__(), reverse=True)
        for m in instances:
            m.__custom_connect__()


    @staticmethod
    def _configure_submodules():
        instances = sorted(Module.instances, key=lambda obj: obj.__startup_priority__(), reverse=True)
        for m in instances:
            m.__custom_configure__()

    def configure(self):
        """
            Configure the Modules used in this Pipeline
        """
        raise NotImplementedError("Please implement this method in subclass")

    def _connect(self):
        self.connect()
        # self._update_connections()

    def connect(self):
        """
            Define the Module-Connections
        """
        raise NotImplementedError("Please implement this method in subclass")

    def start(self, spin_kwargs=None):
        self.log(colored('===============================USER-CONFIGURATION===============================',
                         'blue', attrs=['bold']))
        self.configure()
        self.log(colored('===============================SUBMODULE_CONFIGURE==============================',
                         'blue', attrs=['bold']))
        self._configure_submodules()
        self.log(colored('================================SUBMODULE_CONNECT===============================',
                         'blue', attrs=['bold']))
        self._connect_submodules()
        self.log(colored('===================================CONNECTING===================================',
                         'blue', attrs=['bold']))
        self._connect()
        self.log(colored('====================================STARTING====================================',
                         'blue', attrs=['bold']))
        Module.__START_ALL__(connect_submodules=False, configure_submodules=False)
        self.log(colored('====================================RUNNING=====================================',
                         'blue', attrs=['bold']))
        Module.__SPIN__(**spin_kwargs if spin_kwargs is not None else {})

    def configure_global_logging(self, filename=None, filemode='w',
                                 file_log_level=logging.DEBUG,
                                 console_log_level=None,
                                 core_module_log=None):
        if filename is None:
            file_count = 0

            filename = os.path.join(Loggable.main_directory, 'logs',
                                    '%s_%s_' % (self.__class__.__name__,
                                                datetime.datetime.now().strftime('%Y-%m-%d')))
            while os.path.exists(filename+str(file_count)+'.log') is True:
                file_count += 1
            filename = filename+str(file_count)+'.log'
        else:
            if not os.path.isabs(filename):
                filename = os.path.join(Loggable.main_directory, 'logs', filename)
        logging._acquireLock()
        try:

            filehandler = logging.FileHandler(filename, filemode)
            formatter = logging.Formatter('[%(levelname)s] - %(message)s')
            filehandler.setFormatter(formatter)
            root_logger = logging.getLogger()  # root logger
            for handler in root_logger.handlers[:]:  # remove all old handlers
                root_logger.removeHandler(handler)
            root_logger.addHandler(filehandler)
            root_logger.setLevel(file_log_level)
        finally:
            logging._releaseLock()

        Loggable.file_logger_level = file_log_level
        if console_log_level is not None:
            Loggable.console_logger_level = console_log_level
        if core_module_log is not None:
            Module.enable_core_module_log = core_module_log
        self.log('Filename for logging: %s exists: %s' % (colored(os.path.abspath(filename), 'blue'),
                                                       os.path.exists(filename)))
        self.log('Console-Logger-Level set to %r' % logging.getLevelName(Loggable.console_logger_level))
        self.log('File-Logger-Level set to %r' % logging.getLevelName(Loggable.file_logger_level))

