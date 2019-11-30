import datetime
import logging
import os
import re
import sys
import cv2 as cv
from enum import Enum
from typing import Dict, Union, List

from termcolor import colored


class LogLevel(Enum):
    CRITICAL = 50
    FATAL = CRITICAL
    ERROR = 40
    WARNING = 30
    WARN = WARNING
    INFO = 20
    DEBUG = 10
    TRACE = 5
    NOTSET = 0


class Loggable(object):
    console_logger_level = logging.DEBUG
    file_logger_level = logging.DEBUG

    main_directory = os.path.join(os.path.dirname(__file__), '../..')

    @staticmethod
    def __print_msg__(msg, prefix='', level=logging.INFO, module_log_level=None):
        if level is None or level >= \
                (module_log_level if module_log_level is not None else Loggable.console_logger_level):
            sys.stdout.write('%s%s %s\n' % (Loggable.__get_padded_level_name(level), prefix, msg))

    @staticmethod
    def __get_padded_level_name(level):
        if level is None:
            return ''
        name = '[%s]' % logging.getLevelName(level).replace('WARNING', 'WARN')
        pad = 7 - len(name)
        if name == '[WARN]':
            name = colored(name, 'yellow')
        elif name == '[ERROR]' or name == '[FATAL]':
            name = colored(name, 'red')
        return '%s%s:' % (name, ' ' * pad)

    @staticmethod
    def __format_msg__(msg, *args, module_name='UNKNOWN', simple_time_format=False):
        msg = str(msg)
        for a in args:
            msg += ' %s' % str(a)

        prefix = '[%s][%s]' % (datetime.datetime.now().strftime("%H:%M:%S" if simple_time_format else "%H:%M:%S:%f"),
                               colored(module_name, attrs=['bold']))
        return prefix, msg

    @staticmethod
    def _log(msg, *args, level=logging.INFO, module_name='UNKNOWN', module_log_level=None, simple_time_format=False):
        prefix, message = Loggable.__format_msg__(msg, *args, module_name=module_name,
                                                  simple_time_format=simple_time_format)
        # de-colorize, remove line-breaks and multiple spaces:
        no_color_prefix = re.sub(r"\x1b\[\d+m", '', prefix)
        no_color_msg = re.sub(r"\x1b\[\d+m", '', re.sub(r' +|\n', ' ', message))

        if level is not None:
            if level >= Loggable.file_logger_level:
                logging.log(level, '%s %s' % (no_color_prefix, no_color_msg))
        else:
            logging.log(logging.DEBUG, '%s %s' % (no_color_prefix, no_color_msg))
        Loggable.__print_msg__(message, prefix=prefix, level=level, module_log_level=module_log_level)


class Property(object):
    def __init__(self, value, getter=None, setter=None):
        self.name = None
        self.object = None
        self.value = value
        self.getter = (lambda p, o: p.value) if getter is None else getter
        self.setter = (lambda p, o, v: setattr(p, 'value', v)) if setter is None else setter

    def __get__(self, obj, obj_type):
        return self.getter(self, obj)

    def __set__(self, obj, value):
        self.setter(self, obj, value)


class ModuleParameter(Property):
    def __init__(self, value, data_type=None, required: bool = True, getter=None, setter=None):
        super(ModuleParameter, self).__init__(value, getter, setter)
        self.data_type = data_type if data_type is not None else type(value)
        self.default_value = value
        self.required = required

    def __get__(self, obj, obj_type):
        return super().__get__(obj, obj_type)

    @staticmethod
    def _equals(one, other):
        eq = one == other
        if not isinstance(eq, bool):
            return eq.all()
        return one == other

    def __set__(self, obj, value):
        if isinstance(self.data_type, type) and isinstance(value, type):
            type_match = issubclass
        else:
            type_match = isinstance
        if not type_match(value, self.data_type):
            obj.log_warn('%r is not of data type %r but %r' % (value, self.data_type, type(value)))
            return
        if not self._equals(self.value, value):
            obj.log_info('updated', colored(self.name, 'blue', attrs=['bold']),
                         'to', colored(value, 'cyan', attrs=['bold']))
        else:
            obj.log_debug('kept', colored(self.name, 'blue', attrs=['bold']),
                          'at', colored(value, 'cyan', attrs=['bold']))
            return
        super().__set__(obj, value)

    def __str__(self):
        return 'ModuleParameter[%r: %r, (%s, default:%r)]' % (self.name, self.value,
                                                              self.data_type.__name__, self.default_value)


class Getter(Property):
    def __init__(self, getter, default_on_error=False, default_value=None):
        if default_on_error:
            def _getter(p, o):
                try:
                    return getter(p, o)
                except Exception as e:
                    print(e)
                    return default_value
        else:
            _getter = getter
        super().__init__(None, _getter, setter=lambda p, o, v: o.log_warn(colored('SETTING %s NOT SUPPORTED' %
                                                                                  self.name, 'red')))


class PropertyObject(object):

    def __setattr__(self, key, value):
        if isinstance(value, Property):
            self.log_debug('init', colored(key, 'blue', attrs=['bold']),
                           'to', colored(value.value, 'cyan', attrs=['bold']))
            value.name = key
            value.object = self

            cls = type(self)
            if not hasattr(cls, '__perinstance'):
                cls = type(cls.__name__, (cls,), {})
                cls.__perinstance = True
                self.__class__ = cls
            setattr(cls, key, value)
            if hasattr(self, '_properties'):
                getattr(self, '_properties').append(value)
            else:
                setattr(self, '_properties', [value])
        else:
            super.__setattr__(self, key, value)


def update_func(_o, _p, _c, _min_v, _max_v, _s):
    return lambda v, o=_o, p=_p, c=_c, max_v=_max_v, min_v=_min_v, s=_s: \
        setattr(o, '%s_%s' % (p, c), (v * ((max_v - min_v) / s)) + min_v)


def create_trackbar(obj, cam_ids: List[int], window_name: str, param_name: str,
                    cam_defaults: Dict[int, Union[int, float]], default: float = 0.0,
                    min_value: float = 0.0, max_value: float = 100.0, steps: int = 100):

    for cam in cam_ids:
        if not hasattr(obj, '%s_%s' % (param_name, cam)):
            setattr(obj, '%s_%s' % (param_name, cam), ModuleParameter(cam_defaults.get(cam, default)))

            cv.createTrackbar('%s %s' % (param_name, cam), window_name,
                              int((getattr(obj, '%s_%s' % (param_name, cam)) - min_value) * (steps / (max_value - min_value))),
                              int(steps),
                              update_func(obj, param_name, cam, min_value, max_value, steps)
                              )
