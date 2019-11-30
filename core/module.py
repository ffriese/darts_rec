import datetime
import logging
import signal
import sys
import time
import os
import traceback

import numpy as np

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2 as cv
from collections import deque
from copy import deepcopy
from threading import Thread, Lock
from typing import List, Iterable, Type

from pylsl import local_clock
from termcolor import colored

from core.datatypes import RecognitionDataType, CVImage
from core.helper import Loggable, PropertyObject, ModuleParameter, Property


# noinspection PyProtectedMember
class ModuleMeta(type):
    # noinspection PyProtectedMember
    def __call__(cls, *args, **kwargs):
        instance = super().__call__(*args, **kwargs)
        instance.check_process_functions()
        return instance


# noinspection PyProtectedMember
class Module(PropertyObject, Loggable, metaclass=ModuleMeta):
    instances = []
    __SKIP_CONFIG__ = False
    enable_core_module_log = True
    __INTERRUPT_FLAG__ = False

    __IM_SHOWS = dict()
    __im_show_lock__ = Lock()

    def __init__(self,
                 module_name: str = None,
                 logger_level: int = None):
        self.module_logger_level = logger_level
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(filename='system_tmp.log', filemode='w', level=logging.DEBUG)

        if module_name is None:
            self.module_name = self.__class__.__name__
        else:
            self.module_name = module_name

        for module in Module.instances:
            if module.module_name == self.module_name:
                self.module_name = '%s_2' % self.module_name
        Module.instances.append(self)

        self._inputs = []
        self._outputs = []
        self._timing_handlers = {}

    def __setattr__(self, attribute, value):
        if isinstance(value, ConnectionNode):
            value.name = attribute
            if hasattr(self, attribute):
                self.log_warn(colored('WARNING. YOU ARE OVERWRITING A CONNECTION-NODE. '
                                      'THIS SHOULD NOT HAPPEN!', 'red'), attribute, value)
            super().__setattr__(attribute, value)
            if isinstance(value, Input):
                self._register_input(value, attribute)
            elif isinstance(value, Output):
                self._register_output(value, attribute)

            self.log_debug('registering',
                           colored('[%s]' % value.data_type.__name__, 'blue'), type(value).__name__,
                           colored(value.name, 'green' if isinstance(value, Input) else 'magenta'))
            return
        super().__setattr__(attribute, value)

    def _register_input(self, input_node, name):
        if getattr(self, name, None) != input_node:
            setattr(self, name, input_node)
        self._inputs.append(input_node)
        self._timing_handlers[name] = []
        input_node._initialize(self, name)

    def check_process_functions(self):
        for input_node in self._inputs:
            try:
                getattr(self, 'process_%s' % input_node.name)
            except AttributeError:
                if input_node._relay_connections:
                    self.log_trace('input function', colored('process_%s' % input_node.name, 'yellow'),
                                   'is unnecessary due to relay to',
                                   ','.join([colored('[%s]' % c.module.module_name, 'white', attrs=['bold']) + "." +
                                             colored(c.name, 'green') for c in input_node._relay_connections]))
                    setattr(self, 'process_%s' % input_node.name, lambda *args: None)

                else:
                    self.log_error(colored('ERROR: input function', 'red'),
                                   colored('process_%s' % input_node.name, 'yellow'),
                                   colored('is not implemented!', 'red'))
                    setattr(self, 'process_%s' % input_node.name,
                            lambda *args, in_name=input_node.name: self.log_warn(
                                colored('unimplemented function "process_%s" received %r' %
                                        (in_name, (*args,)), 'yellow')))
                    input('Press Enter to acknowledge and continue')

    def _register_output(self, output_node, name):
        try:
            old_node = getattr(self, name)
            identical = old_node == output_node
        except AttributeError:
            old_node = None
            identical = False

        if not identical:
            setattr(self, name, output_node)

        # self.log_warn('registering output', name)
        self._outputs.append(output_node)
        output_node._initialize(self, name)

        if old_node is not None and not identical:
            self.log_warn('overwriting node', name)
            for conn in old_node._registered_connections:
                output_node.connect(conn)
            self._outputs.remove(old_node)

    def input_worker_loop(self, input_node: 'Input'):
        self.log(colored('"%s"-worker started' % input_node.name, 'blue', attrs=['bold']),
                 level=None, simple_time_format=True)
        while input_node.is_working():
            result, timing = input_node.process_item()
            if not result:
                # cv.waitKey(int(input_node.sleep_time*1000))
                time.sleep(input_node.sleep_time)
            else:
                self._publish_timing(timing, input_node)

        self.log(colored('"%s"-worker stopped gracefully' % input_node.name, 'magenta', attrs=['bold']),
                 level=None, simple_time_format=True)

    def _publish_timing(self, timing, input_node):
        for h in self._timing_handlers[input_node.name]:
            h(self.module_name, timing, input_node.get_queue_size())

    def register_timing_handler(self, handler, input_node_name):
        """
        Registers a function h(timing: int, input_name: string, queue_size: int) that will receive the timing of each incoming sample
        :param handler: a function h(timing: int, input_name: string, queue_size: int)
        """
        self._timing_handlers[input_node_name].append(handler)

    def _configure(self, config: dict, **kwargs):
        try:
            config.pop('self')
        except KeyError:
            pass
        changed = False
        for key, value in config.items():
            if not hasattr(self, key):
                self.log_error(colored('ERROR!!!!!!!!!! CANNOT CONFIGURE PARAM %s' % key, 'red'))

            if value is not None and value != getattr(self, key, None):
                self.log_debug('%s requests %s update to %s' % (
                    colored('[%s]' % kwargs.get('_sender_module', 'user'), 'magenta', attrs=['bold']),
                    colored(key, 'blue'),
                    colored(value, 'cyan')))
                setattr(self, key, value)
                changed = True

        if changed:
            for output in self._outputs:
                output.emit_configuration()

    def __activate_threads__(self):
        for input_node in self._inputs:
            # noinspection PyProtectedMember
            input_node._activate_thread()

    def __deactivate_threads__(self):
        for input_node in self._inputs:
            input_node._deactivate_thread()

    def log(self, msg, *args, level=logging.INFO, simple_time_format=False):
        Module._log(msg, *args, level=level, module_name=self.module_name,
                    module_log_level=self.module_logger_level, simple_time_format=simple_time_format)

    def log_trace(self, msg, *args):
        self.log(msg, *args, level=logging.NOTSET)

    def log_debug(self, msg, *args):
        self.log(msg, *args, level=logging.DEBUG)

    def log_info(self, msg, *args):
        self.log(msg, *args, level=logging.INFO)

    def log_warn(self, msg, *args):
        self.log(msg, *args, level=logging.WARN)

    def log_error(self, msg, *args):
        self.log(msg, *args, level=logging.ERROR)

    def log_fatal(self, msg, *args):
        self.log(msg, *args, level=logging.FATAL)

    @staticmethod
    def __core_log__(msg, color='blue', sender='MODULE CORE', sender_color='blue'):
        if not Module.enable_core_module_log:
            return

        upper = lower = '─' * (12 + len(sender) + 2 + len(msg))
        upper += '┐'
        msg = '%s' % (colored('%s │' % msg, color, attrs=['bold']))
        lower += '┘'
        print(colored(upper, color))
        Loggable._log(msg, level=None, module_name=colored(sender, sender_color))
        print(colored(lower, color))

    # overwrite if priority is higher / lower
    @staticmethod
    def __shutdown_priority__():
        return 0

    @staticmethod
    def __startup_priority__():
        return 0

    def __pre_start__(self):
        for node in [getattr(self, entry) for entry in self.__dir__()
                     if isinstance(getattr(self, entry), Input) or isinstance(getattr(self, entry), Output)]:
            if not node.is_connected():
                self.log(colored('WARNING!', 'red'), 'node %s is not connected!' %
                         colored(node.name, 'red', attrs=['bold']),
                         level=None,
                         simple_time_format=True)

        if hasattr(self, '_properties'):
            for param in [prop for prop in getattr(self, '_properties')
                          if isinstance(prop, Property)]:

                if isinstance(param, ModuleParameter) and param.required and param.value is None:
                    self.log('%s is NONE, but required to be set!' %
                             (colored(param.name, 'red', attrs=['bold'])),
                             level=None, simple_time_format=True)
                else:
                    self.log('%s is initialized with %s' %
                             (colored(param.name, 'blue', attrs=['bold']),
                              colored(param.__get__(self, type(self)), 'cyan', attrs=['bold'])),
                             level=None, simple_time_format=True)

    def __start__(self):
        pass

    def __stop__(self):
        pass

    def __custom_cleanup__(self):
        pass

    def __custom_connect__(self):
        pass

    def __custom_configure__(self):
        pass

    @staticmethod
    def __START_ALL__(connect_submodules=True, configure_submodules=True):
        instances = sorted(Module.instances, key=lambda obj: obj.__startup_priority__(), reverse=True)
        if configure_submodules:
            Module.__core_log__('... configuring submodules ...', 'green')
            for m in instances:
                m.__custom_configure__()
        if connect_submodules:
            Module.__core_log__('... connecting submodules ...', 'green')
            for m in instances:
                m.__custom_connect__()

        Module.__core_log__('... starting up Modules ...', 'green')

        for m in instances:
            upper = lower = '─' * (11 + len(m.module_name))
            sys.stdout.write(colored(upper + '┐\n', 'green'))
            sys.stdout.write(colored('[%s][%s] starting Module\n' % (datetime.datetime.now().strftime("%H:%M:%S"),
                                                                     m.module_name), 'green'))
            m.__pre_start__()
            m.__activate_threads__()
            m.__start__()

            sys.stdout.write(
                colored('[%s][%s] Module started successfully\n' % (datetime.datetime.now().strftime("%H:%M:%S"),
                                                                    m.module_name), 'green'))
            sys.stdout.write(colored(lower + '┘\n', 'green'))
        Module.__core_log__('... Modules started ...')

    @staticmethod
    def __CLEANUP__(silent=False):

        if not silent:
            Module.__core_log__('... cleaning up Modules ...')
        instances = sorted(Module.instances, key=lambda obj: obj.__shutdown_priority__(), reverse=True)

        for m in instances:
            if not silent:
                upper = lower = '─' * (11 + len(m.module_name))
                print(colored(upper + '┐', 'red'))
                print(colored('[%s][%s] shutting down Module ...' % (datetime.datetime.now().strftime("%H:%M:%S"),
                                                            m.module_name), 'red'))
            m.__stop__()
            m.__deactivate_threads__()
            m.__custom_cleanup__()
            for input_node in m._inputs:
                input_node._join_thread()
            if not silent:
                print(colored('[%s][%s] Module shut down successfully' % (datetime.datetime.now().strftime("%H:%M:%S"),
                                                                          m.module_name), 'green'))
                print(colored(lower + '┘', 'red'))

        Module.instances = []
        if not silent:
            Module.__core_log__('SHUT DOWN CLEANLY', 'green')

    @staticmethod
    def __SPIN__(timeout_in_seconds=None, silent=False, exit_condition=None):
        if exit_condition is None:

            if timeout_in_seconds:
                start_time = time.time()

                def condition():
                    return time.time() < start_time + timeout_in_seconds
            else:
                def condition():
                    return True
        else:
            def condition():
                return not exit_condition()
        try:
            while condition() and not Module.__INTERRUPT_FLAG__:
                with Module.__im_show_lock__:
                    for frame in Module.__IM_SHOWS.keys():
                        if not [1 for cam in [c for c in Module.__IM_SHOWS[frame] if c != 'axis'] if not Module.__IM_SHOWS[frame][cam]]:
                            ims = []
                            for cam in [c for c in Module.__IM_SHOWS[frame] if c != 'axis']:
                                ims.append(Module.__IM_SHOWS[frame][cam].popleft())
                            try:
                                cv.imshow(frame, np.concatenate(ims, axis=Module.__IM_SHOWS[frame]['axis']))
                            except ValueError:
                                print(frame, [im.shape for im in ims])
                cv.waitKey(1)
        except KeyboardInterrupt:
            pass
        try:
            Module.__CLEANUP__(silent=silent)
        except KeyboardInterrupt:
            print(colored('FATAL! FAILED TO CLEANUP CODE CORRECTLY.'))
            sys.exit(1)

    @staticmethod
    def __INTERRUPT__():
        Module.__INTERRUPT_FLAG__ = True

    @staticmethod
    def show_image(frame_name: str, image: CVImage, axis=0):
        with Module.__im_show_lock__:
            if frame_name not in Module.__IM_SHOWS:
                Module.__IM_SHOWS[frame_name] = {}
            if image.camera_info['name'] not in Module.__IM_SHOWS[frame_name]:
                Module.__IM_SHOWS[frame_name][image.camera_info['name']] = deque(maxlen=10)
            Module.__IM_SHOWS[frame_name][image.camera_info['name']].append(image)
            Module.__IM_SHOWS[frame_name]['axis'] = axis


class ConnectionNode(object):
    def __init__(self, data_type: Type[RecognitionDataType], config_keys: List[str]):
        self.module = None
        self.name = None
        self.data_type = data_type
        self._registered_connections = []
        self._relay_connections = []

        if config_keys is None:
            self.config_keys = []
        else:
            self.config_keys = config_keys

        for key in self.config_keys:
            setattr(self, key, None)

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, data_type):
        if self.module is not None:
            self.module.log_info('changed datatype of', colored(self.name, 'yellow', attrs=['bold']),
                                 'to', colored(data_type, 'cyan', attrs=['bold']))
        self._data_type = data_type

    def _initialize(self, module: 'Module', name: str):
        self.module = module
        self.name = name

    def is_initialized(self):
        return self.module is not None

    def is_connected(self):
        return (self._registered_connections.__len__() + self._relay_connections.__len__()) > 0

    def __str__(self):
        return '<[%s] %s.%s: %r>' % (self.__class__.__name__,
                                     self.module.module_name if self.module is not None else 'NONE',
                                     self.name,
                                     self.data_type)


class DataStream(object):
    pass


class Output(ConnectionNode):
    """
       A Generic Module Output Node
    """
    def __init__(self, data_type: Type[RecognitionDataType], config_keys: List[str] = None):
        super().__init__(data_type, config_keys)

    def connect(self, input_connection: 'Input'):
        """

        :param input_connection:
        """
        if not isinstance(input_connection, Input):
            raise Exception('cannot connect Output %s.%s to Object of type %r' % (self.module.module_name, self.name,
                                                                                  type(input_connection).__name__))
        if not issubclass(self.data_type, input_connection.data_type):
            raise Exception('cannot connect Output %r of type %r to %s %r of type %r' % (self.name,
                                                                                         self.data_type,
                                                                                         type(input_connection).
                                                                                         __name__,
                                                                                         input_connection.name,
                                                                                         input_connection.data_type
                                                                                         ))
        self._registered_connections.append(input_connection)
        if not Module.__SKIP_CONFIG__:
            # print(Module.__SKIP_CONFIG__, 'configuring')

            self.module.log_info('connected %s %s  %s %s.%s' % (colored(self.data_type.__name__, 'cyan'),
                                                                colored(self.name, 'magenta'),
                                                                colored('=>', 'green'),
                                                                colored('[%s]' % input_connection.module.module_name,
                                                                        attrs=['bold']),
                                                                colored(input_connection.name, 'green')))

        input_connection.module._configure({k: getattr(self, k) for k in self.config_keys},
                                           _sender_module=self.module.module_name,
                                           _sender_node=self)
        input_connection._register_connection(self)

        # TODO: think about multi-level-relays!
        for relay in self._relay_connections:
            relay.connect(input_connection)

        for relay in input_connection._relay_connections:
            self.connect(relay)

    def emit_configuration(self, update=None):
        # if update is not specifically set, update local parameter copy from module
        if update is None:
            update = {key: getattr(self.module, key, None) for key in self.config_keys}
        for key, value in update.items():
            setattr(self, key, value)
        # self.module.log_warn('emitting config', update)
        for connection in self._registered_connections:
            # self.module.log_warn('emitting config', update, 'to', connection)
            connection.module._configure(update,
                                         _sender_module=self.module.module_name,
                                         _sender_node=self)

    def data_ready(self, data: RecognitionDataType):
        """
        relay data to connected modules
        :param data: Data to be relayed
        :return: None
        """
        # if self.data_type not in [DataStream]:
        #     self.module.log_debug(self.name+' -> '+','.join([c.module.module_name+'.'+c.name for c in
        #                                                      self._registered_connections]))
        for connection in self._registered_connections:
            connection.add_to_data_queue(deepcopy(data))

    def relay(self, output: 'Output'):
        self._relay_connections.append(output)
        for connection in self._registered_connections:
            output.connect(connection)


class Input(ConnectionNode):
    def __init__(self, data_type: Type[RecognitionDataType], config_keys: List[str] = None):
        super().__init__(data_type, config_keys)
        self._worker_thread = None
        self._data_queue = deque()
        self._working = True
        self.sleep_time = 0.005

    def _initialize(self, module: 'Module', name: str):
        super(Input, self)._initialize(module, name)

    def _register_connection(self, connection: 'Output'):
        if not isinstance(connection, Output):
            raise Exception('cannot connect Object of type %r to Input %s.%s' % (type(connection),
                                                                                 self.module.module_name, self.name))
        self._registered_connections.append(connection)

    def relay(self, input:'Input'):
        self._relay_connections.append(input)
        for connection in self._registered_connections:
            connection.connect(input)

    def is_working(self):
        return self._working

    def add_to_data_queue(self, item: RecognitionDataType):
        self._data_queue.append(item)

    def extend_data_queue(self, item_list: Iterable[RecognitionDataType]):
        self._data_queue.extend(item_list)

    def get_queue_size(self):
        return len(self._data_queue)

    def process_item(self):
        start_time = local_clock()
        if self._data_queue:
            next_job = self._data_queue.popleft()
            if type(next_job) != self.data_type:
                self.module.log_warn(colored('WARNING:', 'yellow'), 'input', colored('"%s"' % self.name, 'blue'),
                                     'expects', colored(self.data_type, 'blue'), 'but the queue held',
                                     colored(type(next_job), 'red'))
            try:
                getattr(self.module, 'process_%s' % self.name)(next_job)
            except Exception as e:
                self.module.log_error('MODULE STOPPED WITH EXCEPTION:', type(e), e, "\n SHUTTING DOWN!")
                self.module.log_error('EXCEPTION INFO:', traceback.format_exc())
                os.kill(os.getpid(), signal.SIGINT)
            proc_time = local_clock() - start_time
            return True, proc_time
        else:
            proc_time = local_clock() - start_time
            return False, proc_time

    def _activate_thread(self):
        # re-new old threads
        if not self._working or self._worker_thread is None:
            self._working = True
            self._worker_thread = Thread(target=self.module.input_worker_loop, args=[self])
            self._worker_thread.daemon = True
        self._worker_thread.start()

    def _deactivate_thread(self):
        self._working = False

    def _join_thread(self):
        if self._worker_thread.isAlive():
            self._worker_thread.join()
        else:
            self.module.log(colored('"%s"-worker was not running' % self.name, 'white', attrs=['bold']), level=None,
                            simple_time_format=True)

    def is_configured(self):
        for key in self.config_keys:
            if getattr(self, key) is None:
                return False
        return True


class Relay(ConnectionNode):
    def __init__(self, relay_node):
        self._relay_node = relay_node
