import os
import shlex
import signal
import subprocess
from collections import deque
from enum import Enum
from threading import Thread

import paho.mqtt.client as mqtt
import time
import psutil
from termcolor import colored


class ServerController(object):
    def __init__(self):
        self.client = mqtt.Client()
        self.client.on_connect = self.on_connect
        self.client.message_callback_add("server_cmd", self.on_cmd)
        self.client.message_callback_add("rasp_shutdown", self.on_shutdown)
        self.client.connect("dartserver", 1883, 60)
        print('connecting')
        self.process = Proc()
        self.status = 0
        self.running = True
        self.perf_thread = Thread(target=self.perf)
        self.perf_thread.setDaemon(True)
        self.perf_thread.start()

    def perf(self):
        while self.running:
            # gives a single float value
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory()._asdict()['percent']
            data = '{"cpu": "%s", "mem": "%s"}' % (cpu, mem)

            self.client.publish('rasp_perf', data)
            for _ in range(4):
                if self.status == 1 and self.process.status == ProcessStatus.CRASHED:
                    self.status = 0
                    self.publish_status()
                    self.process.kill()
                while self.process.errors:
                    self.publish_error(self.process.errors.popleft())
                time.sleep(1)

    def on_shutdown(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
        print('got shutdown', msg.payload)
        if self.process.status == ProcessStatus.RUNNING or self.process.status == ProcessStatus.STARTING:
            self.process.stop()
        self.status = 0
        self.publish_status()
        if msg.payload.decode() == 'SHUTDOWN':
            os.system('sudo shutdown now')

    def on_cmd(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
        status = int(msg.payload)
        print('request STATUS: ', status)
        set_status = None

        if (0 < status < 3) and self.status == 0:
            pipeline = ['recognize_darts', 'recalibrate'][status-1]
            self.publish_error('')
            self.process.start(pipeline=pipeline)
            while self.process.status == ProcessStatus.STARTING:
                time.sleep(0.01)
            set_status = status

        elif status == 0 and self.status > 0:
            self.process.stop()

        if ProcessStatus.STOPPED == self.process.status:
            self.status = 0
        elif ProcessStatus.RUNNING == self.process.status\
                and status == set_status:
            self.status = status
        self.publish_status()
        print('Status:', status, self.process.status)

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))

        self.publish_status()
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("server_cmd")
        client.subscribe("rasp_shutdown")

    def publish_status(self):
        self.client.publish('server_status', self.status, qos=2, retain=True)

    def publish_error(self, err):
        self.client.publish('server_error', err, qos=2)

    def stop(self):
        self.status = 0
        self.publish_status()
        self.process.kill()


class ProcessStatus(Enum):
    STOPPED = 0
    STARTING = 1
    RUNNING = 2
    CRASHED = 3


class Proc(object):
    def __init__(self):
        self.sub_proc = None
        self.watcher_thread = None
        self.std_out_watcher = None
        self.std_err_watcher = None
        self.keep_watching = False
        self.status = ProcessStatus.STOPPED
        self.errors = deque(maxlen=20)

    def start(self, pipeline='recognize_darts'):
        self.keep_watching = True
        self.watcher_thread = Thread(target=self.watch)
        self.watcher_thread.setDaemon(True)

        self.std_out_watcher = Thread(target=self.watch_std_out)
        self.std_out_watcher.setDaemon(True)

        self.std_err_watcher = Thread(target=self.watch_std_err)
        self.std_err_watcher.setDaemon(True)

        my_env = os.environ.copy()
        my_env["PYTHONPATH"] = ".."
        self.sub_proc = subprocess.Popen(shlex.split("python3 ../pipelines/%s.py" % pipeline),
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=my_env)
        self.status = ProcessStatus.STARTING
        self.watcher_thread.start()
        self.std_out_watcher.start()
        self.std_err_watcher.start()
        self.errors = deque(maxlen=20)

    def stop(self):
        print('trying to stop...')
        self.sub_proc.send_signal(signal.SIGINT)
        while self.status == ProcessStatus.RUNNING:
            time.sleep(0.01)
        self.keep_watching = False

    def kill(self):
        self.keep_watching = False
        if self.sub_proc is not None:
            try:
                self.sub_proc.kill()
            except Exception as e:
                print(e)

    def watch_std_out(self):
        while self.keep_watching:
            try:
                with self.sub_proc.stdout:
                    for line in iter(self.sub_proc.stdout.readline, b''):
                        output = line.decode().rstrip('\n')
                        print('#', output)
                        if 'complete' in output:
                            print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> started!')
                            self.status = ProcessStatus.RUNNING
            except ValueError:
                pass
            time.sleep(0.001)

    def watch_std_err(self):
        while self.keep_watching:
            try:
                with self.sub_proc.stderr:
                    for line in iter(self.sub_proc.stderr.readline, b''):
                        err = line.decode().rstrip('\n')
                        print(colored('X %s' % err, 'red'))
                        if 'shutting down' in err.lower():
                            self.errors.append(err)
                            print('########################crashed 2!############################')
                            self.status = ProcessStatus.CRASHED
                            # self.keep_watching = False
            except ValueError:
                pass
            time.sleep(0.001)

    def watch(self):
        while self.keep_watching:
            poll = self.sub_proc.poll()
            if poll == 0:
                print('>>>>>>>>>>>>  stopped!')
                self.status = ProcessStatus.STOPPED
                break
            if poll == 1:
                print('########################crashed 1!############################')
                self.status = ProcessStatus.CRASHED
                self.keep_watching = False
            time.sleep(0.01)
        if self.status == ProcessStatus.CRASHED:
            self.sub_proc.kill()



if __name__ == '__main__':
    cont = ServerController()
    print('.....')
    try:
        cont.client.loop_forever()
    except KeyboardInterrupt:
        cont.stop()
        cont.client.disconnect()
        print("closed")
