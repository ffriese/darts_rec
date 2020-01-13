import os
import shlex
import signal
import subprocess
from enum import Enum
from threading import Thread

import paho.mqtt.client as mqtt
import time
import psutil


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
            time.sleep(4)

    def on_shutdown(self, client, userdata, msg):
        print('got shutdown', msg.payload)
        if msg.payload.decode() == 'SHUTDOWN':
            os.system('sudo shutdown now')

    def on_cmd(self, client, userdata, msg):
        status = int(msg.payload)
        print('request STATUS: ', status)
        if status == 1 and self.status == 0:
            self.process.start()
            while self.process.status == ProcessStatus.STARTING:
                time.sleep(0.01)

        elif status == 0 and self.status == 1:
            self.process.stop()

        if ProcessStatus.STOPPED == self.process.status:
            self.status = 0
        elif ProcessStatus.RUNNING == self.process.status:
            self.status = 1

        self.client.publish('server_status', self.status, qos=2)

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))

        self.client.publish('server_status', self.status, qos=2)
        # Subscribing in on_connect() means that if we lose the connection and
        # reconnect then subscriptions will be renewed.
        client.subscribe("server_cmd")
        client.subscribe("rasp_shutdown")

    def stop(self):
        self.client.publish('server_status', 0, qos=0)
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
        self.keep_watching = False
        self.status = ProcessStatus.STOPPED

    def start(self):
        self.keep_watching = True
        self.watcher_thread = Thread(target=self.watch)
        self.watcher_thread.setDaemon(True)
        my_env = os.environ.copy()
        my_env["PYTHONPATH"] = ".."
        self.sub_proc = subprocess.Popen(shlex.split("python3 ../pipelines/rasp4.py"),
                                         stdout=subprocess.PIPE, env=my_env)
        self.status = ProcessStatus.STARTING
        self.watcher_thread.start()

    def stop(self):
        print('trying to stop...')
        print(self.sub_proc.send_signal(signal.SIGINT))
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

    def watch(self):
        while self.keep_watching:
            output = self.sub_proc.stdout.readline().decode().strip()
            poll = self.sub_proc.poll()
            if poll == 0:
                print(output)
                print('stopped!')
                self.status = ProcessStatus.STOPPED
                break
            if poll == 1:
                print(output)
                print('crashed!')
                self.status = ProcessStatus.CRASHED
                self.keep_watching = False
                break
            if 'complete' in output:
                print('started!')
                self.status = ProcessStatus.RUNNING
            if output:
                print(output)


if __name__ == '__main__':
    cont = ServerController()
    print('.....')
    try:
        cont.client.loop_forever()
    except KeyboardInterrupt:
        cont.stop()
        cont.client.disconnect()
        print("closed")
