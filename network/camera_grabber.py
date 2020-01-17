import platform
import uuid
from collections import OrderedDict, deque
from copy import deepcopy
from typing import List
from termcolor import colored

import cv2 as cv
import numpy as np
from pyv4l2.control import Control

from core.helper import ModuleParameter
from core.module import Module, Output, Thread, time, Input
from core.datatypes import CVImage, MultiImage, CollectionTrigger, JsonObject

CTRL_BACK_LIGHT_COMPENSATION = 9963804
CTRL_AUTO_WHITE_BALANCE = 9963788
CTRL_WHITE_BALANCE = 9963802
CTRL_EXPOSURE_MODE = 10094849
CTRL_AUTO_EXPOSURE_PRIORITY = 10094851
CTRL_EXPOSURE_MS = 10094850

OFF = 0
MANUAL_MODE = 1

EXPOSURE_IN_MILLISECONDS = 50


class CameraGrabber(Module):
    def __init__(self):
        super().__init__()

        self.images_out = Output(data_type=MultiImage, config_keys=['cam_ids'])
        self.frame_rate_out = Output(data_type=JsonObject)

        self.event_collection_thread = Thread(target=self.event_loop)
        self.event_collection_thread.setDaemon(True)

        self.frame_rate = 30.0
        self.resolution = (1920, 1080)
        self.running = False

        self.cam_ids = ModuleParameter(None, data_type=list, required=True)
        self.event_camera = ModuleParameter(1)
        self.cameras = OrderedDict()  # type: OrderedDict[int, Camera]

        self.collected_images = deque(maxlen=5)

    def configure(self, cam_ids: List[int] = None):
        self._configure(locals())
        self.images_out.emit_configuration({'cam_ids': self.cam_ids})

    def init_cameras(self):
        for cam_id in self.cam_ids:
            # self.log('Init Camera %s' % cam_id, level=None, simple_time_format=True)
            camera = Camera(cam_id, self.resolution[0], self.resolution[1], self.frame_rate)
            self.cameras[cam_id] = camera
            self.log('Camera %s %r, %f fps' % (cam_id, camera.resolution, camera.frame_rate),
                     level=None, simple_time_format=True)
            # self.cam_locks[camera] = Lock()

    def stabilize_brightness(self):
        threads = []
        self.running = False

        def stabilize(cam_id):
            # self.log(" CAM %s" % cam_id, level=None, simple_time_format=True)
            camera = self.cameras[cam_id]
            brightness = 120
            collected = 0
            camera.grabbing_paused = True
            while brightness > 90 or collected < 10:
                camera.control.set_control_value(CTRL_EXPOSURE_MS, EXPOSURE_IN_MILLISECONDS)
                ret, frame = camera.capture.read()
                if ret and frame is not None:
                    brightness = np.mean(frame)
                    self.log('Camera %s Brightness: %s' % (cam_id, brightness), level=None, simple_time_format=True)
                    collected += 1
                time.sleep(0.1)
            camera.grabbing_paused = False
            camera.brightness_stabilized = True
            self.log(colored('Camera %s Brightness stabilized' % cam_id, 'green'), level=None, simple_time_format=True)

        for camera_id in self.cam_ids:
            stabilizing_thread = Thread(target=stabilize, args=[camera_id])
            threads.append(stabilizing_thread)
            stabilizing_thread.start()

        for t in threads:
            t.join()
        self.log(colored("Brightness Stabilization complete!", 'green'), level=None, simple_time_format=True)
        self.running = True

    def event_loop(self):
        frame_count = 0
        total_sleep = [0 for _ in self.cameras]
        max_sleep = 0
        for camera in self.cameras.values():
            camera.request_frame(blocking=False)
        start_ts = time.time()
        collection_sleep = 1.0/float(list(self.cameras.values())[0].frame_rate)
        while self.running:
            images = []
            ts = time.time()
            frame_count += 1
            frame_id = str(uuid.uuid4())
            for camera in self.cameras.values():
                retrieval_start = time.time()
                while not camera.buffer:
                    time.sleep(0.001)
                single_sleep = time.time()-retrieval_start
                total_sleep[camera.cam_id] += single_sleep
                max_sleep = max(max_sleep, single_sleep)
                frame = camera.buffer.popleft()
                images.append(CVImage(frame, frame_id, {'name': camera.cam_id, 'ts': ts}))
            for camera in self.cameras.values():
                camera.request_frame(blocking=False)
            # event_image = images[0]
            self.collected_images.append(MultiImage(images))
            # self.event_image_out.data_ready(event_image)
            self.images_out.data_ready(MultiImage(images))

            done_ts = time.time()
            if done_ts > (start_ts + 1):
                frame_rate = round(float(frame_count) / (done_ts-start_ts), 1)
                mean_sleep = [round(float(t_s) / float(frame_count), 4) for t_s in total_sleep]
                cam_ret = [round(np.mean(c.retrieval_times), 4) for c in self.cameras.values()]
                self.log_debug('Framerate: %r' % frame_rate)
                self.frame_rate_out.data_ready(
                    JsonObject('{"fr":"%s", "s": "%s", "r":"%s"}' %
                               (frame_rate, mean_sleep, cam_ret), 'frame_rate'))
                start_ts = done_ts
                frame_count = 0
                total_sleep = [0 for _ in self.cameras]

            elapsed = time.time() - ts
            if elapsed < collection_sleep:
                time.sleep(collection_sleep - elapsed)

    def collect(self):
        return self.collected_images.pop().images

    def __start__(self):
        self.init_cameras()
        self.stabilize_brightness()
        self.event_collection_thread.start()
        for cam_id in self.cam_ids:
            self.cameras[cam_id].start()

    def __stop__(self):
        self.running = False
        self.event_collection_thread.join()
        for cam in self.cameras.values():
            cam.stop()


class Camera(object):

    KNOWN_IDS = {

            'iceberg':
                {
                    0: "/dev/v4l/by-id/usb-CNFFH37O394430025A10_Integrated_Webcam_HD-video-index0",
                    1: "/dev/v4l/by-id/usb-HD_Camera_Manufacturer_USB_2.0_Camera-video-index0"
                },
            'dartserver':
                {
                    0: "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-video-index0",
                    1: "/dev/v4l/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1:1.0-video-index0"
                }

    }

    def __init__(self, cam_id: int, width: int = 1920, height: int = 1080, fps: float = 60.0):
        self.cam_id = cam_id
        self.brightness_stabilized = False
        hostname = platform.uname()[1]
        capture_id = Camera.KNOWN_IDS[hostname][cam_id] if hostname in Camera.KNOWN_IDS else cam_id
        self.capture = cv.VideoCapture(capture_id)
        if hostname == 'iceberg':
            self.capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, width)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv.CAP_PROP_FPS, fps)
        self.frame_rate = self.capture.get(cv.CAP_PROP_FPS)
        self.resolution = (self.capture.get(cv.CAP_PROP_FRAME_WIDTH), self.capture.get(cv.CAP_PROP_FRAME_HEIGHT))
        # print('RESOLUTION: ', self.resolution)
        # print('FRAME_RATE: ', self.frame_rate)

        self.control = Control(capture_id)
        # self.control = Control("/dev/video%s" % cam_id)
        self.control.set_control_value(CTRL_BACK_LIGHT_COMPENSATION, OFF)
        self.control.set_control_value(CTRL_AUTO_WHITE_BALANCE, OFF)
        self.control.set_control_value(CTRL_WHITE_BALANCE, 2800)
        self.control.set_control_value(CTRL_EXPOSURE_MODE, MANUAL_MODE)
        self.control.set_control_value(CTRL_AUTO_EXPOSURE_PRIORITY, 0)
        self.control.set_control_value(CTRL_EXPOSURE_MS, EXPOSURE_IN_MILLISECONDS)
        # self.print_config()
        self.retrieval_jobs = deque()
        self.buffer = deque(maxlen=5)
        self.retrieval_times = deque(maxlen=10)
        self.running = True
        self.grabbing_paused = False
        self.grabber_thread = Thread(target=self.continuous_grab)
        self.grabber_thread.setDaemon(True)

    def start(self):
        self.grabber_thread.start()

    def stop(self):
        self.running = False
        self.grabber_thread.join()
        self.capture.release()

    def continuous_grab(self):
        while self.running:
            if self.retrieval_jobs:
                retrieval_start = time.time()
                ret, frame = self.capture.retrieve()
                retrieval_time = time.time()-retrieval_start
                self.retrieval_times.append(retrieval_time)
                self.buffer.append(frame)
                self.retrieval_jobs.popleft()
            elif not self.grabbing_paused:
                self.capture.grab()
            time.sleep(0.000001)

    def request_frame(self, blocking=True):
        self.retrieval_jobs.append(1)
        if blocking:
            while not self.buffer:
                time.sleep(0.0001)
            return self.buffer.popleft()

    def get_config(self):
        return self.control.get_controls()

    def print_config(self):
        print('CONTROLS:', self.control.get_controls())
        for c in self.control.get_controls():
            print(c)
