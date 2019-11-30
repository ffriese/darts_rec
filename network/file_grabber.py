import pickle
import uuid

import numpy as np

from core.helper import ModuleParameter
from core.module import Module, Output, os, Thread, time
from core.datatypes import CVImage


class FileGrabber(Module):
    def __init__(self):
        super().__init__()
        self.image_out = Output(data_type=CVImage, config_keys=['cam_ids'])
        self.reader_thread = Thread(target=self.collect)
        self.reader_thread.setDaemon(True)
        self.frame_rate = 4.0
        self.data = None
        self.running = False
        self.data = []

        self.cam_ids = ModuleParameter([0, 1], data_type=list)
        self.image_out.emit_configuration({'cam_ids': self.cam_ids})

    def collect(self):
        while self.running:
            for f in self.data:
                if self.running:
                    self.image_out.data_ready(f)
                    time.sleep(1.0/self.frame_rate)
                else:
                    break

    def __start__(self):
        with open('IMAGES', 'rb') as bg_file:
            data = pickle.load(bg_file)

            self.log_debug('loading images...')
            self.data = []
            for f in data:
                image = f['image']
                image.id = f['id']
                image.camera_info = f['camera_info']
                self.data.append(image)

        self.log_debug('loaded', len(self.data), 'images from file')
        self.running = True
        self.reader_thread.start()

    def __stop__(self):
        self.running = False

