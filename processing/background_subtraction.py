import numpy as np
import pickle

from core.helper import ModuleParameter, create_trackbar
from core.module import Module, Input, Output
from core.datatypes import CVImage, SetBackgroundTrigger
from core.convenience import resize
import cv2 as cv


class BackgroundSubtraction(Module):
    def __init__(self):
        super().__init__()
        self.raw_image_in = Input(data_type=CVImage, config_keys=['cam_ids'])
        self.set_background_trigger_in = Input(data_type=SetBackgroundTrigger)
        self.foreground_out = Output(data_type=CVImage, config_keys=['cam_ids'])
        self.background_subtractor = None

        self.temp_bg_sub = None
        self.temp_subtraction_active = False

        self.cam_ids = ModuleParameter(None, data_type=list)

        # cv.namedWindow('Input')

    def get_bg_sub(self):
        if self.temp_subtraction_active:
            if self.temp_bg_sub is None:
                self.temp_bg_sub = {c: cv.createBackgroundSubtractorMOG2() for c in self.cam_ids}
            return self.temp_bg_sub
        else:
            if self.background_subtractor is None:
                self.background_subtractor = {c: cv.createBackgroundSubtractorMOG2() for c in self.cam_ids}
            return self.background_subtractor

    def process_set_background_trigger_in(self, trigger):
        if trigger.dart_number == 0:
            self.temp_subtraction_active = False
        else:
            self.temp_bg_sub = {c: cv.createBackgroundSubtractorMOG2() for c in self.cam_ids}
            self.temp_subtraction_active = True

    def add_background(self, background):
        # self.log_debug("got_bg")
        Module.show_image("BG", resize(background, 0.2))
        self.get_bg_sub()[background.camera_info['name']].apply(background, learningRate=0.5)

    def process_raw_image_in(self, image):
        _image__2 = CVImage(image, image.id, image.camera_info).copy()
        foreground = self.get_bg_sub()[image.camera_info['name']].apply(image, learningRate=0)

        diff = cv.bilateralFilter(foreground, 11, 57, 57)

        kernel = np.ones((5, 5), np.uint8)
        opened = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel)
        opened = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
        opened = cv.threshold(opened, 5, 255, cv.THRESH_BINARY)[1]
        if np.sum(opened) < 100000:
            self.add_background(image)

        self.foreground_out.data_ready(CVImage(foreground, image.id, image.camera_info))

        # Module.show_image("Input", resize(_image__2, 0.4))
        Module.show_image(self.module_name, resize(CVImage(foreground, image.id, image.camera_info), 0.4))

    def __start__(self):
        try:
            # data = np.load('BACKGROUNDS.npy')
            with open('BACKGROUNDS', 'rb') as bg_file:
                data = pickle.load(bg_file)
                for f in data:
                    image = f['image']
                    image.id = f['id']
                    image.camera_info = f['camera_info']
                    self.add_background(image)
                    pass
        except FileNotFoundError as e:
            print(e)


