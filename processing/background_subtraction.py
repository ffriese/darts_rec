import time
from copy import deepcopy
from threading import Lock

import numpy as np

from termcolor import colored

from core.helper import ModuleParameter
from core.module import Module, Input, Output
from core.datatypes import CVImage, SetBackgroundTrigger, MultiImage
import cv2 as cv


class BackgroundSubtraction(Module):

    create_new = cv.createBackgroundSubtractorMOG2

    def __init__(self):
        super().__init__()
        self.images_in = Input(data_type=MultiImage, config_keys=['cam_ids'], num_worker_threads=1)
        self.rois_in = Input(data_type=MultiImage, config_keys=['cam_ids'], num_worker_threads=3)
        self.set_background_trigger_in = Input(data_type=SetBackgroundTrigger)
        self.synced_foregrounds_out = Output(data_type=MultiImage, config_keys=['cam_ids'])

        self.background_subtractor = None
        self.temp_subtractor = None
        self.temp_subtraction_active = False

        self.latest_safe_event_bg = time.time()
        self.synced_sub_in_progress = False
        self.sync_sub_lock = Lock()

        self.sub_lock = Lock()
        self.initial_images = None

        self.thresh_low = 2000
        self.thresh_high = 20000
        self.thresh_too_high = 150000

        self.cam_ids = ModuleParameter(None, data_type=list)
        self.enable_debug_images = ModuleParameter(False)
        self.min_amount_of_initial_images = ModuleParameter(5)

    def configure(self,
                  enable_debug_images: bool = None,
                  min_amount_of_initial_images: int = None):
        self._configure(locals())

    def re_init_subtractors(self):
        self.log_debug(colored("creating background subtractors", 'red'))

        cam_ids = [c for c in self.cam_ids]
        for c in self.cam_ids:
            cam_ids.append('EVENT_%s' % c)

        subs = {c: BackgroundSubtraction.create_new(detectShadows=False) for c in cam_ids}
        self.initial_images = {c: 0 for c in cam_ids}
        return subs

    def get_bg_sub(self):
        with self.sub_lock:
            if self.temp_subtraction_active:
                if self.temp_subtractor is None:
                    self.log_debug('temp-sub none')
                    self.temp_subtractor = self.re_init_subtractors()
                return self.temp_subtractor
            else:
                if self.background_subtractor is None:
                    self.log_debug('bg sub none')
                    self.background_subtractor = self.re_init_subtractors()
                return self.background_subtractor

    def process_set_background_trigger_in(self, trigger: SetBackgroundTrigger):
        self.log_debug('set-bg trigger')
        while self.synced_sub_in_progress:
            time.sleep(0.001)
        with self.sub_lock:
            if trigger.dart_number == 0:
                self.temp_subtraction_active = False
            else:
                self.temp_subtractor = self.re_init_subtractors()
                self.temp_subtraction_active = True

    def add_background(self, background):
        self.log_debug('add bg for', background.camera_info['name'], background.shape)

        self.initial_images[background.camera_info['name']] += 1
        self.get_bg_sub()[background.camera_info['name']].apply(background, learningRate=0.5)
        Module.show_image("BG %s" % background.cam_id(), background)

    def process_rois_in(self, rois: MultiImage):
        foregrounds = []
        for image in rois.images:
            roi = image.camera_info['suggested_roi']
            self.log_debug('getting bg-sub for cam', image.cam_id(), 'with',
                           self.initial_images[image.cam_id()], 'initial images')
            foreground = CVImage(self.get_bg_sub()[image.cam_id()].apply(image, learningRate=0),
                                 image.id, image.camera_info)

            foreground.camera_info['roi'] = roi
            foreground.camera_info.pop('suggested_roi')
            foregrounds.append(foreground)
        with self.sub_lock:
            self.synced_sub_in_progress = False
        self.synced_foregrounds_out.data_ready(MultiImage(foregrounds))

    def process_images_in(self, images: MultiImage):
        diffs = []
        image_collection = {}

        if self.initial_images is None:
            self.log_debug('initial images none')
            self.get_bg_sub()

        do_return = False

        for image in images.images:
            cam_id = image.cam_id()
            image_collection[image.cam_id()] = {}
            roi = image.camera_info['suggested_roi']
            roi_image = image[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
            image_collection[image.cam_id()]['roi'] = roi_image
            scaled_image = cv.resize(roi_image, dsize=(int(roi[2] / 4), int(roi[3] / 4)),
                                     interpolation=cv.INTER_NEAREST)

            scaled_image = CVImage(scaled_image, image.id, deepcopy(image.camera_info))
            scaled_image.camera_info['name'] = 'EVENT_%s' % cam_id
            image_collection[image.cam_id()]['scaled'] = scaled_image

            if self.initial_images[scaled_image.cam_id()] < self.min_amount_of_initial_images:
                self.add_background(scaled_image)
                self.add_background(CVImage(roi_image, image.id, image.camera_info))
                do_return = True
                continue

            # start event-detection
            small = CVImage(self.get_bg_sub()[scaled_image.cam_id()].apply(scaled_image, learningRate=0),
                                 image.id, image.camera_info)
            small_fg = cv.bilateralFilter(small, 5, 57, 57)

            kernel = np.ones((2, 2), np.uint8)
            small_fg = cv.morphologyEx(small_fg, cv.MORPH_OPEN, kernel)
            small_fg = cv.morphologyEx(small_fg, cv.MORPH_CLOSE, kernel)
            small_fg = cv.threshold(small_fg, 5, 255, cv.THRESH_BINARY)[1]

            diff = np.sum(small_fg)
            diffs.append(diff)

        if do_return:
            return

        max_diff = max(diffs)
        min_diff = min(diffs)
        if max_diff > 0:
            self.log_debug('diff:', diffs, max_diff, 'queue-size:', self.images_in.get_queue_size(),
                           'INITIAL IMS:', self.initial_images)

        if self.thresh_low < max_diff < self.thresh_high:
            for collection in image_collection.values():
                self.add_background(collection['scaled'])
                self.add_background(collection['roi'])

        # this probably won't work nicely until we have 3 cameras
        if max_diff > self.thresh_high and min_diff > (self.thresh_low * 2):
            self.log_debug(colored('OVER THRESHOLD! %r' % diffs, 'cyan'))
            if min(self.initial_images.values()) < self.min_amount_of_initial_images:
                self.log_debug('too few images..... ignoring')
                return
            with self.sub_lock:
                self.synced_sub_in_progress = True
            self.set_background_trigger_in.add_to_data_queue(SetBackgroundTrigger(1), self)
            self.rois_in.add_to_data_queue(MultiImage([c['roi'] for c in image_collection.values()]), self)



