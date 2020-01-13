import time
from copy import deepcopy
from threading import Lock

import numpy as np
import pickle

from termcolor import colored

from core.helper import ModuleParameter, create_trackbar
from core.module import Module, Input, Output
from core.datatypes import CVImage, SetBackgroundTrigger, CollectionTrigger, MultiImage
from core.convenience import resize
import cv2 as cv


class BackgroundSubtraction(Module):

    #new = cv.BackgroundSubtractor
    new = cv.createBackgroundSubtractorMOG2

    def __init__(self):
        super().__init__()
        # self.event_image_in = Input(data_type=CVImage, num_worker_threads=1)
        self.images_in = Input(data_type=MultiImage, config_keys=['cam_ids'], num_worker_threads=1)
        # self.synced_images_in = Input(data_type=MultiImage, config_keys=['cam_ids'], num_worker_threads=1)
        self.rois_in = Input(data_type=MultiImage, config_keys=['cam_ids'], num_worker_threads=3)
        self.set_background_trigger_in = Input(data_type=SetBackgroundTrigger)
        # self.foreground_out = Output(data_type=CVImage, config_keys=['cam_ids'])
        self.synced_foregrounds_out = Output(data_type=MultiImage, config_keys=['cam_ids'])
        # self.collection_trigger_out = Output(data_type=CollectionTrigger)

        self.background_subtractor = None
        self.temp_subtractor = None
        self.temp_subtraction_active = False

        self.latest_safe_event_bg = time.time()
        self.synced_sub_in_progress = False
        self.sync_sub_lock = Lock()

        self.sub_lock = Lock()
        self.initial_images = None

        self.roi = [50, 350, 1850, 130]
        self.thresh_low = 2000
        self.thresh_high = 20000

        self.cam_ids = ModuleParameter(None, data_type=list)
        self.enable_debug_images = ModuleParameter(False)

    def configure(self, enable_debug_images: bool = None):
        self._configure(locals())

    def re_init_subtractors(self):
        self.log_debug(colored("creating background subtractors", 'red'))

        cam_ids = [c for c in self.cam_ids]
        for c in self.cam_ids:
            cam_ids.append('EVENT_%s' % c)

        subs = {c: BackgroundSubtraction.new(detectShadows=False) for c in cam_ids}
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

    # def process_synced_images_in(self, images: MultiImage):
    #     if images.has_processing_trigger:
    #         foregrounds = []
    #         for image in images.images:
    #             im = image[int(self.roi[1]):int(self.roi[1] + self.roi[3]),
    #                  int(self.roi[0]):int(self.roi[0] + self.roi[2])]
    #             self.log_debug('getting bg-sub for cam', image.cam_id(), 'with',
    #                            self.initial_images[image.cam_id()], 'initial images')
    #             foreground = CVImage(self.get_bg_sub()[image.camera_info['name']].apply(im, learningRate=0),
    #                                  image.id, image.camera_info)
    #             foreground.camera_info['roi'] = self.roi
    #             foregrounds.append(foreground)
    #         with self.sync_sub_lock:
    #             self.synced_sub_in_progress = False
    #         self.synced_foregrounds_out.data_ready(MultiImage(foregrounds))
    #     else:
    #         for image in images.images:
    #             im = image[int(self.roi[1]):int(self.roi[1] + self.roi[3]),
    #                  int(self.roi[0]):int(self.roi[0] + self.roi[2])]
    #             self.add_background(im)
    #         # self.synced_foregrounds_out.data_ready(images)

    def process_rois_in(self, rois: MultiImage):
        foregrounds = []
        for image in rois.images:
            self.log_debug('getting bg-sub for cam', image.cam_id(), 'with',
                           self.initial_images[image.cam_id()], 'initial images')
            foreground = CVImage(self.get_bg_sub()[image.cam_id()].apply(image, learningRate=0),
                                 image.id, image.camera_info)

            foreground.camera_info['roi'] = self.roi
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
            roi_image = image[int(self.roi[1]):int(self.roi[1] + self.roi[3]), int(self.roi[0]):int(self.roi[0] + self.roi[2])]
            image_collection[image.cam_id()]['roi'] = roi_image
            scaled_image = cv.resize(roi_image, dsize=(int(self.roi[2] / 4), int(self.roi[3] / 4)),
                                     interpolation=cv.INTER_NEAREST)

            scaled_image = CVImage(scaled_image, image.id, deepcopy(image.camera_info))
            scaled_image.camera_info['name'] = 'EVENT_%s' % cam_id
            image_collection[image.cam_id()]['scaled'] = scaled_image

            if self.initial_images[scaled_image.cam_id()] < 2:
                self.add_background(scaled_image)
                self.add_background(CVImage(roi_image, image.id, image.camera_info))
                do_return = True
                continue

            # self.log_debug('GETTING DIFF FOR %s WITH %s initial images' %
            #                (scaled_image.cam_id(), self.initial_images[scaled_image.cam_id()]))
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

        m_diff = max(diffs)

        self.log_debug('diff:', diffs, m_diff, 'queue-size:', self.images_in.get_queue_size(),
                       'INITIAL IMS:', self.initial_images)

        if self.thresh_low < m_diff < self.thresh_high:
            for collection in image_collection.values():
                self.add_background(collection['scaled'])
                self.add_background(collection['roi'])
        if m_diff > self.thresh_high:
            self.log_debug(colored('OVER THRESHOLD!', 'cyan'))
            with self.sub_lock:
                self.synced_sub_in_progress = True
            self.set_background_trigger_in.add_to_data_queue(SetBackgroundTrigger(1), self)
            self.rois_in.add_to_data_queue(MultiImage([c['roi'] for c in image_collection.values()]), self)
#
#     def process_event_image_in(self, image: CVImage):
#         if self.synced_sub_in_progress:
#             return
#         # _image__2 = CVImage(image, image.id, image.camera_info).copy()
#         im = image[int(self.roi[1]):int(self.roi[1] + self.roi[3]), int(self.roi[0]):int(self.roi[0] + self.roi[2])]
#         im = cv.resize(im, dsize=(int(self.roi[2]/4), int(self.roi[3]/4)), interpolation=cv.INTER_NEAREST)
#
#         im = CVImage(im, image.id, deepcopy(image.camera_info))
#         im.camera_info['name'] = 'EVENT'
#         Module.show_image('event', im)
#         if self.initial_images is None:
#             self.log_debug('initial images none')
#             self.get_bg_sub()
#         if self.initial_images['EVENT'] < 2:
#             self.latest_safe_event_bg = time.time()+0.5
#             self.collection_trigger_out.data_ready(CollectionTrigger(processing_trigger=False))
#             self.add_background(im)
#             return
#         start = time.time()
#         foreground = CVImage(self.get_bg_sub()['EVENT'].apply(im, learningRate=0),
#                                  image.id, image.camera_info)
#
#         small = foreground
# #        small_fg = cv.bilateralFilter(small, 11, 57, 57)
#         small_fg = cv.bilateralFilter(small, 5, 57, 57)
#
#         kernel = np.ones((2, 2), np.uint8)
#         small_fg = cv.morphologyEx(small_fg, cv.MORPH_OPEN, kernel)
#         small_fg = cv.morphologyEx(small_fg, cv.MORPH_CLOSE, kernel)
#         small_fg = cv.threshold(small_fg, 5, 255, cv.THRESH_BINARY)[1]
#         # self.log_debug('BG_SUB took: %f' % (time.time()-start))
#         diff = np.sum(small_fg)
#
#         self.log_debug('diff:', diff, 'queue-size:', len(self.event_image_in._data_queue),
#                        'INITIAL IMS:',  self.initial_images['EVENT'])
#
#         if self.thresh_low < diff < self.thresh_high:
#             self.add_background(im)
#             self.latest_safe_event_bg = image.camera_info['ts']
#             self.collection_trigger_out.data_ready(CollectionTrigger(processing_trigger=False))
#         if diff > self.thresh_high:
#             # TEMPORARY!!!!
#             self.collection_trigger_out.data_ready(CollectionTrigger(processing_trigger=True))
#             self.log_debug(colored('OVER THRESHOLD!', 'cyan'))
#             self.foreground_out.data_ready(CVImage(im, image.id, image.camera_info))
#             with self.sync_sub_lock:
#                 self.synced_sub_in_progress = True
#             self.set_background_trigger_in.add_to_data_queue(SetBackgroundTrigger(1))
#
#         # Module.show_image(self.module_name, CVImage(small_fg, image.id, image.camera_info))
#
#         # Module.show_image("Input", resize(_image__2, 0.4))
#         # if self.enable_debug_images:
#         #     Module.show_image(self.module_name, f, 0.4))

    # def __start__(self):
    #     try:
    #         # data = np.load('BACKGROUNDS.npy')
    #         with open('BACKGROUNDS', 'rb') as bg_file:
    #             data = pickle.load(bg_file)
    #             for f in data:
    #                 image = f['image']
    #                 image.id = f['id']
    #                 image.camera_info = f['camera_info']
    #                 self.add_background(image)
    #                 pass
    #     except FileNotFoundError as e:
    #         print(e)


