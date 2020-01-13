import copy
from collections import defaultdict
from typing import List

import numpy as np

from core.constants import *
from core.helper import ModuleParameter, create_trackbar
from core.module import Module, Input, Output
from core.datatypes import CVImage, MultiImage, \
    CollectionTrigger
import cv2 as cv


class Calibrator(Module):
    def __init__(self):
        super().__init__()
        self.raw_images_in = Input(data_type=MultiImage, config_keys=['cam_ids'])
        self.calibrated_images_out = Output(data_type=MultiImage, config_keys=['cam_ids'])
        self.display_images_out = Output(data_type=MultiImage, config_keys=['cam_ids'])
        self.calibration_trigger_out = Output(data_type=CollectionTrigger)
        self.cam_ids = ModuleParameter(None, data_type=list)
        self.defaults = {
            'bull_location': defaultdict(lambda: 0.5, [(0, 0.487), (1, 0.50575)]),
            'board_radius': defaultdict(lambda: 0.26, [(0, 0.26125), (1, 0.259)]),
            'board_surface': defaultdict(lambda: 0.3, [(0, 0.269), (1, 0.3)]),
        }

    def process_raw_images_in(self, raw_images: MultiImage):
        processed_images = []
        display_images = []
        for raw_image in raw_images.images:
            cam_id = raw_image.camera_info['name']

            bull_x = int(raw_image.shape[1] * getattr(self, 'bull_location_%s' % cam_id))
            board_rad = int(raw_image.shape[1] * getattr(self, 'board_radius_%s' % cam_id))
            board_surface_y = int(raw_image.shape[0] * getattr(self, 'board_surface_%s' % cam_id))

            c_info = raw_image.camera_info
            c_info['bull'] = bull_x
            c_info['radius'] = board_rad
            c_info['board_surface_y'] = board_surface_y

            processed_images.append(CVImage(raw_image, raw_image.id, c_info))
            display_image = copy.deepcopy(raw_image)
            # bull-line
            cv.line(display_image, (bull_x, 0), (bull_x, display_image.shape[0]), (0, 255, 0), 1)

            for l in [RADIUS_OUTER_DOUBLE_MM, RADIUS_INNER_DOUBLE_MM, RADIUS_INNER_TRIPLE_MM,
                      RADIUS_OUTER_TRIPLE_MM, RADIUS_INNER_BULL_MM, RADIUS_OUTER_BULL_MM]:
                _x = int(board_rad * (l / RADIUS_OUTER_DOUBLE_MM))

                # outer-double-line left
                cv.line(display_image, (bull_x-_x, 0), (bull_x-_x, display_image.shape[0]), (255, 255, 0), 1)
                # outer-triple-line left
                cv.line(display_image, (bull_x+_x, 0), (bull_x+_x, display_image.shape[0]), (255, 255, 0), 1)

            cv.line(raw_image, (0, board_surface_y), (display_image.shape[1], board_surface_y), (0, 255, 0), 1)
            display_images.append(CVImage(display_image, display_image.id, c_info))
        # disp_size = (int(1920/4), int(1080/4))
        # display_image = CVImage(cv.resize(np.concatenate(display_images, axis=0),
        #                                   dsize=(disp_size[0], disp_size[1]*len(display_images)),
        #                                   interpolation=cv.INTER_CUBIC), display_images[0].id,
        #                         {'name': display_images[0].cam_id()})
        self.calibrated_images_out.data_ready(MultiImage(processed_images, raw_images.has_processing_trigger))
        self.display_images_out.data_ready(MultiImage(display_images))

    def __custom_pre_start__(self):
        for param_name in self.defaults.keys():
            for cam_id in self.cam_ids:
                if not hasattr(self, '%s_%s' % (param_name, cam_id)):
                    setattr(self, '%s_%s' % (param_name, cam_id), ModuleParameter(self.defaults[param_name][cam_id]))


