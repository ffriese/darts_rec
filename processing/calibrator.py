from typing import List

import numpy as np

from core.constants import *
from core.helper import ModuleParameter, create_trackbar
from core.module import Module, Input, Output
from core.datatypes import CVImage, Contours, ImpactPoint, ImpactPoints, ContourCollection
from core.convenience import resize
import cv2 as cv


class Calibrator(Module):
    def __init__(self):
        super().__init__()
        self.raw_image_in = Input(data_type=CVImage, config_keys=['cam_ids'])
        self.calibrated_image_out = Output(data_type=CVImage, config_keys=['cam_ids'])
        self.cam_ids = ModuleParameter(None, data_type=list)
        self.enable_reconfiguration = ModuleParameter(False)

        cv.namedWindow(self.module_name)

        # cv.namedWindow("PERSPECTIVE")


    def process_raw_image_in(self, raw_image):
        cam_id = raw_image.camera_info['name']

        bull_x = int(raw_image.shape[1] * getattr(self, 'bull_location_%s' % cam_id))
        board_rad = int(raw_image.shape[1] * getattr(self, 'board_radius_%s' % cam_id))
        board_surface_y = int(raw_image.shape[0]*getattr(self, 'board_surface_%s' % cam_id))

        c_info = raw_image.camera_info
        c_info['bull'] = bull_x
        c_info['radius'] = board_rad
        c_info['board_surface_y'] = board_surface_y

        self.calibrated_image_out.data_ready(CVImage(raw_image, raw_image.id, c_info))

        # bull-line
        cv.line(raw_image, (bull_x, 0), (bull_x, raw_image.shape[0]), (0, 255, 0), 1)

        for l in [RADIUS_OUTER_DOUBLE_MM, RADIUS_INNER_DOUBLE_MM, RADIUS_INNER_TRIPLE_MM,
                  RADIUS_OUTER_TRIPLE_MM, RADIUS_INNER_BULL_MM, RADIUS_OUTER_BULL_MM]:
            _x = int(board_rad * (l / RADIUS_OUTER_DOUBLE_MM))

            # outer-double-line left
            cv.line(raw_image, (bull_x-_x, 0), (bull_x-_x, raw_image.shape[0]), (255, 255, 0), 1)
            # outer-triple-line left
            cv.line(raw_image, (bull_x+_x, 0), (bull_x+_x, raw_image.shape[0]), (255, 255, 0), 1)

        cv.line(raw_image, (0, board_surface_y), (raw_image.shape[1], board_surface_y), (0, 255, 0), 1)


        Module.show_image(self.module_name, resize(raw_image, 0.5), axis=1)



    def __start__(self):
        create_trackbar(self,
                        self.cam_ids, self.module_name, 'bull_location',
                        cam_defaults={0: 0.487, 1: 0.50575}, default=0.50,
                        min_value=0.45, max_value=0.55, steps=400)
        create_trackbar(self,
                        self.cam_ids, self.module_name, 'board_radius',
                        cam_defaults={0: 0.2535, 1: 0.259}, default=0.26,
                        min_value=0.2, max_value=0.3, steps=400)
        create_trackbar(self,
                        self.cam_ids, self.module_name, 'board_surface',
                        cam_defaults={0: 0.347, 1: 0.3}, default=0.5,
                        min_value=0.2, max_value=0.6, steps=400)

