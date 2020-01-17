import json
import re
import time
from copy import deepcopy
from typing import Union, Dict

from core.constants import *
from core.helper import ModuleParameter
from core.module import Module, Input, Output
from core.datatypes import CVImage, JsonObject
import cv2 as cv


class VisualCalibration(Module):
    def __init__(self):
        super().__init__()
        self.raw_image_in = Input(data_type=CVImage, config_keys=['cam_ids'], num_worker_threads=2)
        self.config_in = Input(data_type=JsonObject)
        self.config_out = Output(data_type=JsonObject)

        self.cam_ids = ModuleParameter(None, data_type=list)
        self.enable_reconfiguration = ModuleParameter(False)

        self.windows = {cam_id: cv.namedWindow('Calibrate %s' % cam_id) for cam_id in [0,1]}
        self.raw_images = {}
        self.config = {}
        self.initialized_configs = []

    def process_raw_image_in(self, raw_image):
        cam_id = raw_image.camera_info['name']
        self.log_debug('got image', cam_id)
        self.raw_images[cam_id] = CVImage(raw_image, raw_image.id, raw_image.camera_info)
        if cam_id in self.initialized_configs:
            self.redraw_window(cam_id)

    def process_config_in(self, config: JsonObject):
        cam_id = re.match('.*?([0-9]+)$', config.topic).group(1)
        self.log_debug('got config', cam_id, config.get_dict())
        for param in config.get_dict().keys():
            if not int(cam_id) in self.config:
                self.config[int(cam_id)] = {}
            self.config[int(cam_id)][param] = config.get_dict()[param]
        self.create_window(int(cam_id))
        self.initialized_configs.append(cam_id)

    def redraw_window(self, window_id):
        while not window_id in self.raw_images.keys():
            time.sleep(1)
            self.log_debug('waiting for image '% window_id)
        copied_im = deepcopy(self.raw_images[window_id])
        copied_im = cv.resize(copied_im, (int(1920/1.5), int(1080/1.5)))

        bull_x = int(copied_im.shape[1] * getattr(self, 'bull_location_%s' % window_id))
        board_rad = int(copied_im.shape[1] * getattr(self, 'board_radius_%s' % window_id))
        board_surface_y = int(copied_im.shape[0] * getattr(self, 'board_surface_%s' % window_id))
        roi_start_y = int(copied_im.shape[0] * getattr(self, 'roi_start_%s' % window_id))
        roi_end_y = int(copied_im.shape[0] * getattr(self, 'roi_end_%s' % window_id))

        cv.line(copied_im, (bull_x, 0), (bull_x, copied_im.shape[0]), (0, 255, 0), 1)

        for l in [RADIUS_OUTER_DOUBLE_MM, RADIUS_INNER_DOUBLE_MM, RADIUS_INNER_TRIPLE_MM,
                  RADIUS_OUTER_TRIPLE_MM, RADIUS_INNER_BULL_MM, RADIUS_OUTER_BULL_MM]:
            _x = int(board_rad * (l / RADIUS_OUTER_DOUBLE_MM))

            # outer-double-line left
            cv.line(copied_im, (bull_x - _x, 0), (bull_x - _x, copied_im.shape[0]), (255, 255, 0), 1)
            # outer-triple-line left
            cv.line(copied_im, (bull_x + _x, 0), (bull_x + _x, copied_im.shape[0]), (255, 255, 0), 1)

        cv.line(copied_im, (0, board_surface_y), (copied_im.shape[1], board_surface_y), (0, 255, 0), 1)
        cv.line(copied_im, (0, roi_start_y), (copied_im.shape[1], roi_start_y), (255, 0, 255), 1)
        cv.line(copied_im, (0, roi_end_y), (copied_im.shape[1], roi_end_y), (0, 0, 255), 1)

        cv_im = CVImage(copied_im, time.time(), {'name': window_id})
        Module.show_image('Calibrate %s' % window_id, cv_im, axis=1)

    def create_window(self, cam_id):
        self.log_debug('trying to create', cam_id)
        window_name = 'Calibrate %s' % cam_id

        cv.createTrackbar('SEND ON CLICK', window_name, 0, 1, self.create_update_button_function(cam_id))
        for param in self.config[cam_id].keys():
            self.create_trackbar(cam_id, window_name, param, self.config[cam_id][param], steps=400,
                                 min_value=0.2, max_value=0.6)
        self.log_debug('created', cam_id)
        self.redraw_window(cam_id)

    def create_update_button_function(self, _cam):
        def update(_, cam=_cam):
            calib = {cam: self.config[cam]}
            self.config_out.data_ready(JsonObject(json_obj=calib, topic='calibration/data/new_calibration/%s' % cam))
        return update

    def update_func(_o, _p, _c, _min_v, _max_v, _s):
        def update(v, o=_o, p=_p, c=_c, max_v=_max_v, min_v=_min_v, s=_s):
            setattr(o, '%s_%s' % (p, c), (v * ((max_v - min_v) / s)) + min_v)
            o.redraw_window(c)

        return update

    def create_trackbar(obj, cam: int, window_name: str, param_name: str,
                        value: float = 0.0,
                        min_value: float = 0.0, max_value: float = 100.0, steps: int = 100):

        if not hasattr(obj, '%s_%s' % (param_name, cam)):
            setattr(obj, '%s_%s' % (param_name, cam), ModuleParameter(value))

            cv.createTrackbar('%s %s' % (param_name, cam), window_name,
                              int((getattr(obj, '%s_%s' % (param_name, cam)) - min_value) * (
                                          steps / (max_value - min_value))),
                              int(steps),
                              obj.update_func( param_name, cam, min_value, max_value, steps)
                              )
