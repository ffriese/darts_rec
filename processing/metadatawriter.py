import copy
import dill
import json
import pickle
from collections import defaultdict
from core.constants import *
from core.helper import ModuleParameter
from core.module import Module, Input, Output
from core.datatypes import CVImage, MultiImage, \
    CollectionTrigger, JsonObject
import cv2 as cv


class MetaDataWriter(Module):
    def __init__(self):
        super().__init__()
        self.raw_images_in = Input(data_type=MultiImage, config_keys=['cam_ids'])
        self.calibrated_images_out = Output(data_type=MultiImage, config_keys=['cam_ids'])
        self.display_images_out = Output(data_type=MultiImage, config_keys=['cam_ids'])
        self.config_in = Input(data_type=JsonObject)

        self.calibration_trigger_out = Output(data_type=CollectionTrigger)
        self.cam_ids = ModuleParameter(None, data_type=list)
        self.defaults = {
            'bull_location': defaultdict(lambda: 0.5, [(0, 0.487), (1, 0.50575)]),
            'board_radius': defaultdict(lambda: 0.26, [(0, 0.26125), (1, 0.259)]),
            'board_surface': defaultdict(lambda: 0.3, [(0, 0.269), (1, 0.3)]),
            'roi_start': defaultdict(lambda: 0.32, [(0, 0.3), (1, 0.31)]),
            'roi_end': defaultdict(lambda: 0.4, [(0, 0.4), (1, 0.4)]),
        }

        self.roi = [50, 350, 1850, 130]

    def process_raw_images_in(self, raw_images: MultiImage):
        processed_images = []
        display_images = []
        for raw_image in raw_images.images:
            cam_id = raw_image.camera_info['name']

            # todo: save all of this somewhere
            bull_x = int(raw_image.shape[1] * getattr(self, 'bull_location_%s' % cam_id))
            board_rad = int(raw_image.shape[1] * getattr(self, 'board_radius_%s' % cam_id))
            board_surface_y = int(raw_image.shape[0] * getattr(self, 'board_surface_%s' % cam_id))
            roi_start_y = int(raw_image.shape[0] * getattr(self, 'roi_start_%s' % cam_id))
            roi_end_y = int(raw_image.shape[0] * getattr(self, 'roi_end_%s' % cam_id))

            c_info = raw_image.camera_info
            c_info['bull'] = bull_x
            c_info['radius'] = board_rad
            c_info['board_surface_y'] = board_surface_y
            c_info['suggested_roi'] = self.roi
            c_info['calibration'] = {param: getattr(self, '%s_%s' % (param, cam_id)) for param in self.defaults.keys()}

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
        self.calibrated_images_out.data_ready(MultiImage(processed_images, raw_images.has_processing_trigger))
        self.display_images_out.data_ready(MultiImage(display_images))

    def process_config_in(self, config: JsonObject):
        self.log_debug('got', config.get_dict())
        cam = list(config.get_dict().keys())[0]
        for k, v in config.get_dict()[cam].items():
            self.defaults[k][int(cam)] = v

        print(self.defaults)
        with open('CALIBRATION', 'wb') as conf_file:
            pickle._dump(self.defaults, conf_file)

        # TODO: MAYBE SET self.defaults already

    def __custom_pre_start__(self):
        try:
            with open('CALIBRATION', 'rb') as conf_file:
                self.defaults = pickle.load(conf_file)

        except Exception as e:
            print('no calibration data found')
            print(e)

        # TODO: WRITE data TO self.defaults

        for param_name in self.defaults.keys():
            for cam_id in self.cam_ids:
                if not hasattr(self, '%s_%s' % (param_name, cam_id)):
                    setattr(self, '%s_%s' % (param_name, cam_id), ModuleParameter(self.defaults[param_name][cam_id]))


