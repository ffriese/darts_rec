from typing import List

import numpy as np

from core.helper import ModuleParameter, create_trackbar
from core.module import Module, Input, Output
from core.datatypes import CVImage, MultiImage
from core.convenience import resize
import cv2 as cv


class CleanDifference(Module):
    def __init__(self):
        super().__init__()
        self.foregrounds_in = Input(data_type=MultiImage, config_keys=['cam_ids'])
        self.diff_out = Output(data_type=MultiImage, config_keys=['cam_ids'])
        self.cam_ids = ModuleParameter(None, data_type=list)
        self.cut_defaults = {0: 0.8, 1: 0.5}

        # cv.namedWindow(self.module_name)

    def _configure(self, config: dict, **kwargs):
        super()._configure(config, **kwargs)
        # create_trackbar(self,
        #                 self.cam_ids, self.module_name, 'image_cut',
        #                 cam_defaults=self.cut_defaults, default=0.5,
        #                 min_value=0.0, max_value=1.0, steps=100)

    def process_foregrounds_in(self, fgs: MultiImage):

        images = []
        for fg in fgs.images:
            # cv.rectangle(fg, (0, int(fg.shape[0] * getattr(self, 'image_cut_%s' % fg.camera_info['name']))),
            #              (fg.shape[1], fg.shape[0]), (0, 0, 0), cv.FILLED)
            diff = cv.bilateralFilter(fg, 11, 57, 57)

            kernel = np.ones((3, 3), np.uint8)
            opened = cv.morphologyEx(diff, cv.MORPH_OPEN, kernel)
            opened = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel)
            opened = cv.threshold(opened, 5, 255, cv.THRESH_BINARY)[1]
            images.append(CVImage(opened, fg.id, fg.camera_info))
        # Module.show_image(self.module_name, resize(CVImage(opened, fg.id, fg.camera_info), 0.4))
        self.diff_out.data_ready(MultiImage(images))


