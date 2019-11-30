from typing import List

import numpy as np

from core.constants import *
from core.helper import ModuleParameter, create_trackbar
from core.module import Module, Input, Output
from core.datatypes import CVImage, Contours, ImpactPoint, ImpactPoints, ContourCollection
from core.convenience import resize
import cv2 as cv


class Warper(Module):
    def __init__(self):
        super().__init__()
        self.raw_image_in = Input(data_type=CVImage, config_keys=['cam_ids'])
        self.cam_ids = ModuleParameter(None, data_type=list)

        # cv.namedWindow(self.module_name)

        cv.namedWindow("PERSPECTIVE")


    def process_raw_image_in(self, raw_image):
        cam_id = raw_image.camera_info['name']


        perspective = raw_image.copy()


        ps = [
            {0: [80,  205],       1: [26*4,  99*4]},
            {0: [1805, 205],     1: [722*4, 82*4]},
            {0: [1295, 370],   1: [1270, 400]},
            {0: [555, 370],  1: [650,  395]},
        ]

        

        # fluchtpunkt
        # o1 = ps[]
        #
        # x = o2 - o1
        # d1 = p1 - o1;
        # d2 = p2 - o2;
        #
        # cross = d1.x * d2.y - d1.y * d2.x
        #
        # t1 = (x.x * d2.y - x.y * d2.x) / cross
        # r = o1 + d1 * t1

        my_ps = [p[cam_id] for p in ps]
        my_ps = np.array([my_ps], dtype=np.int32)
        cv.polylines(perspective, my_ps, True, (0,0,255), 2)

        # rect = order_points(my_ps[0])
        rect = my_ps[0]
        (tl, tr, br, bl) = rect

        dst = np.array([
            [0, 0],
            [1920, 0],
            [1920, 1080],
            [0, 1080 ]

            # [100, 100],
            # [1820, 100],
            # [1820, 120],
            # [100, 120]
        ], dtype="float32")
        rect = np.array(rect, dtype=np.float32)

        print(cam_id, '----')
        print(cam_id, rect)
        print(cam_id, dst)
        print(cam_id, '####')
        # compute the perspective transform matrix and then apply it
        M = cv.getPerspectiveTransform(rect, dst)
        warped = cv.warpPerspective(perspective, M, (perspective.shape[1], int(perspective.shape[0])))

        Module.show_image('PERSPECTIVE', resize(perspective, 0.8), axis=1)
        Module.show_image('WARPED', resize(CVImage(warped, perspective.id, perspective.camera_info), 0.5), axis=1)


