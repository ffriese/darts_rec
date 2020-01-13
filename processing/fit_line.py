from collections import OrderedDict
from threading import Lock
from typing import List

import numpy as np

from core.helper import ModuleParameter, create_trackbar
from core.module import Module, Input, Output
from core.datatypes import CVImage, Contours, ImpactPoint, ImpactPoints, ContourCollection, MultiImage
from core.convenience import resize
import cv2 as cv


class FitLine(Module):
    def __init__(self):
        super().__init__()
        self.impact_points_out = Output(data_type=ImpactPoints, config_keys=['cam_ids'])
        self.raw_images_in = Input(data_type=MultiImage, config_keys=['cam_ids'])
        self.contour_collection_in = Input(data_type=ContourCollection, config_keys=['cam_ids'])
        self.debug_images_out = Output(data_type=MultiImage, config_keys=['cam_ids'])
        self.raw_multis = OrderedDict()
        self.cam_ids = ModuleParameter(None, data_type=list)

        # cv.namedWindow(self.module_name)

    def process_raw_images_in(self, raw_images):
        self.raw_multis[raw_images.images[0].id] = raw_images
        # prevent memory leak
        while len(self.raw_multis) > 10:
            self.raw_multis.popitem(False)

    def process_contour_collection_in(self, contour_collection):
        if not contour_collection.collection:
            self.log_error('NOT ENOUGH CONTOURS!!!!!!!!!!!!!!!!!!!!!!!')
            return
        image_id = contour_collection.collection[0].image_id
        if image_id not in self.raw_multis:
            self.log_error('raw image not found!!!!!!!!!!!!!!!!!!!')
            return
        raw_multi_images = self.raw_multis[contour_collection.collection[0].image_id]
        impact_points = []
        images = []
        raw_images = {image.cam_id(): image for image in raw_multi_images.images}
        self.log_debug(list(raw_images.keys()))
        for contours in contour_collection.collection:
            raw_image = raw_images[contours.camera_info['name']]
            roi = contours.camera_info['roi']
            largest = sorted(contours.contours, key=self.a_len, reverse=True)[:10]

            def board_intersect(_line, board_surface_y):

                a = _line[2]
                b = _line[3]
                c = _line[0]
                d = _line[1]
                intersect_x = int(c * ((board_surface_y-b)/d) + a)

                return intersect_x, board_surface_y

            # todo: make sure we always pick the right contour
            for i, contour in enumerate(largest[:1]):
                line = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
                line[2] += roi[0]
                line[3] += roi[1]

                vx = line[0]
                vy = line[1]
                x = line[2]
                y = line[3]

                point1 = (x - vx * 5000, y - vy * 5000)
                point2 = (x + vx * 5000, y + vy * 5000)
                board_y = min([p[1] for p in contour[0]]) + roi[1]

                cv.line(raw_image, point1, point2, (0, 0, 255), 1)
                cv.circle(raw_image, board_intersect(line, board_y), 7, (255, 255, 0))
                cv.rectangle(raw_image, tuple(roi[:2]), (roi[0]+roi[2], roi[1]+roi[3]), (0,0,255))
                impact_points.append(ImpactPoint(board_intersect(line, board_y), contours.image_id, contours.camera_info))
            images.append(CVImage(raw_image, contours.image_id, {'name': contours.camera_info['name'],
                                                                 'topic': 'fit_line'}))

            # for raw_image in images:
            #     Module.show_image(self.module_name, resize(raw_image, 0.5))

        self.impact_points_out.data_ready(ImpactPoints(impact_points))
        self.debug_images_out.data_ready(MultiImage(images))

    @staticmethod
    def a_len(c):
        return cv.arcLength(c, True)



