from typing import List

import numpy as np

from core.helper import ModuleParameter, create_trackbar
from core.module import Module, Input, Output
from core.datatypes import CVImage, Contours, ImpactPoint, ImpactPoints, ContourCollection
from core.convenience import resize
import cv2 as cv


class FitLine(Module):
    def __init__(self):
        super().__init__()
        self.impact_points_out = Output(data_type=ImpactPoints, config_keys=['cam_ids'])
        self.raw_image_in = Input(data_type=CVImage, config_keys=['cam_ids'])
        self.contour_collection_in = Input(data_type=ContourCollection, config_keys=['cam_ids'])
        # self.line_image_out = Output(data_type=CVImage, config_keys=['cam_ids'])
        self.raw_images = dict()
        self.cam_ids = ModuleParameter(None, data_type=list)

        cv.namedWindow(self.module_name)
    #
    # def _configure(self, config: dict, **kwargs):
    #     super()._configure(config, **kwargs)
    #     if 'cam_ids' in config:
    #         create_trackbar(self,
    #                         self.cam_ids, self.module_name, 'bull_location',
    #                         cam_defaults={0: 0.5025, 1: 0.507}, default=0.5,
    #                         min_value=0.25, max_value=0.75, steps=200)
    #         create_trackbar(self,
    #                         self.cam_ids, self.module_name, 'board_radius',
    #                         cam_defaults={0: 0.355, 1: 0.26}, default=0.355,
    #                         min_value=0.1, max_value=0.5, steps=200)
    #         create_trackbar(self,
    #                         self.cam_ids, self.module_name, 'board_surface',
    #                         cam_defaults={0: 0.55, 1: 0.3}, default=0.5,
    #                         min_value=0.1, max_value=0.9, steps=200)

    def process_raw_image_in(self, raw_image):
        if not raw_image.id in self.raw_images:
            self.raw_images[raw_image.id] = dict()
        self.raw_images[raw_image.id][raw_image.camera_info['name']] = raw_image

    def process_contour_collection_in(self, contour_collection):

        impact_points = []
        for contours in contour_collection.collection:
            raw_image = self.raw_images[contours.image_id][contours.camera_info['name']]
            board_surface_y = contours.camera_info['board_surface_y']
            #
            # bull_x = int(raw_image.shape[1] * getattr(self, 'bull_location_%s' % raw_image.camera_info['name']))
            # board_rad = int(raw_image.shape[1] * getattr(self, 'board_radius_%s' % raw_image.camera_info['name']))
            #
            # c_info = raw_image.camera_info
            # c_info['bull'] = bull_x
            # c_info['radius'] = board_rad
            #
            # cv.line(raw_image, (bull_x, 0), (bull_x, raw_image.shape[0]), (0, 255, 0), 2)
            # cv.line(raw_image, (bull_x-board_rad, 0), (bull_x-board_rad, raw_image.shape[0]), (255, 255, 0), 2)
            # cv.line(raw_image, (bull_x+board_rad, 0), (bull_x+board_rad, raw_image.shape[0]), (255, 255, 0), 2)
            #
            # board_surface_y = int(raw_image.shape[0]*getattr(self, 'board_surface_%s' % contours.camera_info['name']))
            # cv.line(raw_image, (0, board_surface_y), (raw_image.shape[1], board_surface_y), (0, 255, 0), 2)

            largest = sorted(contours.contours, key=self.a_len, reverse=True)[:10]

            def board_intersect(_line):

                a = _line[2]
                b = _line[3]
                c = _line[0]
                d = _line[1]
                intersect_x = int(c * ((board_surface_y-b)/d) + a)

                return intersect_x, board_surface_y

            # todo: make sure we always pick the right contour
            for i, contour in enumerate(largest[:1]):
                line = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)

                vx = line[0]
                vy = line[1]
                x = line[2]
                y = line[3]

                point1 = (x - vx * 5000, y - vy * 5000)
                point2 = (x + vx * 5000, y + vy * 5000)

                cv.line(raw_image, point1, point2, (0, 0, 255), 1)
                cv.circle(raw_image, board_intersect(line), 7, (255, 255, 0))
                impact_points.append(ImpactPoint(board_intersect(line), contours.image_id, contours.camera_info))

            Module.show_image(self.module_name, resize(raw_image, 0.3))

            self.raw_images[raw_image.id].pop(contours.camera_info['name'])
            if not len(self.raw_images[raw_image.id]) > 0:
                self.raw_images.pop(raw_image.id)

        self.impact_points_out.data_ready(ImpactPoints(impact_points))

    @staticmethod
    def a_len(c):
        return cv.arcLength(c, True)



