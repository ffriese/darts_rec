import base64
import io

import numpy as np
import math

from imageio import imread

from core.constants import RADIUS_INNER_BULL_MM, RADIUS_OUTER_BULL_MM, RADIUS_INNER_TRIPLE_MM, RADIUS_OUTER_TRIPLE_MM, \
    RADIUS_INNER_DOUBLE_MM, RADIUS_OUTER_DOUBLE_MM, RADIUS_BOARD_MM, FIELDS
from core.helper import ModuleParameter
from core.module import Module, Input, Output
from core.datatypes import CVImage, ImpactPoints, BoardCoordinate
from core.convenience import resize
import cv2 as cv

COLOR_LIGHT = (0.8, 0.8, 0.8)
COLOR_DARK = (0.1, 0.1, 0.1)
COLOR_DARK_MULTI = (0.1, 0.1, 0.7)
COLOR_LIGHT_MULTI = (0.1, 0.6, 0.0)



class ProjectOnBoard(Module):
    def __init__(self):
        super().__init__()
        self.sector_angle = 2 * math.pi / 20
        self.sector_degrees = 360/20
        self.impact_points_in = Input(data_type=ImpactPoints, config_keys=['cam_ids'])
        self.dartboard_out = Output(data_type=CVImage)
        self.coordinate_out = Output(data_type=BoardCoordinate)

        self.background = None
        self.curr_id = None
        self.factor = 1
        self.center = int(500*self.factor)
        self.cam_ids = ModuleParameter(None, data_type=list)

        self.cached_bg = CVImage(np.zeros((self.center*2, self.center*2, 3)), None, None)

        self.draw_dartboard(self.cached_bg)

        self.direction_factors = {
            0: -1,
            1: 1,
            2: 1
        }

    def draw_dartboard(self, background):
        for i in range(20):
            color = [COLOR_DARK, COLOR_LIGHT][i % 2]
            color_multi = [COLOR_DARK_MULTI, COLOR_LIGHT_MULTI][i % 2]
            # DOUBLES
            cv.ellipse(background, (self.center, self.center),
                       (int(self.factor * RADIUS_OUTER_DOUBLE_MM), int(self.factor * RADIUS_OUTER_DOUBLE_MM)),
                       0.0, self.sector_degrees * i - (90 + 360 / 40), self.sector_degrees * (i + 1) - (90 + 360 / 40),
                       color_multi, thickness=-1)
            # OUTER SINGLE
            cv.ellipse(background, (self.center, self.center),
                       (int(self.factor * RADIUS_INNER_DOUBLE_MM), int(self.factor * RADIUS_INNER_DOUBLE_MM)),
                       0.0, self.sector_degrees * i - (90 + 360 / 40), self.sector_degrees * (i + 1) - (90 + 360 / 40),
                       color, thickness=-1)
            # TRIPLE
            cv.ellipse(background, (self.center, self.center),
                       (int(self.factor * RADIUS_OUTER_TRIPLE_MM), int(self.factor * RADIUS_OUTER_TRIPLE_MM)),
                       0.0, self.sector_degrees * i - (90 + 360 / 40), self.sector_degrees * (i + 1) - (90 + 360 / 40),
                       color_multi, thickness=-1)
            # INNER SINGLE
            cv.ellipse(background, (self.center, self.center),
                       (int(self.factor * RADIUS_INNER_TRIPLE_MM), int(self.factor * RADIUS_INNER_TRIPLE_MM)),
                       0.0, self.sector_degrees * i - (90 + 360 / 40), self.sector_degrees * (i + 1) - (90 + 360 / 40),
                       color, thickness=-1)


            # WIRES
            cv.line(background,
                    (
                        int(self.center + int(self.factor * RADIUS_OUTER_BULL_MM) * math.cos((0.5 + i - 6) *
                                                                                             self.sector_angle)),
                        int(self.center + int(self.factor * RADIUS_OUTER_BULL_MM) * math.sin((0.5 + i - 6) *
                                                                                             self.sector_angle))
                     ),
                    (
                        int(self.center + int(self.factor * RADIUS_OUTER_DOUBLE_MM) * math.cos((0.5 + i - 6) *
                                                                                               self.sector_angle)),
                        int(self.center + int(self.factor * RADIUS_OUTER_DOUBLE_MM) * math.sin((0.5 + i - 6) *
                                                                                               self.sector_angle))
                     ),
                    (1, 1, 1), 1)

            cv.putText(background, str(FIELDS[i]),
                       (
                           int(self.center - 20 +
                               int(self.factor * (RADIUS_OUTER_DOUBLE_MM + 20)) * math.cos((i - 6) * self.sector_angle)),
                           int(self.center + 10 +
                               int(self.factor * (RADIUS_OUTER_DOUBLE_MM + 20)) * math.sin((i - 6) * self.sector_angle))
                       ),
                       cv.FONT_HERSHEY_DUPLEX,
                       1, (1, 1, 1))

        # OUTER BULL
        cv.ellipse(background, (self.center, self.center),
                   (int(self.factor * RADIUS_OUTER_BULL_MM), int(self.factor * RADIUS_OUTER_BULL_MM)),
                   0.0, 0, 360,
                   COLOR_LIGHT_MULTI, thickness=-1)
        # INNER BULL
        cv.ellipse(background, (self.center, self.center),
                   (int(self.factor * RADIUS_INNER_BULL_MM), int(self.factor * RADIUS_INNER_BULL_MM)),
                   0.0, 0, 360,
                   COLOR_DARK_MULTI, thickness=-1)
        # outer rim:
        cv.circle(background, (self.center, self.center), int(self.factor * RADIUS_BOARD_MM),
                  (1, 1, 1), 2)
        cv.circle(background, (self.center, self.center), int(self.factor * RADIUS_OUTER_DOUBLE_MM),
                  (1, 1, 1), 1)
        cv.circle(background, (self.center, self.center), int(self.factor * RADIUS_INNER_DOUBLE_MM),
                  (1, 1, 1), 1)
        cv.circle(background, (self.center, self.center), int(self.factor * RADIUS_OUTER_TRIPLE_MM),
                  (1, 1, 1), 1)
        cv.circle(background, (self.center, self.center), int(self.factor * RADIUS_INNER_TRIPLE_MM),
                  (1, 1, 1), 1)
        cv.circle(background, (self.center, self.center), int(self.factor * RADIUS_OUTER_BULL_MM),
                  (1, 1, 1), 1)
        cv.circle(background, (self.center, self.center), int(self.factor * RADIUS_INNER_BULL_MM),
                  (1, 1, 1), 1)

    def redraw_bg(self, impact_point):
        self.background = self.cached_bg.copy()
        ci = {k: v for k, v in impact_point.camera_info.items()}
        ci['name'] = 0
        self.background = self.cached_bg.copy()
        self.background.id = impact_point.image_id
        self.background.camera_info = ci

    @staticmethod
    def line_intersection(line1, line2):
        xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
        ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

        def det(a, b):
            return a[0] * b[1] - a[1] * b[0]

        div = det(xdiff, ydiff)
        if div == 0:
            raise Exception('lines do not intersect')

        d = (det(*line1), det(*line2))
        x = det(d, xdiff) / div
        y = det(d, ydiff) / div
        return x, y

    def process_impact_points_in(self, impact_points):

        self.redraw_bg(impact_points.points[0])

        lines = {}

        for impact_point in impact_points.points:

            cam_id = impact_point.camera_info['name']

            w = 1920

            # pixel-coordinates
            camera_center = w/2
            impact_x = (impact_point.point[0] - camera_center) * self.direction_factors[cam_id]
            bull_offset = (impact_point.camera_info['bull'] - camera_center) * self.direction_factors[cam_id]

            radius = impact_point.camera_info['radius']
            conversion_factor = RADIUS_OUTER_DOUBLE_MM / float(radius)

            # mm
            cam_dist_board_center = 460
            bull_offset_mm = bull_offset * conversion_factor
            impact_mm = impact_x * conversion_factor

            p_1 = (-cam_dist_board_center, bull_offset_mm) if cam_id == 0 else (bull_offset_mm, -cam_dist_board_center)
            p_2 = (0, impact_mm-bull_offset_mm) if cam_id == 0 else (impact_mm-bull_offset_mm, 0)

            lines[cam_id] = (p_1, p_2)

            cv.line(self.background,
                    (int(p_1[0]+self.center), int(p_1[1]+self.center)),
                    (int((2*p_2[0]-p_1[0])+self.center), int((2*p_2[1]-p_1[1])+self.center)),
                    (1, 0, 0) if cam_id == 1 else (1, 1, 0), 1)

        if len(lines.keys()) < 2:
            self.log_debug('NOT ENOUGH LINES!')
            return
        intersection = self.line_intersection(lines[0], lines[1])
        board_coordinate = (int(intersection[0] + RADIUS_OUTER_DOUBLE_MM),
                            int(intersection[1] + RADIUS_OUTER_DOUBLE_MM))
        display_coordinate = (int(intersection[0]+self.center),
                            int(intersection[1] + self.center))
        cv.circle(self.background, display_coordinate, 4, (0.5, 0, 1), thickness=2)
        self.log_info('BOARD-COORDINATE:', board_coordinate)
        self.coordinate_out.data_ready(BoardCoordinate(board_coordinate))
        self.background.camera_info['topic'] = 'dartboard'
        self.log_debug(self.background.shape, self.background.camera_info, self.background.id)

        self.dartboard_out.data_ready(CVImage(np.uint8((np.array(self.background[
                                                    int(self.center - self.factor * RADIUS_BOARD_MM * 1.2):
                                                        int(self.center + self.factor * RADIUS_BOARD_MM * 1.2),
                                                    int(self.center - self.factor * RADIUS_BOARD_MM * 1.2):
                                                        int(self.center + self.factor * RADIUS_BOARD_MM * 1.2)
                                                    ]))*255/2.), '0000', {'name': 0, 'topic': 'dartboard'}))


