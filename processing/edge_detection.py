import imutils
import numpy as np

from core.helper import ModuleParameter
from core.module import Module, Input, Output
from core.datatypes import CVImage, Contours
from core.convenience import resize
import cv2 as cv

class EdgeDetection(Module):
    def __init__(self):
        super().__init__()
        self.diff_in = Input(data_type=CVImage, config_keys=['cam_ids'])
        self.edged_out = Output(data_type=CVImage, config_keys=['cam_ids'])
        self.contours_out = Output(data_type=Contours, config_keys=['cam_ids'])
        self.edge_limit = 1080/20

        self.cam_ids = ModuleParameter(None, data_type=list)
        slider_max = 1080/5

        cv.namedWindow("edged")
        trackbar_name = 'Alpha x %d' % slider_max
        cv.createTrackbar(trackbar_name, "edged", int(self.edge_limit), int(slider_max), self.on_trackbar)

    def on_trackbar(self, val):
        self.edge_limit = val
        self.log_debug('set', self, 'to', val)

    def process_diff_in(self, diff):
        edged = cv.Canny(diff, 255 / 3, 255)
        cnts = imutils.grab_contours(cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE))
        contours = [c for c in cnts if self.v_diff(c) > self.edge_limit]  # diff.shape[0]/20]
        edged_c = cv.cvtColor(edged, cv.COLOR_GRAY2BGR)
        for cnt in contours:
            rect = cv.minAreaRect(cnt)
            box = cv.boxPoints(rect)
            box = np.int0(box)
            cv.drawContours(edged_c, [box], 0, (0, 0, 255), 2)
        Module.show_image("edged", resize(CVImage(edged, diff.id, diff.camera_info), 0.3))
        self.contours_out.data_ready(Contours(contours, diff.id, diff.camera_info))

    @staticmethod
    def v_diff(c):
        return max([p[0][1] for p in c]) - min([p[0][1] for p in c])


