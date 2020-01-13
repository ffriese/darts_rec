import imutils
import numpy as np

from core.helper import ModuleParameter
from core.module import Module, Input, Output
from core.datatypes import CVImage, Contours, MultiImage, ContourCollection
from core.convenience import resize
import cv2 as cv

class EdgeDetection(Module):
    def __init__(self):
        super().__init__()
        self.diff_in = Input(data_type=MultiImage, config_keys=['cam_ids'])
        self.edged_out = Output(data_type=MultiImage, config_keys=['cam_ids'])
        self.contours_out = Output(data_type=ContourCollection, config_keys=['cam_ids'])
        self.edge_limit = ModuleParameter(int(1080/20), data_type=int)

        self.cam_ids = ModuleParameter(None, data_type=list)
        slider_max = 1080/5

        # cv.namedWindow("edged")
        # trackbar_name = 'Edge Limit'
        # cv.createTrackbar(trackbar_name, "edged", int(self.edge_limit), int(slider_max), self.on_trackbar)

    def on_trackbar(self, val):
        self.edge_limit = val

    def process_diff_in(self, diffs: MultiImage):
        images = []
        contour_collection = []
        for diff in diffs.images:
            edged = cv.Canny(diff, 255 / 3, 255)
            cnts = imutils.grab_contours(cv.findContours(edged.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE))
            contours = [c for c in cnts if self.v_diff(c) > self.edge_limit]  # diff.shape[0]/20]
            largest = sorted(contours, key=self.a_len, reverse=True)[:10]
            if largest:
                # contour = self.a_len(largest[0])
                contour_collection.append(Contours(contours, diff.id, diff.camera_info))
            edged_c = cv.cvtColor(edged, cv.COLOR_GRAY2BGR)
            for cnt in contours:
                rect = cv.minAreaRect(cnt)
                box = cv.boxPoints(rect)
                box = np.int0(box)
                cv.drawContours(edged_c, [box], 0, (0, 0, 255), 2)
            # Module.show_image("edged", resize(CVImage(edged, diff.id, diff.camera_info), 0.3))
            images.append(CVImage(edged, diff.id, diff.camera_info))

        self.edged_out.data_ready(MultiImage(images))
        self.contours_out.data_ready(ContourCollection(contour_collection))

    @staticmethod
    def v_diff(c):
        return max([p[0][1] for p in c]) - min([p[0][1] for p in c])

    @staticmethod
    def a_len(c):
        return cv.arcLength(c, True)
