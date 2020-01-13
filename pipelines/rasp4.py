import time
import platform

from core.experiment import Experiment
from core.module import Module
from network.camera_grabber import CameraGrabber
from network.mqtt_client import MQTTClient
from processing.background_subtraction import BackgroundSubtraction
from processing.calibrator import Calibrator
from processing.clean_difference import CleanDifference
from processing.edge_detection import EdgeDetection
from processing.fit_line import FitLine
from processing.project_on_board import ProjectOnBoard


class RaspiSystem(Experiment):
    def __init__(self):
        Module.__ENABLE_IM_SHOWS__ = platform.uname()[1] == 'iceberg'
        super().__init__()
        self.grabber = CameraGrabber()
        self.network_client = MQTTClient()
        self.calibrator = Calibrator()
        self.bg_sub = BackgroundSubtraction()
        self.clean_diff = CleanDifference()
        self.edge_det = EdgeDetection()
        self.fit_line = FitLine()
        self.project = ProjectOnBoard()

    def connect(self):
        self.grabber.images_out.connect(self.calibrator.raw_images_in)
        self.calibrator.calibrated_images_out.connect(self.bg_sub.images_in)
        self.calibrator.calibrated_images_out.connect(self.fit_line.raw_images_in)
        self.bg_sub.synced_foregrounds_out.connect(self.clean_diff.foregrounds_in)
        self.clean_diff.diff_out.connect(self.edge_det.diff_in)
        self.edge_det.contours_out.connect(self.fit_line.contour_collection_in)
        self.fit_line.impact_points_out.connect(self.project.impact_points_in)

        self.project.coordinate_out.connect(self.network_client.coordinate_in)

        self.project.dartboard_out.connect(self.network_client.image_in)
        self.edge_det.edged_out.connect(self.network_client.multi_image_in)
        self.fit_line.debug_images_out.connect(self.network_client.multi_image_in)

    def configure(self):
        self.grabber.configure(cam_ids=[0, 1])
        self.bg_sub.configure(enable_debug_images=False)
        self.network_client.configure(mqtt_host='localhost')


if __name__ == '__main__':
    sd = RaspiSystem()
    sd.start()
