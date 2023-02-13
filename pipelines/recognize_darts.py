import platform

from core.pipeline import Pipeline
from core.module import Module
from network.camera_grabber import CameraGrabber
from network.mqtt_client import MQTTClient
from processing.background_subtraction import BackgroundSubtraction
from processing.metadatawriter import MetaDataWriter
from processing.clean_difference import CleanDifference
from processing.edge_detection import EdgeDetection
from processing.fit_line import FitLine
from processing.project_on_board import ProjectOnBoard


class RecognizeDarts(Pipeline):
    def __init__(self):
        Module.__ENABLE_IM_SHOWS__ = platform.uname()[1] == 'iceberg'
        super().__init__()
        self.grabber = CameraGrabber()
        self.network_client = MQTTClient()
        self.calibrator = MetaDataWriter()
        self.bg_sub = BackgroundSubtraction()
        self.clean_diff = CleanDifference()
        self.edge_det = EdgeDetection()
        self.fit_line = FitLine()
        self.board_projection = ProjectOnBoard()

    def connect(self):
        # Grabbed images always go through calibrator first to add meta-info
        self.grabber.images_out.connect(self.calibrator.raw_images_in)
        # Next we need to do background-subtraction (event-detection also happens here)
        self.calibrator.calibrated_images_out.connect(self.bg_sub.images_in)
        # If we have an event -> clean that up a bit
        self.bg_sub.synced_foregrounds_out.connect(self.clean_diff.foregrounds_in)
        # Detect edges
        self.clean_diff.diff_out.connect(self.edge_det.diff_in)
        # Fit a line, get a 2D impact point (we need the raw images only to paint that line for debugging purposes)
        self.edge_det.contours_out.connect(self.fit_line.contour_collection_in)
        self.calibrator.calibrated_images_out.connect(self.fit_line.raw_images_in)
        # Project the 2D points onto the board
        self.fit_line.impact_points_out.connect(self.board_projection.impact_points_in)
        # Send coordinate to whoever wants it (the scoring app for instance)
        self.board_projection.coordinate_out.connect(self.network_client.coordinate_in)

        # ALL THESE CONNECTIONS ARE OPTIONAL AND JUST FOR REMOTE DEBUG INFORMATION
        self.grabber.frame_rate_out.connect(self.network_client.json_in)
        self.bg_sub.synced_foregrounds_out.connect(self.network_client.multi_image_in)
        self.clean_diff.diff_out.connect(self.network_client.multi_image_in)
        self.board_projection.dartboard_out.connect(self.network_client.image_in)
        self.edge_det.edged_out.connect(self.network_client.multi_image_in)
        self.fit_line.debug_images_out.connect(self.network_client.multi_image_in)

    def configure(self):
        self.grabber.configure(cam_ids=[0, 1])
        self.bg_sub.configure(enable_debug_images=False)
        self.network_client.configure(mqtt_host='localhost')


if __name__ == '__main__':
    sd = RecognizeDarts()
    sd.start()
