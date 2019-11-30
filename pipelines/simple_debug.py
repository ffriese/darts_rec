from core.experiment import Experiment
from network.file_grabber import FileGrabber
from network.ros_coordinate_sender import ROSCoordinateSender
from network.ros_grabber import ROSGrabber
from processing.background_subtraction import BackgroundSubtraction
from processing.calibrator import Calibrator
from processing.clean_difference import CleanDifference
from processing.edge_detection import EdgeDetection
from processing.fit_line import FitLine
from processing.project_on_board import ProjectOnBoard
from processing.state_machine import StateMachine


class SimpleDebug(Experiment):
    def __init__(self):
        super().__init__()
        # self.grabber = FileGrabber()
        self.grabber = ROSGrabber()
        self.calibrator = Calibrator()
        self.bg_sub = BackgroundSubtraction()
        self.clean_diff = CleanDifference()
        self.edge_detection = EdgeDetection()
        self.state_machine = StateMachine()
        self.line_fitter = FitLine()
        self.projection = ProjectOnBoard()
        self.coordinate_sender = ROSCoordinateSender()

    def connect(self):
        self.grabber.image_out.connect(self.calibrator.raw_image_in)
        self.calibrator.calibrated_image_out.connect(self.bg_sub.raw_image_in)
        self.bg_sub.foreground_out.connect(self.clean_diff.foreground_in)
        self.clean_diff.diff_out.connect(self.edge_detection.diff_in)
        self.edge_detection.contours_out.connect(self.state_machine.contours_in)
        self.state_machine.contour_collection_out.connect(self.line_fitter.contour_collection_in)
        self.state_machine.set_background_trigger_out.connect(self.bg_sub.set_background_trigger_in)

        self.grabber.image_out.connect(self.line_fitter.raw_image_in)

        self.line_fitter.impact_points_out.connect(self.projection.impact_points_in)
        self.projection.coordinate_out.connect(self.coordinate_sender.coordinate_in)

    def configure(self):
        pass


if __name__ == '__main__':
    sd = SimpleDebug()
    sd.start()
