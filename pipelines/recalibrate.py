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
from processing.warper import Warper


class Recalibrate(Experiment):
    def __init__(self):
        super().__init__()
        #self.grabber = FileGrabber()
        self.grabber = ROSGrabber()
        self.calibrator = Calibrator()
        #self.calibrator = Warper()

    def connect(self):
        self.grabber.image_out.connect(self.calibrator.raw_image_in)

    def configure(self):
        pass


if __name__ == '__main__':
    sd = Recalibrate()
    sd.start()
