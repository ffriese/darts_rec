from core.experiment import Experiment
from network.ros_grabber import ROSGrabber


class RecordImages(Experiment):
    def __init__(self):
        super().__init__()
        self.grabber = ROSGrabber()

    def connect(self):
        pass

    def configure(self):
        self.grabber.configure(record_images=True, image_file="IMAGES")


if __name__ == '__main__':
    sd = RecordImages()
    sd.start()
