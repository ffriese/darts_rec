from core.pipeline import Pipeline
from network.camera_grabber import CameraGrabber
from network.mqtt_client import MQTTClient, CalibrationMode
from processing.metadatawriter import MetaDataWriter


class Recalibrate(Pipeline):
    def __init__(self):
        super().__init__()
        self.grabber = CameraGrabber()
        self.calibrator = MetaDataWriter()
        self.client = MQTTClient()

    def connect(self):
        self.grabber.images_out.connect(self.calibrator.raw_images_in)
        self.calibrator.calibrated_images_out.connect(self.client.multi_image_in)
        self.client.calibration_config_out.connect(self.calibrator.config_in)

    def configure(self):
        self.grabber.configure(cam_ids=[0, 1])
        self.client.configure(calibration_mode=CalibrationMode.HEADLESS_SERVER)


if __name__ == '__main__':
    sd = Recalibrate()
    sd.start()
