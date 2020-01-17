from core.experiment import Experiment
from network.mqtt_client import MQTTClient, CalibrationMode
from processing.visual_calibration import VisualCalibration


class RemoteVisualCalibration(Experiment):
    def __init__(self):
        super().__init__()
        self.client = MQTTClient()
        self.visual_calibration = VisualCalibration()

    def connect(self):
        self.client.calibration_image_out.connect(self.visual_calibration.raw_image_in)
        self.client.calibration_config_out.connect(self.visual_calibration.config_in)
        self.visual_calibration.config_out.connect(self.client.json_in)

    def configure(self):
        self.client.configure(calibration_mode=CalibrationMode.VISUAL_CALIBRATOR)


if __name__ == '__main__':
    sd = RemoteVisualCalibration()
    sd.start()
