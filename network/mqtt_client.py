import json
from enum import Enum

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from core.helper import ModuleParameter
from core.module import Module, Input, Output
from core.datatypes import CVImage, MultiImage, JsonObject, BoardCoordinate


class CalibrationMode(Enum):
    NONE = 0
    HEADLESS_SERVER = 1
    VISUAL_CALIBRATOR = 2


class MQTTClient(Module):
    def __init__(self):
        super().__init__()

        self.json_in = Input(data_type=JsonObject)
        self.coordinate_in = Input(data_type=BoardCoordinate)
        self.image_in = Input(data_type=CVImage, config_keys=['cam_ids'], num_worker_threads=2)
        self.multi_image_in = Input(data_type=MultiImage, config_keys=['cam_ids'], num_worker_threads=2)

        self.calibration_image_out = Output(data_type=CVImage)
        self.calibration_config_out = Output(data_type=JsonObject)

        self.calibration_mode = ModuleParameter(CalibrationMode.NONE)
        self.mqtt_host = ModuleParameter("192.168.178.67")
        self.cam_ids = ModuleParameter(None, data_type=list)

        self.client = mqtt.Client()
        self.calibration_image_published = False

    def configure(self,
                  mqtt_host: str = None,
                  calibration_mode: CalibrationMode = None):
        self._configure(locals())

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        self.log_debug("Connected with result code " + str(rc))
        if self.calibration_mode == CalibrationMode.HEADLESS_SERVER:
            self.client.subscribe('calibration/data/#', qos=2)
        elif self.calibration_mode == CalibrationMode.VISUAL_CALIBRATOR:
            self.client.subscribe('calibration/#', qos=2)

    def process_json_in(self, json: JsonObject):
        self.log_debug('publishing', json.get_dict(), 'on', json.topic)
        self.client.publish(json.topic, json.get_string(), qos=2)

    def process_coordinate_in(self, coord: BoardCoordinate):
        topic = 'board_coordinate'
        self.log_debug('publishing', coord.point, 'on', topic)
        json_str = '{"x":%f, "y":%f}' % coord.point
        self.client.publish(topic, json_str, qos=2)

    def process_image_in(self, image: CVImage):
        topic = image.camera_info['topic'] if 'topic' in image.camera_info else image.source.module_name
        self.log_debug('trying to publish on %s' % topic)
        self.client.publish(topic, cv2.imencode('.png', image)[1].tostring(), qos=2)

    def process_multi_image_in(self, multi_image: MultiImage):
        if self.calibration_mode == CalibrationMode.NONE:
            for image in multi_image.images:
                cam_id = image.camera_info['name']
                topic = image.camera_info['topic'] if 'topic' in image.camera_info else multi_image.source.module_name
                self.log_debug('trying to publish on %s/%s' % (topic, cam_id))
                self.client.publish("%s/%s" % (topic, cam_id), cv2.imencode('.jpg', image)[1].tostring(),
                                    qos=2, retain=True)
        else:
            if not self.calibration_image_published:
                self.calibration_image_published = True
                for image in multi_image.images:
                    self.log_debug('retaining image %s' % image.cam_id())
                    self.client.publish("calibration/image/%s" % image.cam_id(),
                                        cv2.imencode('.jpg', image)[1].tostring(),
                                        retain=True, qos=2)
                    self.client.publish("calibration/data/old_calibration/%s" % image.cam_id(),
                                        json.dumps(image.camera_info['calibration']),
                                        retain=True, qos=2)

    def __start__(self):
        self.client.on_connect = self.on_connect
        self.client.connect(self.mqtt_host, 1883, 60)
        if self.calibration_mode == CalibrationMode.HEADLESS_SERVER:
            self.client.message_callback_add("calibration/data/new_calibration/#", self.on_client_calibration)
        elif self.calibration_mode == CalibrationMode.VISUAL_CALIBRATOR:
            self.client.message_callback_add("calibration/image/#", self.on_image)
            self.client.message_callback_add("calibration/data/old_calibration/#", self.on_server_calibration)
        self.client.loop_start()

    def __stop__(self):
        self.client.loop_stop()
        self.client.disconnect()

    def on_client_calibration(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
        self.calibration_image_published = False
        self.calibration_config_out.data_ready(JsonObject(json_obj=msg.payload.decode(), topic=msg.topic))

    def on_server_calibration(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
        self.calibration_config_out.data_ready(JsonObject(json_obj=msg.payload.decode(), topic=msg.topic))

    def on_image(self, client: mqtt.Client, userdata, msg: mqtt.MQTTMessage):
        cam_id = int(msg.topic.replace('calibration/image/', ''))
        np_arr = np.fromstring(msg.payload, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        self.calibration_image_out.data_ready(CVImage(image, msg.timestamp, {'name': cam_id}))
