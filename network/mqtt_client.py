import cv2
import paho.mqtt.client as mqtt
from core.helper import ModuleParameter
from core.module import Module, Input
from core.datatypes import CVImage, MultiImage, JsonObject, BoardCoordinate


class MQTTClient(Module):
    def __init__(self):
        super().__init__()

        self.json_in = Input(data_type=JsonObject)
        self.coordinate_in = Input(data_type=BoardCoordinate)
        self.image_in = Input(data_type=CVImage, config_keys=['cam_ids'], num_worker_threads=4)
        self.multi_image_in = Input(data_type=MultiImage, config_keys=['cam_ids'], num_worker_threads=2)

        self.client = mqtt.Client()
        self.mqtt_host = ModuleParameter("192.168.178.67")
        self.cam_ids = ModuleParameter(None, data_type=list)
        self.last_image = None

    def configure(self,
                  mqtt_host: str = None,
                  record_images: bool = None,
                  image_file: str = None):
        self._configure(locals())

    def process_json_in(self, json: JsonObject):
        self.log_debug('publishing', json.json, 'on', json.topic)
        self.client.publish(json.topic, str(json.json), qos=2)

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
        for image in multi_image.images:
            cam_id = image.camera_info['name']
            topic = image.camera_info['topic'] if 'topic' in image.camera_info else multi_image.source.module_name
            self.log_debug('trying to publish on %s/%s' % (topic, cam_id))
            self.client.publish("%s/%s" % (topic, cam_id), cv2.imencode('.jpg', image)[1].tostring(), qos=2)

    def __start__(self):
        self.client.on_connect = self.on_connect
        self.client.connect(self.mqtt_host, 1883, 60)
        self.client.loop_start()

    def __stop__(self):
        self.client.loop_stop()
        self.client.disconnect()

    # The callback for when the client receives a CONNACK response from the server.
    def on_connect(self, client, userdata, flags, rc):
        self.log_debug("Connected with result code " + str(rc))

