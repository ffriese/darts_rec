import numpy as np
import pickle

from core.helper import ModuleParameter
from core.module import Module, Output, os, sys, Thread
from core.datatypes import CVImage
sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


class ROSGrabber(Module):
    def __init__(self):
        super().__init__()

        self.image_out = Output(data_type=CVImage, config_keys=['cam_ids'])
        self.cv_bridge = CvBridge()
        self.sub = None
        self.bgs = []
        self.images = []

        self.ros_master_uri = ModuleParameter("http://dartscam:11311")
        self.cam_ids = ModuleParameter([0, 1], data_type=list)

        self.record_images = ModuleParameter(False)
        self.image_file = ModuleParameter("IMAGES")

        self.image_out.emit_configuration({'cam_ids': self.cam_ids})

    def configure(self,
                  ros_master_uri: str = None,
                  record_images: bool = None,
                  image_file: str = None):
        self._configure(locals())

    def collect(self, ros_image, cam_name):
        self.image_out.data_ready(CVImage(self.cv_bridge.compressed_imgmsg_to_cv2(ros_image),
                                              ros_image.header.frame_id, {'name': cam_name}))
        if self.record_images:
            self.log_debug('recording', ros_image.header.frame_id, cam_name)
            self.images.append(CVImage(self.cv_bridge.compressed_imgmsg_to_cv2(ros_image),
                                       ros_image.header.frame_id, {'name': cam_name}))

    def collect_top(self, ros_image):
        self.collect(ros_image, 0)

    def collect_left(self, ros_image):
        self.collect(ros_image, 1)

    def __start__(self):
        os.environ["ROS_MASTER_URI"] = self.ros_master_uri
        self.log_debug('connecting to master at', os.environ["ROS_MASTER_URI"])
        rospy.init_node('DARTS_GRABBER', disable_signals=True)
        self.log_debug('connected')
        self.sub = rospy.Subscriber("/darts_image/0", CompressedImage, callback=self.collect_left)
        self.sub = rospy.Subscriber("/darts_image/1", CompressedImage, callback=self.collect_top)

    def __stop__(self):
        if self.record_images:
            with open(self.image_file, 'wb') as img_file:
                pickle.dump([{'image': img, 'id': img.id, 'camera_info': img.camera_info} for img in self.images], img_file)
            self.log_info("Saved", len(self.images), "images to", self.image_file)
