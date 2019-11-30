
from core.helper import ModuleParameter
from core.module import Module, Output, os, sys, Thread, Input
from core.datatypes import CVImage, BoardCoordinate

sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.append('/usr/lib/python2.7/dist-packages')
import rospy
from geometry_msgs.msg import Point

sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')


class ROSCoordinateSender(Module):
    def __init__(self):
        super().__init__()

        self.coordinate_in = Input(data_type=BoardCoordinate)
        self.pub = None
        self.ros_master_uri = ModuleParameter("http://dartscam:11311")

    def configure(self,
                  ros_master_uri: str = None,
                  record_images: bool = None,
                  image_file: str = None):
        self._configure(locals())

    def process_coordinate_in(self, coordinate):
        self.pub.publish(Point(coordinate.point[0], coordinate.point[1], 0.0))

    def __start__(self):
        os.environ["ROS_MASTER_URI"] = self.ros_master_uri
        self.log_debug('connecting to master at', os.environ["ROS_MASTER_URI"])
        try:
            rospy.init_node('DARTS_GRABBER', disable_signals=True)
        except Exception as e:
            print(type(e), e)

        self.log_debug('connected')
        self.pub = rospy.Publisher("/thrown_dart", Point, queue_size=1)

