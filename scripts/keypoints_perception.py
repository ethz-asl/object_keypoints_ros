#!/usr/bin/env python3
import copy
import rospy
import tf2_ros
import numpy as np
from scipy.spatial.transform import Rotation
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseArray
from cv_bridge import CvBridge

from object_keypoints_ros.msg import Keypoints
from object_keypoints_ros.srv import KeypointsDetection, KeypointsDetectionRequest, KeypointsDetectionResponse, \
    KeypointsPerception, KeypointsPerceptionResponse, KeypointsPerceptionRequest

class PerceptionModule:
    def __init__(self):
        self.bridge = CvBridge()
        self.base_frame = rospy.get_param("~base_frame")
        self.camera_frame = rospy.get_param("~camera_frame")
        self.image_topic = rospy.get_param("~image_topic")
        self.depth_topic = rospy.get_param("~depth_topic")
        self.depth_window_size = rospy.get_param("~depth_window_size", 5)
        self.depth_window_mode = rospy.get_param("~depth_window_mode", "median")
        self.max_distance_threshold = rospy.get_param("~max_distance_threshold", 5)
        self.calibration_topic = rospy.get_param("~calibration_topic", "/camera_info")
        self.camera_info: CameraInfo = None

        # Images
        self.depth = None
        self.depth_header = None
        self.rgb = None

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # ROS
        self.calibration_sub = rospy.Subscriber(self.calibration_topic, CameraInfo, self._calibration_callback, queue_size=1)
        self.image_sub = rospy.Subscriber(self.image_topic, Image, self._image_callback, queue_size=1)
        self.depth_sub = rospy.Subscriber(self.depth_topic, Image, self._depth_callback, queue_size=1)
        self.marker_pub = rospy.Publisher("perception", MarkerArray, queue_size=1)
        self.detection_srv_client = rospy.ServiceProxy("detect", KeypointsDetection)
        self.trigger_detection_srv = rospy.Service("perceive", KeypointsPerception, self.perceive)

    def _make_marker(self, id, frame_id, x, y, z):
        marker = Marker()
        marker.id = id
        marker.header.frame_id = frame_id
        marker.type = marker.SPHERE
        marker.action = marker.ADD
        marker.scale.x = 0.04
        marker.scale.y = 0.04
        marker.scale.z = 0.04
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = x
        marker.pose.position.y = y
        marker.pose.position.z = z
        return marker

    def _calibration_callback(self, msg: CameraInfo):
        self.camera_info = msg

    def _depth_callback(self, msg):
        multiplier = 1.0
        if msg.encoding == "32FC1":     # already encoded in meters
            multiplier = 1.0
        elif msg.encoding == "16UC1":   # encoded in millimiters
            multiplier = 0.001

        self.depth_header = msg.header
        self.depth = multiplier * self.bridge.imgmsg_to_cv2(msg, msg.encoding)

    def _image_callback(self, msg: Image):
        self.rgb = msg

    def perceive(self, req: KeypointsPerceptionRequest):
        """
        Uses a keypoint detection service and depth information to perceive 3D points
        :return: 3D keypoint poses
        """
        self.detection_srv_client.wait_for_service(timeout=10)
        rospy.logwarn("Service {} not available yet".format(self.detection_srv_client.resolved_name))

        if self.camera_info is None:
            rospy.logwarn("No camera info received yet, skipping detection")
            return
        if self.depth is None:
            rospy.logwarn("No depth image received yet, skipping detection")
            return
        if self.rgb is None:
            rospy.logwarn("No rgb image received yet, skipping detection")
            return

        req = KeypointsDetectionRequest()
        req_depth = copy.deepcopy(self.depth) # store locally the current depth image as well
        req.rgb = self.rgb
        res_det: KeypointsDetectionResponse = self.detection_srv_client.call(req)

        # The check for the correct number of keypoints is already implemented in the keypoint detection service

        marker_array = MarkerArray() # For visualization
        pose_array = PoseArray()     # For usage
        pose_array.header.frame_id = self.base_frame

        # let it fail if not found
        trans = self.tf_buffer.lookup_transform(self.base_frame, self.camera_frame, self.depth_header.stamp, timeout=rospy.Duration(3.0))
        t_w_cam = np.array([trans.transform.translation.x, trans.transform.translation.y, trans.transform.translation.z])
        q = np.array([trans.transform.rotation.x, trans.transform.rotation.y, trans.transform.rotation.z, trans.transform.rotation.w])
        R_w_cam = Rotation.from_quat(q).as_matrix()

        for i, kpt in enumerate(res_det.detection.keypoints):
            print("Processing keypoint {} {}".format(kpt.x, kpt.y))

            # take a window of depths and extract the median to remove outliers
            # note that the window scales by depth_window_size in all directions
            x_min, x_max = max(int(kpt.x) - self.depth_window_size, 0), min(int(kpt.x) + self.depth_window_size, req_depth.shape[1]-1)
            y_min, y_max = max(int(kpt.y) - self.depth_window_size, 0), min(int(kpt.y) + self.depth_window_size, req_depth.shape[0]-1)

            # # take depth window
            depth_wdw = req_depth[y_min: y_max, x_min: x_max]
            depth_wdw = np.reshape(depth_wdw, (-1,))
            depth_wdw = depth_wdw[depth_wdw>0]  # threshold invalid values
            depth_wdw = depth_wdw[depth_wdw<self.max_distance_threshold]  # threshold far away values

            if len(depth_wdw) == 0:
                raise Exception(f"No valid depth information could be extracted for keypoint ({kpt.x}, {kpt.y}), maybe it is too close to the edge of the image")

            P_cam = np.zeros((3,))
            if self.depth_window_mode == "median":
                P_cam[2] = np.median(depth_wdw)
            elif self.depth_window_mode == "min":
                P_cam[2] = np.min(depth_wdw)
            elif self.depth_window_mode == "max":
                P_cam[2] = np.max(depth_wdw)
            else:
                raise Exception(f"Depth window mode {self.depth_window_mode} is not supported")
            rospy.loginfo(f"Using depth window mode {self.depth_window_mode} with window size {self.depth_window_size}")

            # retrieve the global x, y coordinate of the point given z
            # TODO not accounting for distortion --> would be better to just operate on undistorted images
            P_cam[0] = (kpt.x - self.camera_info.K[2]) * P_cam[2] / self.camera_info.K[0]
            P_cam[1] = (kpt.y - self.camera_info.K[5]) * P_cam[2] / self.camera_info.K[4]

            P_base = R_w_cam @ P_cam + t_w_cam
            marker = self._make_marker(i, self.base_frame, P_base[0], P_base[1], P_base[2])
            marker_array.markers.append(marker)
            pose_array.poses.append(marker.pose)
            print(f"Keypoint {i} perceived at P_cam={P_cam} and P_base={P_base}")

        res = KeypointsPerceptionResponse()
        res.camera_pose.position = trans.transform.translation
        res.camera_pose.orientation = trans.transform.rotation 
        res.keypoints2d = res_det.detection.keypoints
        res.header.stamp = rospy.get_rostime()
        res.header.frame_id = self.camera_frame
        res.camera_info = self.camera_info
        res.keypoints = pose_array
        self.marker_pub.publish(marker_array)
        return res


if __name__ == "__main__":
    rospy.init_node("perception")
    perception = PerceptionModule()
    rospy.spin()
