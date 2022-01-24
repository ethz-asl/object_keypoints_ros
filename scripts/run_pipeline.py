#!/usr/bin/env python3

# To find catkin python3 build of tf2_py
from curses import raw
import os
import sys

from numpy.core.fromnumeric import size
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
import json

import rospy
import message_filters
import torch
import numpy as np
import cv2
import tf2_ros
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from geometry_msgs.msg import PointStamped, PoseStamped
from perception.utils import ros as ros_utils
from sensor_msgs.msg import Image
from perception.datasets.video import SceneDataset
from perception import pipeline
from perception.utils import camera_utils
from matplotlib import cm
from vision_msgs.msg import BoundingBox3D

import utils

class Info:
    def __init__(self) -> None:
        self.string = ""
        self.name = "[ObjectKeypointPipeline]"
    
    def append(self, msg):
        self.string += self.name + " " + msg + "\n"

    def print(self):
        rospy.loginfo(self.string)
        self.string = ""


def print_objects(obj):
    print("====================================")
    print("Inference results:")
    for i, o in enumerate(obj):
        print("Object [{}]".format(i))
        print("p_centers")
        for pc in o['p_centers']:
            print(pc)
        print("keypoints")
        for kp in o['keypoints']:
            print(kp[0]) # dim is 1 x 2
        print("3d points")
        for p3d in o['p_C']:
            print(p3d[0])
           

def _to_msg(keypoint, time, frame):
    msg = PointStamped()
    msg.header.stamp = time
    msg.header.frame_id = frame
    msg.point.x = keypoint[0]
    msg.point.y = keypoint[1]
    msg.point.z = keypoint[2]
    return msg


class ObjectKeypointPipeline:
    def __init__(self):
        # read ros parameters
        self.use_gpu = rospy.get_param("~gpu", True)
        self.model = rospy.get_param('~load_model', "/home/ken/Hack/catkin_ws/src/object_keypoints/model/modelv2.pt")
        self.keypoint_config_path = rospy.get_param('~keypoints', "")
        self.image_file = rospy.get_param("~image_file", "")
        self.calibration_file = rospy.get_param('~calibration')
        self.verbose = rospy.get_param('~verbose', False)
    
        self.left_image_topic = rospy.get_param("~left_image_topic", "/zedm/zed_node/left_raw/image_raw_color")
        self.right_image_topic = rospy.get_param("~right_image_topic", "/zedm/zed_node/right_raw/image_raw_color")
        self.left_camera_frame = rospy.get_param('~left_camera_frame')
        self.left_image = None
        self.left_image_ts = None
        self.right_image = None
        self.right_image_ts = None
        self.bridge = CvBridge()
        self.scale = (1, 1)
        self.input_size = (511, 511)
        self.IMAGE_SIZE = (511, 511)
        self.prediction_size = SceneDataset.prediction_size
        self.predict_on_static_image = False
        self.static_image_ros = None
        self.info = Info()

        with open(self.keypoint_config_path, 'rt') as f:
            self.keypoint_config = json.load(f)

        if self.image_file != "":
            self.predict_on_static_image = True
            self._process_static_image()
        
        self.info.append("Initializing keypoint inference pipeline.")
        self.info.append("Loading model from: {}".format(self.model))
        self.info.append("Keypoint config: {}".format(self.keypoint_config))
        self.info.append("Calibration file: {}".format(self.calibration_file))
        self.info.append("Use gpu: {}".format(self.use_gpu))
        self.info.append("Prediction size {}".format(self.prediction_size))
        self.info.append("Static image file: {}".format(self.image_file))
        self.info.append("Verbose: {}".format(self.verbose))
        if not self.predict_on_static_image:
            self.info.append("Static image file is empty... subscribing to ROS topic instead.") 
        self.info.print()
        
        self.pipeline = pipeline.LearnedKeypointTrackingPipeline(self.model, self.use_gpu, self.prediction_size, 
                                                                 self.keypoint_config)

        self.rgb_mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=False, dtype=torch.float32)[:, None, None]
        self.rgb_std = torch.tensor([0.25, 0.25, 0.25], requires_grad=False, dtype=torch.float32)[:, None, None]
        
        # Camera models
        self.left_camera = camera_utils.from_calibration(self.calibration_file)
        self.right_camera = camera_utils.from_calibration(self.calibration_file)
        
        # Dimensions
        scaling_factor = np.array(self.left_camera.image_size) / np.array(self.prediction_size)
        
        # scale between input and output of the network
        self.in_out_scale = np.array(self.input_size) / np.array(self.prediction_size)

        # set the camera model to be used in the inference pipeline
        self.pipeline.reset(self.left_camera)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # 3D pose extraction
        self.keypoints_hist = None
        self.proj_matrices = []
        
        # Publishers
        self.center_point_publisher = rospy.Publisher("object_keypoints_ros/center", PointStamped, queue_size=1)
        self.point0_pub = rospy.Publisher("object_keypoints_ros/0", PointStamped, queue_size=1)
        self.point1_pub = rospy.Publisher("object_keypoints_ros/1", PointStamped, queue_size=1)
        self.point2_pub = rospy.Publisher("object_keypoints_ros/2", PointStamped, queue_size=1)
        self.point3_pub = rospy.Publisher("object_keypoints_ros/3", PointStamped, queue_size=1)
        self.left_heatmap_pub = rospy.Publisher("object_keypoints_ros/heatmap_left", Image, queue_size=1)
        self.right_heatmap_pub = rospy.Publisher("object_keypoints_ros/heatmap_right", Image, queue_size=1)
        self.pose_pub = rospy.Publisher("object_keypoints_ros/pose", PoseStamped, queue_size=1)
        self.result_img_pub = rospy.Publisher("object_keypoints_ros/result_img", Image, queue_size=1)
        self.image_static_pub = rospy.Publisher(self.left_image_topic, Image, queue_size=1)
        self.heatmap_raw_pub = rospy.Publisher("object_keypoints_ros/heatmap_raw", Image, queue_size=1)
        
        # Subscribers
        if self.predict_on_static_image:
            self.left_sub = rospy.Subscriber(self.left_image_topic, Image, callback=self._left_image_callback, queue_size=1)
            self.right_sub = rospy.Subscriber(self.right_image_topic, Image, callback=self._right_image_callback, queue_size=1)
        
    def run(self):
        self.info.append("Up and running!")
        self.info.print()
        rate = rospy.Rate(1)
        
        while not rospy.is_shutdown():
            if self.predict_on_static_image:
                self.image_static_pub.publish(self.static_image_ros)
                with torch.no_grad():
                    self.step()
            rate.sleep()

    def _process_static_image(self):
        """ 
        Debug method to debug a static image instead of subscribing to a ROS image stream 
        """
        self.left_image = cv2.imread(self.image_file)
        self.left_image_ts = rospy.get_rostime()
        self.static_image_ros = self.bridge.cv2_to_imgmsg(self.left_image, "rgb8")

    
    def _get_camera_projection(self, ts):
        msg = self.tf_buffer.lookup_transform('world', self.left_camera_frame, ts, rospy.Duration(6.0))
        T = np.zeros((3, 4))
        t = msg.transform.translation
        r = msg.transform.rotation
        R = Rotation.from_quat([r.x, r.y, r.z, r.w])
        T[:3, 3]  = np.array([t.x, t.y, t.z])
        T[:3, :3] = R.as_matrix()
        return self.K @ T


    def _right_image_callback(self, image):
        img = self.bridge.imgmsg_to_cv2(image, 'rgb8')
        self.right_image = img
        self.right_image_ts = image.header.stamp

    def _left_image_callback(self, image):
        # this runs async with the step --> bad timing
        img = self.bridge.imgmsg_to_cv2(image, 'rgb8')
        self.left_image = img
        self.left_image_ts = image.header.stamp
        with torch.no_grad():
            self.step()

    def _preprocess_image(self, image):
        self.scale = (image.shape[1] / self.input_size[1], image.shape[0] / self.input_size[0])
        image = image.transpose([2, 0, 1])
        image = torch.tensor(image / 255.0, dtype=torch.float32)
        image -= self.rgb_mean
        image /= self.rgb_std
        image = image[None]
        return torch.nn.functional.interpolate(image, size=self.input_size, mode='bilinear', align_corners=False).detach()

    def _publish_keypoints(self, keypoints, time):
        for i in range(min(keypoints.shape[0], 4)):
            msg = _to_msg(keypoints[i], rospy.Time(0), self.left_camera_frame)
            getattr(self, f'point{i}_pub').publish(msg)

    def _publish_pose(self, pose_msg, time):
        pose_msg = ros_utils.transform_to_pose(T, self.left_camera_frame, rospy.Time(0))
        self.pose_pub.publish(pose_msg)
        self._publish_bounding_box(pose_msg)

    def _publish_heatmaps(self, left=None, right=None, left_keypoints=None, right_keypoints=None):

        if left is not None:
            left = ((left + 1.0) * 0.5).sum(axis=0)
            left = np.clip(cm.inferno(left) * 255.0, 0, 255.0).astype(np.uint8)
            left_msg = self.bridge.cv2_to_imgmsg(left[:, :, :3], encoding='passthrough')
            self.left_heatmap_pub.publish(left_msg)

        if right is not None:
            right = ((right + 1.0) * 0.5).sum(axis=0)
            right = np.clip(cm.inferno(right) * 255.0, 0, 255.0).astype(np.uint8)
            right_msg = self.bridge.cv2_to_imgmsg(right[:, :, :3], encoding='passthrough')
            self.right_heatmap_pub.publish(right_msg)

    def _publish_heatmap_raw(self, hmap, objects):
        # take only the keypoints belonging to the first object
        obj = objects[0]

        # the object keypoints field is a list of keypoints.
        # the first entry is the geometric center of the keypoints
        # entry 1, ... corresponds to the semantic class and has as many keypoints as the number of keypoints
        # in the corresponding class
        for points_set in obj['keypoints']:
            n_points_in_class = points_set.shape[0]
            for i in range(n_points_in_class):
                point = (points_set[i][:] + 1.0).astype(int)
                hmap = cv2.drawMarker(hmap, (point[0], point[1]), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=5, thickness=1)
        
        self.heatmap_raw_pub.publish(self.bridge.cv2_to_imgmsg(hmap, encoding='passthrough'))
        
    def _publish_result(self,left_image, left_heatmap, objects):
        result_img = self._compute_result_image(left_image, left_heatmap)  
        result_img = self._draw_keypoints(result_img, objects)
        result_msg = self.bridge.cv2_to_imgmsg(result_img[:, :, :3], encoding='passthrough')
        self.result_img_pub.publish(result_msg)

    def _to_heatmap(self, target, resize=True):
        target = np.clip(target, 0.0, 1.0)
        target = target.sum(axis=0)
        target = (cm.inferno(target) * 255.0).astype(np.uint8)[:, :, :3]
        if resize:
            return cv2.resize(target[:, :, :3], self.IMAGE_SIZE)
        else:
            return target.astype(np.uint8)

    def _compute_result_image(self, frame, heatmap):
        """ 
        Combines the predicted heatmap and current frame to obtain a 
        visual results of the network prediction 
        """
        frame = SceneDataset.to_image(frame)
        result_img = cv2.resize(frame, self.IMAGE_SIZE)
        return (0.3 * result_img + 0.7 * heatmap).astype(np.uint8)

    def _draw_keypoints(self, img, objects):
        """
        Draw keypoints on the input image
        objects: the first output of the pipeline -> a list of objects, each define by its keypoints
        """
        # take only the keypoints belonging to the first object
        obj = objects[0]

        # the object keypoints field is a list of keypoints.
        # the first entry is the geometric center of the keypoints
        # entry 1, ... corresponds to the semantic class and has as many keypoints as the number of keypoints
        # in the corresponding class
        for points_set in obj['keypoints']:
            n_points_in_class = points_set.shape[0]
            for i in range(n_points_in_class):
                point = (points_set[i][:] + 1.0).astype(int)
                point[0] = point[0] * self.in_out_scale[0]
                point[1] = point[1] * self.in_out_scale[1]
                img = cv2.drawMarker(img, (point[0], point[1]), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=5)
        
        return img

    def step(self):
        if self.left_image is not None:
            left_image = self._preprocess_image(self.left_image)
            objects, heatmap = self.pipeline(left_image)
            
            if self.verbose:
                print_objects(objects)

            raw_heatmap = self._to_heatmap(heatmap[0].numpy(), resize=False)
            left_heatmap = self._to_heatmap(heatmap[0].numpy())
            left_image = left_image.cpu().numpy()[0]
            
            self._publish_result(left_image, left_heatmap, objects)
            self._publish_heatmap_raw(raw_heatmap, objects)


if __name__ == "__main__":
    with torch.no_grad():
        rospy.init_node("object_keypoints_ros")
        keypoint_pipeline = ObjectKeypointPipeline()
        keypoint_pipeline.run()
    

