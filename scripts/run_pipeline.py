#!/usr/bin/env python3

# To find catkin python3 build of tf2_py
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
from perception.datasets.video import StereoVideoDataset
from perception import pipeline
from perception.utils import camera_utils
from matplotlib import cm
from vision_msgs.msg import BoundingBox3D

import utils

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
        left_image_topic = rospy.get_param("~left_image_topic", "/zedm/zed_node/left_raw/image_raw_color")
        right_image_topic = rospy.get_param("~right_image_topic", "/zedm/zed_node/right_raw/image_raw_color")
        self.left_camera_frame = rospy.get_param('~left_camera_frame')
        self.left_image = None
        self.left_image_ts = None
        self.right_image = None
        self.right_image_ts = None
        self.bridge = CvBridge()
        self.scale = (1, 1)
        self.input_size = (511, 511)
        self.IMAGE_SIZE = (511, 511)
        self.prediction_size = StereoVideoDataset.prediction_size

        self.keypoint_config_path = rospy.get_param('~keypoints')
        with open(self.keypoint_config_path, 'rt') as f:
            self.keypoint_config = json.load(f)

        model = rospy.get_param('~load_model', "/home/ken/Hack/catkin_ws/src/object_keypoints/model/modelv2.pt")
        rospy.loginfo("[ObjectKeypointPipeline] Loading model: {}".format(model))

        self.use_gpu = rospy.get_param("~gpu", True)
        rospy.loginfo("[ObjectKeypointPipeline] Using gpu? {}".format(self.use_gpu))
        
        self.pnp = rospy.get_param("~pnp", True)
        rospy.loginfo("[ObjectKeypointPipeline] PnP? {}".format(self.pnp))
        
        if self.pnp:
            self.pipeline = pipeline.PnPKeypointPipeline(model, self._read_keypoints(), torch.cuda.is_available())
        else:
            # self.pipeline = pipeline.LearnedKeypointTrackingPipeline(model, torch.cuda.is_available(),
            #                 self.prediction_size,self._read_keypoints(), self.keypoint_config)
            self.pipeline = pipeline.LearnedKeypointTrackingPipeline(model, self.use_gpu,
                self.prediction_size, self.keypoint_config)

        self.rgb_mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=False, dtype=torch.float32)[:, None, None]
        self.rgb_std = torch.tensor([0.25, 0.25, 0.25], requires_grad=False, dtype=torch.float32)[:, None, None]
        
        self._read_calibration()
        scaling_factor = np.array(self.image_size) / np.array(self.prediction_size)
        
        # scale between input and output of the network
        self.in_out_scale = np.array(self.input_size) / np.array(self.prediction_size)

        #self.pipeline.reset(self.K, self.Kp, self.D, self.Dp, self.T_RL, scaling_factor)
        self.pipeline.reset(self.stereo_camera)

        self._compute_bbox_dimensions()

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # 3D pose extraction
        self.keypoints_hist = None
        self.proj_matrices = []
        
        # Subscribers
        self.left_sub = rospy.Subscriber(left_image_topic, Image, callback=self._left_image_callback, queue_size=1)
        self.right_sub = rospy.Subscriber(right_image_topic, Image, callback=self._right_image_callback, queue_size=1)
        
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
        
        # Only used if an object mesh is set.
        if self.bbox_size is not None:
            self.bbox_pub = rospy.Publisher("object_keypoints_ros/bbox", BoundingBox3D, queue_size=1)
        else:
            self.bbox_pub = None

    def _get_camera_projection(self, ts):
        msg = self.tf_buffer.lookup_transform('world', self.left_camera_frame, ts, rospy.Duration(6.0))
        T = np.zeros((3, 4))
        t = msg.transform.translation
        r = msg.transform.rotation
        R = Rotation.from_quat([r.x, r.y, r.z, r.w])
        T[:3, 3]  = np.array([t.x, t.y, t.z])
        T[:3, :3] = R.as_matrix()
        return self.K @ T

    def _read_calibration(self):
        path = rospy.get_param('~calibration')
        params = camera_utils.load_calibration_params(path)
        self.K = params['K']
        self.Kp = params['Kp']
        self.D = params['D']
        self.Dp = params['Dp']
        self.T_RL = params['T_RL']
        self.image_size = params['image_size']
      
        left_camera = camera_utils.FisheyeCamera(params['K'], params['D'], params['image_size'])
        right_camera = camera_utils.FisheyeCamera(params['Kp'], params['Dp'], params['image_size'])
        self.left_camera = left_camera
        self.right_camera = right_camera

        self.stereo_camera = camera_utils.StereoCamera(self.left_camera, self.right_camera, params['T_RL'])

        #self.stereo_camera_small = camera_utils.StereoCamera(left_camera, right_camera, params['T_RL'])


    def _read_keypoints(self):  # not included in prediction process
        path = rospy.get_param('~keypoints')
        with open(path, 'rt') as f:
            return np.array(json.loads(f.read())['3d_points'])

    def _compute_bbox_dimensions(self):
        mesh_file = rospy.get_param('~object_mesh', None)
        if mesh_file is not None:
            bounding_box = utils.compute_bounding_box(mesh_file)
            # Size is in both directions, surrounding the object from the object center.
            self.bbox_size = (bounding_box.max(axis=0) - bounding_box.min(axis=0)) * 0.5
        else:
            self.bbox_size = None

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
            for kp in left_keypoints:
                kp = kp.round().astype(int)
                left = cv2.circle(left, (kp[0], kp[1]), radius=2, color=(0, 255, 0), thickness=1)
            left_msg = self.bridge.cv2_to_imgmsg(left[:, :, :3], encoding='passthrough')
            self.left_heatmap_pub.publish(left_msg)

        if right is not None:
            right = ((right + 1.0) * 0.5).sum(axis=0)
            right = np.clip(cm.inferno(right) * 255.0, 0, 255.0).astype(np.uint8)
            for kp in right_keypoints:
                kp = kp.round().astype(int)
                right = cv2.circle(right, (kp[0], kp[1]), radius=1, color=(0, 255, 0, 100), thickness=-1)
            right_msg = self.bridge.cv2_to_imgmsg(right[:, :, :3], encoding='passthrough')
            self.right_heatmap_pub.publish(right_msg)

    def _publish_bounding_box(self, T, pose_msg):
        if self.bbox_size is not None:
            msg = BoundingBox3D()
            msg.pose = pose_msg.pose
            msg.size.x = self.bbox_size[0]
            msg.size.y = self.bbox_size[1]
            msg.size.z = self.bbox_size[2]
            self.bbox_pub.publish(msg)

    def _publish_result(self,image):
        #print("result: ", image.shape)
        image_msg = self.bridge.cv2_to_imgmsg(image[:, :, :3], encoding='passthrough')
        self.result_img_pub.publish(image_msg)

    def _to_heatmap(self, target):
        target = np.clip(target, 0.0, 1.0)
        target = target.sum(axis=0)
        target = (cm.inferno(target) * 255.0).astype(np.uint8)[:, :, :3]
        return cv2.resize(target[:, :, :3], self.IMAGE_SIZE)

    def _to_image(self, frame):
        frame = StereoVideoDataset.to_image(frame)
        return cv2.resize(frame, self.IMAGE_SIZE)

    def step(self):
        print("In step")
        I = torch.eye(4)[None]
        if self.left_image is not None:
            left_image = self._preprocess_image(self.left_image)
            objects, heatmap = self.pipeline(left_image)
            # self.left_image = None
 
            heatmap_left = self._to_heatmap(heatmap[0].numpy())
  
            left_image = left_image.cpu().numpy()[0]

            left_rgb = self._to_image(left_image)
            image_left = (0.3 * left_rgb + 0.7 * heatmap_left).astype(np.uint8)
            points_left = []
            for obj in objects:
                p_left = np.concatenate([p + 1.0 for p in obj['keypoints_left'] if p.size != 0], axis=0)
                rospy.loginfo(p_left)
                points_left.append(p_left)
                #points_right.append(p_right)

            for object_points in points_left:
                for point in object_points:
                    point_int = np.asarray([point[0] * self.in_out_scale[0] * self.scale[0], 
                                            point[1] * self.in_out_scale[1] * self.scale[1]], dtype=int)
                    self.left_image = cv2.drawMarker(self.left_image, point_int, (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)

            # accumulate keypoints for triangulation
            # if len(points_left) > 0:
            #     if len(points_left[0]) == 5: # TODO(giuseppe) hard coded for now
            #         kpts_new = np.asarray(points_left[0]).T
            #         if self.keypoints_hist is None:
            #             self.keypoints_hist = kpts_new
            #         else:
            #             self.keypoints_hist = np.hstack((self.keypoints_hist, kpts_new))
            #         self.proj_matrices.append(self._get_camera_projection(self.left_image_ts))

            # self._publish_keypoints(out['keypoints'][0], self.left_image_ts)
            # self._publish_pose(out['pose'][0], self.left_image_ts)
            self._publish_heatmaps(left=heatmap_left, left_keypoints=objects[0]['keypoints_left'])
            self._publish_result(self.left_image)
            #print("out ", objects)
            # self._publish_keypoints(objects[0]['keypoints_left'], self.left_image_ts)
    
def linear_triangulation(keypoints, proj_matrices):
    """
    keypoints: list of N_kpts ordered keypoints expressed as a [2 x N_meas] matrix
    proj_matrices: list of N_meas projection matrices expressed as a [3 x 4] matrices

    return: [3 x N_kpts] matrix of triangulated points in homogeneous coordinates
    """
    
    N_kpts = len(keypoints)
    assert N_kpts > 0

    N_meas = keypoints[0].shape[1]
    assert N == len(proj_matrices)

    points = np.zeros((4, N_meas))
    points[3, :] = np.ones((1, N_meas))
    
    
    for i in range(N_kpts):
        for j in range(N_meas):
            if j == 0:
                A = np.skew(keypoints[i][:, j]) @ proj_matrices[j]    
            else:
                A = np.vstack((A, np.skew(keypoints[i][:, j]) @ proj_matrices[j]))

        u, s, vh = np.linalg.svd(A, full_matrices=True)

        # the solution is the eigenvector corresponding to the minimum non zero eigenvalue
        # which is the corresponding column in v
        points[:, i] = vh[-1, :].T
        points[:, i] /= points[3, i]  # make the points homogeneous     

    return points[:3, :]

if __name__ == "__main__":
    with torch.no_grad():
        rospy.init_node("object_keypoints_ros")
        keypoint_pipeline = ObjectKeypointPipeline()

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

