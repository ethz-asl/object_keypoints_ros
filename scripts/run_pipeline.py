#!/usr/bin/env python3

import json
import copy
import cv2
import torch
import numpy as np

import rospy
import tf2_ros

from matplotlib import cm
from cv_bridge import CvBridge

from perception import pipeline
from perception.utils import camera_utils
from perception.datasets.video import SceneDataset

from sensor_msgs.msg import Image
from object_keypoints_ros.msg import Keypoint, Keypoints


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
        self.scale = np.array([1, 1])
        self.global_scale = np.array([1, 1])
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
        self.left_heatmap_pub = rospy.Publisher("object_keypoints_ros/heatmap_left", Image, queue_size=1)
        self.right_heatmap_pub = rospy.Publisher("object_keypoints_ros/heatmap_right", Image, queue_size=1)
        self.result_img_pub = rospy.Publisher("object_keypoints_ros/result_img", Image, queue_size=1)
        self.image_static_pub = rospy.Publisher(self.left_image_topic, Image, queue_size=1)
        self.heatmap_raw_pub = rospy.Publisher("object_keypoints_ros/heatmap_raw", Image, queue_size=1)
        self.annotation_img_pub = rospy.Publisher("object_keypoints_ros/annotation", Image, queue_size=1)
        self.keypoints_pub =  rospy.Publisher("object_keypoints_ros/keypoints", Keypoints, queue_size=1)
        
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

    def step(self):
        inference_img = copy.deepcopy(self.left_image)
        if inference_img is not None:
            img_in = self._preprocess_image(inference_img)
            objects, heatmap = self.pipeline(img_in)
            
            if self.verbose:
                print_objects(objects)

            raw_heatmap = self._to_heatmap(heatmap[0].numpy(), resize=False)
            left_heatmap = self._to_heatmap(heatmap[0].numpy())
            img_in = img_in.cpu().numpy()[0]
            
            self._publish_result(img_in, left_heatmap, objects)
            self._publish_annotation(inference_img, objects)
            self._publish_heatmap_raw(raw_heatmap, objects)
            self._publish_keypoints(objects, self.left_image_ts)

    ### Images callbacks
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

    ### Utilities and conversions
    def _process_static_image(self):
        """ 
        Debug method to parsa a static image instead of subscribing to a ROS image stream 
        """
        self.left_image = cv2.imread(self.image_file)
        self.left_image_ts = rospy.get_rostime()
        self.static_image_ros = self.bridge.cv2_to_imgmsg(self.left_image, "rgb8")

        
    def _preprocess_image(self, image):
        self.scale = np.array([image.shape[1] / self.input_size[1], image.shape[0] / self.input_size[0]])
        self.global_scale = self.scale * self.in_out_scale
        image = image.transpose([2, 0, 1])
        image = torch.tensor(image / 255.0, dtype=torch.float32)
        image -= self.rgb_mean
        image /= self.rgb_std
        image = image[None]
        return torch.nn.functional.interpolate(image, size=self.input_size, mode='bilinear', align_corners=False).detach()

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

    def _draw_keypoints(self, img, objects, scale, size=5, thickness=1):
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
                point[0] = point[0] * scale[0]
                point[1] = point[1] * scale[1]
                img = cv2.drawMarker(img, (point[0], point[1]), (0,255,0), markerType=cv2.MARKER_CROSS, markerSize=size, thickness=thickness)
        
        return img

   
    ### Publish methods
    def _publish_keypoints(self, objects, time):
        keypoints = objects[0]['keypoints']
        n_kpts = len(self.keypoint_config['keypoint_config'])
        if (len(keypoints)-1) != n_kpts:   # the first is a fictious keypoint 
            rospy.logwarn_throttle(2.0, "Not publishing keypoints. Detected less then specified ({} < {})".format(len(keypoints)-1, n_kpts))
            return
    
        kpts_msg = Keypoints()
        kpts_msg.header.stamp = time
        for i in range(1, n_kpts+1):
            n_kpt_this_class_found = keypoints[i].shape[0]
            n_kpt_this_class = self.keypoint_config['keypoint_config'][i-1]
            if n_kpt_this_class_found != n_kpt_this_class:
                rospy.logwarn("Category [{}] keypoints: only {}/{} detected".format(i-1, n_kpt_this_class_found, n_kpt_this_class))
                return
            
            for j in range(n_kpt_this_class):
                kpt_msg = Keypoint()
                kpt_msg.x = keypoints[i][j][0]
                kpt_msg.y = keypoints[i][j][1]
                kpts_msg.keypoints.append(kpt_msg)
        self.keypoints_pub.publish(kpts_msg)

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
        hmap = self._draw_keypoints(hmap, objects, scale=(1, 1), size=5)
        self.heatmap_raw_pub.publish(self.bridge.cv2_to_imgmsg(hmap, encoding='passthrough'))
        
    def _publish_result(self,left_image, left_heatmap, objects):
        result_img = self._compute_result_image(left_image, left_heatmap)  
        result_img = self._draw_keypoints(result_img, objects, scale=self.in_out_scale, size=30, thickness=5)
        result_msg = self.bridge.cv2_to_imgmsg(result_img[:, :, :3], encoding='passthrough')
        self.result_img_pub.publish(result_msg)

    def _publish_annotation(self, img, objects):
        img = self._draw_keypoints(img, objects, scale=self.global_scale, size=80, thickness=8)
        img_msg = self.bridge.cv2_to_imgmsg(img[:, :, :3], encoding='passthrough')
        self.annotation_img_pub.publish(img_msg)

if __name__ == "__main__":
    with torch.no_grad():
        rospy.init_node("object_keypoints_ros")
        keypoint_pipeline = ObjectKeypointPipeline()
        keypoint_pipeline.run()
    
