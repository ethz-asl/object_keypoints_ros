#!/usr/bin/env python3

import os
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

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from object_keypoints_ros.msg import Keypoint, Keypoints
from object_keypoints_ros.srv import KeypointsDetection, KeypointsDetectionResponse, KeypointsDetectionRequest


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
        self.calibration_file = rospy.get_param('~calibration_file', "")
        self.verbose = rospy.get_param('~verbose', False)
    
        self.camera_info_topic = rospy.get_param("~calibration_topic", "/camera_info")
        self.left_image_topic = rospy.get_param("~left_image_topic", "/zedm/zed_node/left_raw/image_raw_color")
        self.camera_info = None
        self.left_image = None
        self.left_image_header = None
        self.bridge = CvBridge()
        self.scale = np.array([1, 1])
        self.global_scale = np.array([1, 1])
        self.input_size = (511, 511)
        self.IMAGE_SIZE = (511, 511)
        self.prediction_size = SceneDataset.prediction_size
        self.info = Info()

        with open(self.keypoint_config_path, 'rt') as f:
            self.keypoint_config = json.load(f)
        
        self.info.append("Initializing keypoint inference pipeline.")
        self.info.append("Loading model from: {}".format(self.model))
        self.info.append("Keypoint config: {}".format(self.keypoint_config))
        self.info.append("Calibration file: {}".format(self.calibration_file))
        self.info.append("Use gpu: {}".format(self.use_gpu))
        self.info.append("Prediction size {}".format(self.prediction_size))
        self.info.append("Verbose: {}".format(self.verbose))
        self.info.print()
        
        self.pipeline = pipeline.LearnedKeypointTrackingPipeline(self.model, self.use_gpu, self.prediction_size, 
                                                                 self.keypoint_config)

        self.rgb_mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=False, dtype=torch.float32)[:, None, None]
        self.rgb_std = torch.tensor([0.25, 0.25, 0.25], requires_grad=False, dtype=torch.float32)[:, None, None]
        
        
        # Dimensions
        self.in_out_scale = np.array(self.input_size) / np.array(self.prediction_size)

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # 3D pose extraction
        self.keypoints_hist = None
        self.proj_matrices = []
        
        # Publishers
        self.left_heatmap_pub = rospy.Publisher("heatmap_left", Image, queue_size=1)
        self.right_heatmap_pub = rospy.Publisher("heatmap_right", Image, queue_size=1)
        self.result_img_pub = rospy.Publisher("result_img", Image, queue_size=1)
        self.image_static_pub = rospy.Publisher(self.left_image_topic, Image, queue_size=1)
        self.heatmap_raw_pub = rospy.Publisher("heatmap_raw", Image, queue_size=1)
        self.annotation_img_pub = rospy.Publisher("annotation", Image, queue_size=1)
        self.keypoints_pub =  rospy.Publisher("keypoints", Keypoints, queue_size=1)

        # Camera models
        if os.path.isfile(self.calibration_file):
            self.left_camera = camera_utils.from_calibration(self.calibration_file)
            self.right_camera = camera_utils.from_calibration(self.calibration_file)
        else:
            rospy.logwarn("No file {} found. Default to CameraInfo on topic {}".format(self.calibration_file, self.camera_info_topic))
            self.camera_info_sub = rospy.Subscriber(self.camera_info_topic, CameraInfo, self._camera_info_cb)
            rate = rospy.Rate(1)
            while self.camera_info == None:
                rospy.logwarn("Waiting for {} msg.".format(self.camera_info_topic))
                rate.sleep()
            rospy.loginfo("Camera info received.")
            self.left_camera = camera_utils.from_msg(self.camera_info)
            self.right_camera = camera_utils.from_msg(self.camera_info)

        # set the camera model to be used in the inference pipeline
        self.pipeline.reset(self.left_camera)

        # Messages
        self.kpts_msg = Keypoints()

    def _camera_info_cb(self, msg):
        self.camera_info = msg

    def step(self):
        inference_img = copy.deepcopy(self.left_image)
        if inference_img is not None:
            img_in = self._preprocess_image(inference_img)
            objects, heatmap = self.pipeline(img_in)

            if len(objects) == 0:
                rospy.logerr("No objects detected")
                self.kpts_msg = Keypoints()
                self.kpts_msg.header.stamp = self.left_image_header.stamp
                return
            
            if self.verbose:
                print_objects(objects)

            raw_heatmap = self._to_heatmap(heatmap[0].numpy(), resize=False)
            left_heatmap = self._to_heatmap(heatmap[0].numpy())
            img_in = img_in.cpu().numpy()[0]
            
            self._publish_result(img_in, left_heatmap, objects)
            self._publish_annotation(inference_img, objects)
            self._publish_heatmap_raw(raw_heatmap, objects)
            self._publish_keypoints(objects, self.left_image_header.stamp)

    ### Utilities and conversions
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
        self.kpts_msg = Keypoints()
        self.kpts_msg.header.stamp = time
        
        keypoints = objects[0]['keypoints']
        n_kpts = len(self.keypoint_config['keypoint_config'])
        if (len(keypoints)-1) != n_kpts:   # the first is a fictious keypoint 
            rospy.logwarn_throttle(2.0, "Not publishing keypoints. Detected less then specified ({} < {})".format(len(keypoints)-1, n_kpts))
            return
    
        for i in range(1, n_kpts+1):
            n_kpt_this_class_found = keypoints[i].shape[0]
            n_kpt_this_class = self.keypoint_config['keypoint_config'][i-1]
            if n_kpt_this_class_found != n_kpt_this_class:
                rospy.logwarn("Category [{}] keypoints: only {}/{} detected".format(i-1, n_kpt_this_class_found, n_kpt_this_class))
                return
            
            for j in range(n_kpt_this_class):
                kpt_msg = Keypoint()
                kpt_msg.x = keypoints[i][j][0] * self.global_scale[0]
                kpt_msg.y = keypoints[i][j][1] * self.global_scale[1]
                self.kpts_msg.keypoints.append(kpt_msg)
        self.keypoints_pub.publish(self.kpts_msg)

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
        result_msg = self.bridge.cv2_to_imgmsg(result_img[:, :, :3], encoding='rgb8')
        result_msg.header = self.left_image_header
        self.result_img_pub.publish(result_msg)

    def _publish_annotation(self, img, objects):
        img = self._draw_keypoints(img, objects, scale=self.global_scale, size=80, thickness=8)
        img_msg = self.bridge.cv2_to_imgmsg(img[:, :, :3], encoding='passthrough')
        self.annotation_img_pub.publish(img_msg)

#### Class specializations

class ObjectKeypointsContinuous(ObjectKeypointPipeline):
    """
    Uses ROS subscribers to image streams to get and process the image
    """
    def __init__(self):
        ObjectKeypointPipeline.__init__(self)
        self.left_sub = rospy.Subscriber(self.left_image_topic, Image, callback=self._left_image_callback, queue_size=1)

    ### Images callbacks
    def _left_image_callback(self, image):
        # this runs async with the step --> bad timing
        img = self.bridge.imgmsg_to_cv2(image, 'rgb8')
        self.left_image = img
        self.left_image_header = image.header
        with torch.no_grad():
            self.step()

    def run(self):
        self.info.append("Up and running!")
        self.info.print()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            rate.sleep()

class ObjectKeypointsStatic(ObjectKeypointPipeline):
    """
    Uses a path to a image file to process the file and provide annotated output (same as per continuous stream)
    """
    def __init__(self):
        ObjectKeypointPipeline.__init__(self)
        self.static_image_ros = None
        self.image_file = rospy.get_param("~image_file", "")
        
        if not os.path.isfile(self.image_file):
            raise NameError("Failed to find file {}".format(self.image_file))
            
        self._process_static_image()

    def _process_static_image(self):
        """ 
        Debug method to parse a static image instead of subscribing to a ROS image stream 
        """
        self.left_image = cv2.imread(self.image_file)
        header = Header()
        header.stamp = rospy.get_rostime()
        self.left_image_header = header
        self.static_image_ros = self.bridge.cv2_to_imgmsg(self.left_image, "rgb8")

    def run(self):
        self.info.append("Up and running!")
        self.info.print()
        rate = rospy.Rate(10)
        
        while not rospy.is_shutdown():
            self.image_static_pub.publish(self.static_image_ros)
            with torch.no_grad():
                self.step()
            rate.sleep()

class ObjectKeypointsService(ObjectKeypointPipeline):
    """
    Implements a perception service, that returns detections on demand
    """
    def __init__(self):
        ObjectKeypointPipeline.__init__(self)
        self.detection_srv = rospy.Service("detect", KeypointsDetection, self._detection_callback)

    def _detection_callback(self, req: KeypointsDetectionRequest):
        self.info.append("Processing detection request...")
        self.info.append("Image at time {}".format(req.rgb.header.stamp.to_sec()))
        self.info.print()

        self.left_image = self.bridge.imgmsg_to_cv2(req.rgb, 'rgb8')
        self.left_image_header = req.rgb.header
        with torch.no_grad():
            self.step()

        res = KeypointsDetectionResponse()
        res.detection = self.kpts_msg
        self.info.append("Detection done!")
        self.info.print()
        return res

    def run(self):
        self.info.append("Up and running!")
        self.info.print()

        rospy.spin()

def main():
    operation_mode = rospy.get_param("~operation_mode", "continous")
    if operation_mode == "continous":
        keypoint_pipeline = ObjectKeypointsContinuous()
    elif operation_mode == "static":
        keypoint_pipeline = ObjectKeypointsStatic()
    elif operation_mode == "service":
        keypoint_pipeline = ObjectKeypointsService()
    else:
        rospy.logerr("Unknown operation mode [{}]".format(operation_mode))
    keypoint_pipeline.run()
    

if __name__ == "__main__":
    rospy.init_node("object_keypoints_ros")
    main()
        
