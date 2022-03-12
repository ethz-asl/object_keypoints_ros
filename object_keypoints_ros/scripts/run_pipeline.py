#!/usr/bin/env python3

# To find catkin python3 build of tf2_py
from cgitb import grey
import enum
import imp
from pickle import NONE
import sys
from matplotlib.pyplot import axis

from numpy.core.fromnumeric import size
import json

from sklearn.feature_extraction import img_to_graph
import rospy
import torch
import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
from perception.utils import ros as ros_utils
from sensor_msgs.msg import Image
from perception.datasets.video import SceneDataset
from perception import pipeline
from perception.utils import camera_utils
from matplotlib import cm
from vision_msgs.msg import BoundingBox3D
import utils
from keypoint_msgs.msg import KeypointsArray, keypoint, ObjectsArray
from visualization_msgs.msg import Marker, MarkerArray
import json
import albumentations as A

class ObjectKeypointPipeline:
    def __init__(self):
        image_topic = rospy.get_param("object_keypoints_ros/image_topic", "/camera/color/image_raw")
        self.camera_frame = rospy.get_param('object_keypoints_ros/camera_frame', "camera_color_optical_frame")
        self.img_sub = rospy.Subscriber(image_topic, Image, callback=self._mono_image_callback, queue_size=1)
        self.mono_image = None
        self.mono_image_ts = None
        self.bridge = CvBridge()
        self.use_gpu = rospy.get_param("object_keypoints_ros/gpu", True)
        self.info_period = 1.0
        
        # params
        self.input_size = (511, 511)  # TODO: hard-coded
        self.IMAGE_SIZE = (1280, 720)
        self.sq_IMAGE_SIZE = (720,720)
        self.prediction_size = SceneDataset.prediction_size # 64 x 64
        self.image_offset = SceneDataset.image_offset
        
        # config    
        self.keypoint_config_path = rospy.get_param('object_keypoints_ros/keypoints')
        with open(self.keypoint_config_path, 'rt') as f:
            self.keypoint_config = json.load(f)
        
        # model
        model = rospy.get_param('object_keypoints_ros/load_model', "/home/ken/Hack/catkin_ws/src/object_keypoints/model/modelv2.pt")
        _3d_point = []
        self.pipeline = pipeline.LearnedKeypointTrackingPipeline(model, self.use_gpu,
                self.prediction_size, _3d_point, self.keypoint_config)
        self.rgb_mean = torch.tensor([0.5, 0.5, 0.5], requires_grad=False, dtype=torch.float32)[:, None, None]
        self.rgb_std = torch.tensor([0.25, 0.25, 0.25], requires_grad=False, dtype=torch.float32)[:, None, None]
        
        # calibration
        self._read_calibration()
        self.pipeline.detection_to_point.reset(self.camera)
        self.scaling_factor = np.array(self.sq_IMAGE_SIZE) / np.array(self.prediction_size)
        rospy.logdebug("scalling facotr is : ")
        rospy.logdebug(self.scaling_factor)
        
        # Monocular Version
        self.pipeline.reset(self.camera_small)
        self._compute_bbox_dimensions()
        
        # kp ros msg
        self.ObjectArray = ObjectsArray()
        self.keypointArray = KeypointsArray()
        
        # Publishers
        self.result_img_pub = rospy.Publisher("object_keypoints_ros/result_img", Image, queue_size=1)
        self.kp_msg_pub = rospy.Publisher("object_keypoints_ros/keypoints_array", ObjectsArray, queue_size=1)
        
        # Only used if an object mesh is set.
        if self.bbox_size is not None:
            self.bbox_pub = rospy.Publisher("object_keypoints_ros/bbox", BoundingBox3D, queue_size=1)
        else:
            self.bbox_pub = None

        # to vis kp in overlay image
        self.green = np.zeros((8,8,3))
        self.green[:,:,1] = 254
        
        # debug: pub kp in 3D 
        self.kp_marker_array = MarkerArray()
        self.kp_marker_array_publisher = rospy.Publisher("kp_debug/kp_markers", MarkerArray, queue_size=10)
        
        self.kp_gt_marker_array = MarkerArray()
        self.kp_gt_marker_array_publisher = rospy.Publisher("kp_debug/kp_gt_markers", MarkerArray, queue_size=10)
        
        # Image processer, to make the input as the format to the model
        frame_resizer = [A.SmallestMaxSize(max_size=max(self.input_size[0], self.input_size[1])),
                A.CenterCrop(height=self.input_size[0], width=self.input_size[1])]
        self.frame_resizer = A.Compose(frame_resizer)  # first rescale and then crop, to keep the ratio
        
        # Opening JSON file to check GT
        f = open(rospy.get_param('object_keypoints_ros/ref_gt'))
        self.gt_data = json.load(f)
        self.gt_pt = self.gt_data['3d_points']
        
    def _read_calibration(self):
        path = rospy.get_param('object_keypoints_ros/calibration')
        params = camera_utils.load_calibration_params(path)
        self.K = params['K']
        self.Kp = params['Kp']
        self.D = params['D']
        self.Dp = params['Dp']
        self.T_RL = params['T_RL']
        self.image_size = params['image_size']

        # Cooperate with monocular version
        camera = camera_utils.FisheyeCamera(params['K'], params['D'], params['image_size'])
        camera = camera.scale(SceneDataset.height_resized / SceneDataset.height)
        self.camera = camera.cut(self.image_offset)  # cam model for resolution 511 * 511
        
        scale_small = self.prediction_size[0] / SceneDataset.height_resized  # cam model for resolution 64 * 64
        self.camera_small = camera.cut(self.image_offset).scale(scale_small)


    def _read_keypoints(self):  # not included in prediction process
        path = rospy.get_param('object_keypoints_ros/keypoints')
        with open(path, 'rt') as f:
            return np.array(json.loads(f.read())['3d_points'])


    def _compute_bbox_dimensions(self):
        mesh_file = rospy.get_param('object_keypoints_ros/object_mesh', None)
        if mesh_file is not None:
            bounding_box = utils.compute_bounding_box(mesh_file)
            # Size is in both directions, surrounding the object from the object center.
            self.bbox_size = (bounding_box.max(axis=0) - bounding_box.min(axis=0)) * 0.5
        else:
            self.bbox_size = None

    def _mono_image_callback(self, image):
        img = self.bridge.imgmsg_to_cv2(image, 'rgb8')
        self.mono_image = img
        self.mono_image_ts = image.header.stamp


    def _preprocess_image(self, image):
        image = self.frame_resizer(image=image)['image'].astype(np.float32)
        image = image.transpose([2, 0, 1])
        image = torch.tensor(image / 255.0, dtype=torch.float32)
        image -= self.rgb_mean
        image /= self.rgb_std
        image = image[None]
        return image.clone().detach()
    
    def _to_msg(self, objs):
        
        self.ObjectArray = ObjectsArray()
        self.ObjectArray.header.stamp = self.mono_image_ts
        self.ObjectArray.header.frame_id = self.camera_frame
        self.ObjectArray.ObjectSize = len(objs)  # TODO: for single object now
        self.ObjectArray.PointSize = 0 
        
        for i, obj in enumerate(objs):
            
            self.keypointArray = KeypointsArray()
            self.keypointArray.header.stamp = self.mono_image_ts
            self.keypointArray.header.frame_id = self.camera_frame
            self.keypointArray.PointSize = obj['world_xyz'].shape[0]
            self.ObjectArray.PointSize += self.keypointArray.PointSize
            
            # for j, (p_world, p_center) in enumerate(zip(obj['p_C'],obj['keypoints'])):
            for j in range(self.keypointArray.PointSize):
                
                p_world = obj['world_xyz'][j]
                p_center =  obj['coord_xyz'][0].numpy()[j]
                keypoint_msg = keypoint()
                keypoint_msg.object_idx = i
                keypoint_msg.semantic_idx = j
                
                if p_world is None:
                    keypoint_msg.valid = False   # label the not-detected point
                    self.keypointArray.keypoints.append(keypoint_msg)
                    continue
                
                # insert 3D keypoint in world
                keypoint_msg.valid = True 
                keypoint_msg.point3d.x =  p_world[0]
                keypoint_msg.point3d.y =  p_world[1]
                keypoint_msg.point3d.z =  p_world[2]

                # insert 2D keypoint in result img
                p_img = self._heatmapIdx_to_imageIdx(p_center[:2])
                keypoint_msg.point2d.x = p_img[0]
                keypoint_msg.point2d.y = p_img[1]
                
                self.keypointArray.keypoints.append(keypoint_msg)
                
                # pin the point on self.image_overlay
                self.image_overlay[p_img[0]-4:p_img[0]+4, p_img[1]-4:p_img[1]+4,:] = self.green
            
                rospy.logdebug_throttle(self.info_period,"original 2D kp from model: ")
                rospy.logdebug_throttle(self.info_period,p_center)  # indx in 64,64 image
                rospy.logdebug_throttle(self.info_period,"2D kp after scalling in result img: ")
                rospy.logdebug_throttle(self.info_period,p_img)  # indx in output image
        
        
        self.ObjectArray.KeypointsArrays.append(self.keypointArray)
        
    def _publish_keypoints(self, keypoints, time):
        for i in range(min(keypoints.shape[0], 4)):
            msg = self._to_msg(keypoints[i], rospy.Time(0), self.camera_frame)
            getattr(self, f'point{i}_pub').publish(msg)

    def _publish_pose(self, pose_msg, time):
        pose_msg = ros_utils.transform_to_pose(T, self.camera_frame, rospy.Time(0))
        self.pose_pub.publish(pose_msg)
        self._publish_bounding_box(pose_msg)

    def _publish_bounding_box(self, T, pose_msg):
        if self.bbox_size is not None:
            msg = BoundingBox3D()
            msg.pose = pose_msg.pose
            msg.size.x = self.bbox_size[0]
            msg.size.y = self.bbox_size[1]
            msg.size.z = self.bbox_size[2]
            self.bbox_pub.publish(msg)

    def _publish_result_img(self):
        image_msg = self.bridge.cv2_to_imgmsg(self.image_overlay[:, :, :3], encoding='passthrough')
        image_msg.header.stamp = self.mono_image_ts
        image_msg.height = 720
        image_msg.width = 720
        self.result_img_pub.publish(image_msg)
        

    def _to_heatmap(self, target):
         # target = np.clip(target, 0.0, 1.0)  # NO need, sigmoid is done in package model 
        target = target.sum(axis=0)
        target = (cm.inferno(target) * 255.0).astype(np.uint8)[:, :, :3]
        # return cv2.resize(target[:, :, :3], self.IMAGE_SIZE)
        return cv2.resize(target[:,:,:3], self.sq_IMAGE_SIZE)
    
    def _to_image(self, frame):
        frame = SceneDataset.to_image(frame)
        # return cv2.resize(frame, self.IMAGE_SIZE)
        return cv2.resize(frame, self.sq_IMAGE_SIZE)
        
    def kp_map_to_image(self, kp):  # TODO: check if this still needed
        kp_ = []
        for i, pt in enumerate(kp):
            if pt.size == 0:
                kp_.append(np.array([]))
                continue
            pt = np.squeeze(pt)     
            pt = np.multiply( pt, self.scaling_factor).astype(np.int64)
            kp_.append( np.flip(pt) )   # TODO: notice kp idx is flipped from image idx in 2D, this is acturally indx in [height, width] frame
        return kp_
    
    def _heatmapIdx_to_imageIdx(self, pt):
        pt = np.squeeze(pt)  
        pt = np.multiply( pt, self.scaling_factor).astype(np.int64)
        pt_ = np.flip(pt)  # TODO: notice kp idx is flipped from image idx in 2D, this is acturally indx in [height, width] frame
        return pt_
               
    def step(self):
        
        # process rgb image
        if self.mono_image is not None:
            mono_image = self._preprocess_image(self.mono_image)
            
            objects = []
            object = self.pipeline(mono_image)
            objects.append(object)  # TODO: now is the single object pipeline len(objects = 1)
            
            self.mono_image = None
 
            heatmap = self._to_heatmap(object['heatmap'].numpy())
            mono_image = mono_image.cpu().numpy()[0]

            #rospy.loginfo(np.shape(mono_image))        # [3, 511, 511]
            
            mono_rgb = self._to_image(mono_image)
            self.image_overlay = (0.3 * mono_rgb + 0.7 * heatmap).astype(np.uint8)
            
            # extract obj 3D and 2D idx in result img, and convert to ros msg
            self._to_msg(objects)  
            
            # publish
            self.kp_msg_pub.publish(self.ObjectArray)
                        
            # pub result image, i.e. self.image_overlay
            self._publish_result_img()
        
            # kp_marker_array = MarkerArray()   
            # for i in range(object['world_xyz'].shape[0]):
            #     x = object['world_xyz'][i,0]
            #     y = object['world_xyz'][i,1]
            #     z = object['world_xyz'][i,2]
 
            #     self.kp_marker = Marker()
            #     self.kp_marker.header.frame_id = "camera_color_optical_frame"
            #     self.kp_marker.type = Marker.SPHERE
            #     self.kp_marker.action = Marker.ADD
            #     self.kp_marker.scale.x = 0.01
            #     self.kp_marker.scale.y = 0.01
            #     self.kp_marker.scale.z = 0.01
            #     self.kp_marker.color.a = 1.0   # TODO: for multi-object, set different color based on color idx from msg
            #     self.kp_marker.color.r = 1.0
            #     self.kp_marker.color.g = 0.0
            #     self.kp_marker.color.b = 0.0
            #     self.kp_marker.pose.position.x = x
            #     self.kp_marker.pose.position.y = y
            #     self.kp_marker.pose.position.z = z
            #     self.kp_marker.pose.orientation.w = 1.0 
            #     self.kp_marker.id = i
            #     kp_marker_array.markers.append(self.kp_marker)
                        
            # self.kp_marker_array = kp_marker_array
            # self.kp_marker_array_publisher.publish(self.kp_marker_array)
            
            
            kp_gt_marker_array = MarkerArray()   
            for i in range(len(self.gt_pt)):
                x = self.gt_pt[i][0]
                y = self.gt_pt[i][1]
                z = self.gt_pt[i][2]
 
                self.kp_marker = Marker()
                self.kp_marker.header.frame_id = "world"
                self.kp_marker.type = Marker.SPHERE
                self.kp_marker.action = Marker.ADD
                self.kp_marker.scale.x = 0.01
                self.kp_marker.scale.y = 0.01
                self.kp_marker.scale.z = 0.01
                self.kp_marker.color.a = 1.0 
                self.kp_marker.color.r = 0.0
                self.kp_marker.color.g = 1.0
                self.kp_marker.color.b = 0.0
                self.kp_marker.pose.position.x = x
                self.kp_marker.pose.position.y = y
                self.kp_marker.pose.position.z = z
                self.kp_marker.pose.orientation.w = 1.0 
                self.kp_marker.id = i
                kp_gt_marker_array.markers.append(self.kp_marker)
                        
            self.kp_gt_marker_array = kp_gt_marker_array
            self.kp_gt_marker_array_publisher.publish(self.kp_gt_marker_array)
            
            
if __name__ == "__main__":
    log_debug = rospy.get_param('object_keypoints_ros/log_debug', False)
    with torch.no_grad():
        if log_debug:
            rospy.init_node("object_keypoints_ros",log_level=rospy.DEBUG)
        else:
            rospy.init_node("object_keypoints_ros",log_level=rospy.INFO)    
        keypoint_pipeline = ObjectKeypointPipeline()
        rate = rospy.Rate(10)
        with torch.no_grad():
            while not rospy.is_shutdown():
                keypoint_pipeline.step()
                rate.sleep()

