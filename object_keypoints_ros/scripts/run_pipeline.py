#!/usr/bin/env python3

# To find catkin python3 build of tf2_py
from cgitb import grey
from pickle import NONE
import sys
from matplotlib.pyplot import axis

from numpy.core.fromnumeric import size
import json

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
import utils
from keypoint_msgs.msg import KeypointsArray, keypoint, ObjectsArray
from visualization_msgs.msg import Marker, MarkerArray
import json
import albumentations as A
import time

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
        
        # kp ros msg
        self.ObjectArray = ObjectsArray()
        self.keypointArray = KeypointsArray()
        
        # Publishers
        self.result_img_pub = rospy.Publisher("object_keypoints_ros/result_img", Image, queue_size=1)
        self.kp_msg_pub = rospy.Publisher("object_keypoints_ros/keypoints_array", ObjectsArray, queue_size=1)

        # to vis kp in overlay image
        self.greenpatch = np.zeros((8,8,3))
        self.greenpatch[:,:,1] = 254
        
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
        self._read_keypoints()

        
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
        f = open(rospy.get_param('object_keypoints_ros/ref_gt'))
        self.gt_data = json.load(f)
        self.gt_pt = self.gt_data['3d_points']
        return 

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
    
    def _to_msg(self):
        
        # conver the object dictionary into customed ros msg "KeypointsArray"
        
        self.ObjectArray = ObjectsArray()
        self.ObjectArray.header.stamp = self.mono_image_ts
        self.ObjectArray.header.frame_id = self.camera_frame
        self.ObjectArray.ObjectSize = len(self.objects)  # TODO: for single object now
        self.ObjectArray.PointSize = 0 
        
        for i, obj in enumerate(self.objects):
            
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
                try:
                    self.image_overlay[p_img[0]-4:p_img[0]+4, p_img[1]-4:p_img[1]+4,:] = self.greenpatch
                except:
                    print("Cannot overlay point on img")
                    
                rospy.logdebug_throttle(self.info_period,"original 2D kp from model: ")
                rospy.logdebug_throttle(self.info_period,p_center)  # indx in 64,64 image
                rospy.logdebug_throttle(self.info_period,"2D kp after scalling in result img: ")
                rospy.logdebug_throttle(self.info_period,p_img)  # indx in output image

        
        self.ObjectArray.KeypointsArrays.append(self.keypointArray)
        
            

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
    
    def _heatmapIdx_to_imageIdx(self, pt):
        pt = np.squeeze(pt)  
        pt = np.multiply( pt, self.scaling_factor).astype(np.int64)
        pt_ = np.flip(pt)  # TODO: notice kp idx is flipped from image idx in 2D, this is acturally indx in [height, width] frame
        return pt_
               
    def publish_keypointMarker_gt(self):
        
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
    
    def publish_keypointMarger_pred(self,pt_world):
                    
        kp_marker_array = MarkerArray()   
        for i in range(pt_world.shape[0]):
            x = pt_world[i,0]
            y = pt_world[i,1]
            z = pt_world[i,2]

            self.kp_marker = Marker()
            self.kp_marker.header.frame_id = "camera_color_optical_frame"
            self.kp_marker.type = Marker.SPHERE
            self.kp_marker.action = Marker.ADD
            self.kp_marker.scale.x = 0.01
            self.kp_marker.scale.y = 0.01
            self.kp_marker.scale.z = 0.01
            self.kp_marker.color.a = 1.0   # TODO: for multi-object, set different color based on color idx from msg
            self.kp_marker.color.r = 1.0
            self.kp_marker.color.g = 0.0
            self.kp_marker.color.b = 0.0
            self.kp_marker.pose.position.x = x
            self.kp_marker.pose.position.y = y
            self.kp_marker.pose.position.z = z
            self.kp_marker.pose.orientation.w = 1.0 
            self.kp_marker.id = i
            kp_marker_array.markers.append(self.kp_marker)
                    
        self.kp_marker_array = kp_marker_array
        self.kp_marker_array_publisher.publish(self.kp_marker_array)
        return
    
    def step(self):
    
        tic_1 = time.perf_counter()
        
        if self.mono_image is not None:
            
            # process rgb image, make it as the format to the model 
            mono_image = self._preprocess_image(self.mono_image)
            self.mono_image = None
            self.objects = []
            
            # model inference
            tic = time.perf_counter()
            object = self.pipeline(mono_image)
            toc = time.perf_counter()
            # print(f"model inference takes: {toc-tic:0.4f} sec" )
            self.objects.append(object)  # TODO: now is the single object pipeline len(objects = 1)
                # extract obj 3D and 2D idx in result img, and convert objects dict to customed ros msg
            self._to_msg()  
                # publish
            self.kp_msg_pub.publish(self.ObjectArray)
                          
            # generate resulting image
            mono_image = mono_image.cpu().numpy()[0]   # [3, 511, 511  
            self.image_overlay = (0.3 * self._to_image(mono_image) + 0.7 * self._to_heatmap(object['heatmap'].numpy())).astype(np.uint8)
                # pub resulting image, i.e. self.image_overlay
            self._publish_result_img()
                        
                        
            # pub predicted keypoints marker for rviz
            self.publish_keypointMarger_pred(object['world_xyz'])
            
            # pub ground-truth keypoints marker(if there is) for rviz
            # self.publish_keypointMarker_gt()
                
        toc_1 = time.perf_counter()
        # print(f"whole kp process takes: {toc_1-tic_1:0.4f} sec" )
        
        
            
if __name__ == "__main__":
    log_debug = rospy.get_param('object_keypoints_ros/log_debug', False)
    with torch.no_grad():
        if log_debug:
            rospy.init_node("object_keypoints_ros",log_level=rospy.DEBUG)
        else:
            rospy.init_node("object_keypoints_ros",log_level=rospy.INFO)    
        keypoint_pipeline = ObjectKeypointPipeline()
        rate = rospy.Rate(30)
        with torch.no_grad():
            while not rospy.is_shutdown():
                keypoint_pipeline.step()
                rate.sleep()

