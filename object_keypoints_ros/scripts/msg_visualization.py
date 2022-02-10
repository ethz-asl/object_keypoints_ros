#!/usr/bin/env python3

import enum
import sys
from numpy.core.fromnumeric import size
import json
from cv_bridge import CvBridge
from perception.utils import ros
from yaml.error import Mark
import rospy
import numpy as np
import cv2
from geometry_msgs.msg import PointStamped, PoseStamped, PoseArray, Pose
from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from keypoint_msgs.msg import KeypointsArray, keypoint

class Keypoints3dVisualizer:

    def __init__(self):
        rospy.loginfo("self.kp_markers node init") 
        
      # cv2
        self.bridge = CvBridge()
      # ros 
      
        # params
        self.depth_image_topic = rospy.get_param("object_keypoints_ros/depth_image_topic", "/camera/aligned_depth_to_color/image_raw")
        
        # msgs
        self.kp_marker_array = MarkerArray()
        
        # pubs
        self.depth_publisher = rospy.Publisher("object_keypoints_ros/depth_overlay", Image, queue_size=1)
        self.kp_marker_array_publisher = rospy.Publisher("kp_msg_vis_node/kp_markers", MarkerArray, queue_size=10)
        
        # subs
        self.kp_msg_subscriber = rospy.Subscriber("object_keypoints_ros/keypoints_array", KeypointsArray, self.kp_msgs_callback, queue_size=1)
        self.depth_subscriber = rospy.Subscriber(self.depth_image_topic, Image, callback=self._depth_img_callback, queue_size=1)
    
      # local node vars
        self.points_2d = []
        self.depth_img = None
        self.depth_img_overlay = None
        
    def update_kp_msg(self, msg):
        
        kp_marker_array = MarkerArray()
        point_2d = []
        
        for i, pt in enumerate(msg.keypoints):
            if pt.valid == False:
                continue
            
            # kp markers
            self.kp_marker = Marker()
            self.kp_marker.header.stamp = msg.header.stamp
            self.kp_marker.header.frame_id = msg.header.frame_id
            self.kp_marker.type = Marker.SPHERE
            self.kp_marker.action = Marker.ADD
            self.kp_marker.scale.x = 0.01
            self.kp_marker.scale.y = 0.01
            self.kp_marker.scale.z = 0.01
            self.kp_marker.color.a = 1.0   # TODO: for multi-object, set different color based on color idx from msg
            self.kp_marker.color.r = 1.0
            self.kp_marker.color.g = 0.0
            self.kp_marker.color.b = 0.0
            self.kp_marker.pose.position.x = pt.point3d.x
            self.kp_marker.pose.position.y = pt.point3d.y
            self.kp_marker.pose.position.z = pt.point3d.z
            self.kp_marker.pose.orientation.w = 1.0 
            self.kp_marker.id = i
            kp_marker_array.markers.append(self.kp_marker)
                    
            # kp idx in 2d img  
            point_2d.append(pt.point2d)
            
        self.kp_marker_array = kp_marker_array
        self.points_2d = point_2d
        
    def kp_msgs_callback(self, msg):
        self.update_kp_msg(msg)
        
    def _get_depth_cam_intrinsic(self):
        # depth camera intrinscis
        fx = 915.5664672851562
        fy = 914.7263793945312
        cx = 656.3117065429688
        cy = 354.6520080566406
        return np.array([[fx, 0., cx],
                        [0., fy, cy],
                        [0., 0., 1.]])
        
        
    def convert_depth_pixel_to_metric_coordinate(self, depth_pixel, camera_intrinsics):
        """
        Convert the depth and image point information to metric coordinates
        Parameters:
        -----------
        depth_pixels     : list of np.arrray()
                            The list of pixels from depth image

        camera_intrinsics : Intrinsics matrix of camera
        Return:
        ----------
        X : double
            The x value in meters
        Y : double
            The y value in meters
        Z : double
            The z value in meters
        """
        depth = self.depth_img[depth_pixel[0],depth_pixel[1]] / 1000
        X = (depth_pixel[0] - camera_intrinsics[0,2])/camera_intrinsics[0,0] *depth
        Y = (depth_pixel[1] - camera_intrinsics[1,2])/camera_intrinsics[1,1] *depth
        # print("depth: ", X, Y, depth)
        
        return X, Y, depth
    
    
    def _depth_img_callback(self,msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg)
    
    def _process_depth_img(self):
        if self.depth_img is not None:
            self.depth_img_overlay = np.copy(self.depth_img)
            for pt in self.points_2d:
                self.depth_img_overlay[int(pt.x)-10:int(pt.x)+10, int(pt.y)-10:int(pt.y)+10] += 50000
    
    def _publish_processed_depth(self):
        if self.depth_img_overlay is not None:
            img = self.bridge.cv2_to_imgmsg(self.depth_img_overlay*10,encoding="mono16")
            self.depth_publisher.publish(img)
            
    def call_publishers(self):
        # 3d keypoints from model
        self.kp_marker_array_publisher.publish(self.kp_marker_array)
        # overlay depth img
        self._publish_processed_depth()
        
    def update(self):
        self._process_depth_img()
        self.call_publishers()

    def compute_bbox_on_depth(self):
        #     # overlay the kp to the image
        #     for i, pt in enumerate(self.points_in_2d):
        #         depth_pixels = []
                
        #         # print("kp: ", pt)
                    
        #         if pt.size != 0:
        #             image_overlay[self.points_in_2d[i][0]-10:self.points_in_2d[i][0]+10, self.points_in_2d[i][1]-10:self.points_in_2d[i][1]+10,:] += self.green.astype(np.uint8)
            
        #         # process depth image  
        #         # if self.depth_img is not None:

        #         #     depth_pixel = self.convert_depth_pixel_to_metric_coordinate(pt,self._get_depth_cam_intrinsic())
        #         #     depth_pixels.append(depth_pixel)
        #         #     print("depth strenght: ",depth_img_overlay[self.points_in_2d[i][0], self.points_in_2d[i][1]])
        #         #     depth_img_overlay[self.points_in_2d[i][0]-10:self.points_in_2d[i][0]+10, self.points_in_2d[i][1]-10:self.points_in_2d[i][1]+10] += 50000
        #         #     print("depth 3D poses [", i, "] is: ", depth_pixels)
        return
    
if __name__ == "__main__":
    rospy.init_node("kp_msg_vis_node")
    kp_vis = Keypoints3dVisualizer()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        kp_vis.update()
        rate.sleep()
   