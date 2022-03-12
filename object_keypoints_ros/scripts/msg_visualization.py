#!/usr/bin/env python3

import enum
import imp
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
from std_msgs.msg import Header
from sensor_msgs.msg import Image , PointCloud2, PointField
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from keypoint_msgs.msg import KeypointsArray, keypoint, ObjectsArray

class Keypoints3dVisualizer:

    def __init__(self):
        rospy.loginfo("self.kp_markers node init") 
        
      # cv2
        self.bridge = CvBridge()
      # ros 
      
        # params
        self.depth_image_topic = rospy.get_param("object_keypoints_ros/depth_image_topic", "/camera/aligned_depth_to_color/image_raw")
        self.msg_header = Header()
        
        # msgs
        self.kp_marker_array = MarkerArray()
        
        # pubs
        self.depth_publisher = rospy.Publisher("object_keypoints_ros/depth_overlay", Image, queue_size=1)
        self.kp_marker_array_publisher = rospy.Publisher("kp_msg_vis_node/kp_markers", MarkerArray, queue_size=10)
        self.pointcloud_roi_publisher = rospy.Publisher("kp_msg_vis_node/pcl_roi", PointCloud2, queue_size=1)
        
        # subs
        self.kp_msg_subscriber = rospy.Subscriber("object_keypoints_ros/keypoints_array", ObjectsArray, self.kp_msgs_callback, queue_size=1)
        self.depth_subscriber = rospy.Subscriber(self.depth_image_topic, Image, callback=self._depth_img_callback, queue_size=1)
    
      # local node vars
        self.points_2d = []
        self.depth_img = None
        self.imgtodepth_offset = [0, ((1280 - 720.0) / 2.0)]  # the input 2D is the coord in [720 720] img
        self.depth_img_overlay = None
        self.depth_roi = None
        self.top_left_idx = None
        self.bottom_right_idx = None
        
        self.pointcloud_roi = None
        
    def update_kp_msg(self, msg):
        
        kp_marker_array = MarkerArray()
        point_2d = []
        
        for j, pt_arr in enumerate(msg.KeypointsArrays):
            
            for i, pt in enumerate(pt_arr.keypoints):
                if pt.valid == False:
                    continue
                
                # kp markers
                self.kp_marker = Marker()
                self.kp_marker.header = self.msg_header
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
                pt.point2d.x = pt.point2d.x  + self.imgtodepth_offset[0]
                pt.point2d.y = pt.point2d.y  + self.imgtodepth_offset[1]
                point_2d.append(pt.point2d)
            
        self.kp_marker_array = kp_marker_array
        self.points_2d = point_2d
        
    def kp_msgs_callback(self, msg):
        self.msg_header = msg.header
        self.update_kp_msg(msg)
        
    def _get_depth_cam_intrinsic(self):
        # depth camera intrinscis from topic "/camera/aligned_depth_to_color/camera_info"
        fx = 915.5664672851562
        fy = 914.7263793945312
        cx = 656.3117065429688
        cy = 354.6520080566406
        return np.array([[fx, 0., cx],
                        [0., fy, cy],
                        [0., 0., 1.]])
        
        
    def convert_depth_pixel_to_metric_coordinate(self, depth_pixel):
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
        camera_intrinsics = self._get_depth_cam_intrinsic()
        
        depth = self.depth_img[depth_pixel[0],depth_pixel[1]] / 1000
        X = (depth_pixel[1] - camera_intrinsics[0,2])/camera_intrinsics[0,0] *depth
        Y = (depth_pixel[0] - camera_intrinsics[1,2])/camera_intrinsics[1,1] *depth
        # print("depth: ", X, Y, depth)
        
        return np.array([X, Y, depth])
    
    def convert_depth_frame_to_pointcloud(self):
        """
        Convert the depthmap to a 3D point cloud
        Parameters:
        -----------
        depth_frame 	 	 :  np 2d array
                            The depth_frame containing the depth map
        camera_intrinsics : The intrinsic values of the imager in whose coordinate system the depth_frame is computed
        Return:
        ----------
        x : array
            The x values of the pointcloud in meters
        y : array
            The y values of the pointcloud in meters
        z : array
            The z values of the pointcloud in meters
        """
        camera_intrinsics = self._get_depth_cam_intrinsic()
        [height, width] = self.depth_roi.shape

        nx = np.linspace(0, width-1, width)
        ny = np.linspace(0, height-1, height)
        u, v = np.meshgrid(nx, ny)
        
        # need to add the bbox drift in 2D image, to get the correct pixel idx in original depth img
        u = u + self.top_left_idx[1]
        v = v + self.top_left_idx[0]

        x = (u.flatten() - camera_intrinsics[0,2])/camera_intrinsics[0,0]
        y = (v.flatten() - camera_intrinsics[1,2])/camera_intrinsics[1,1]
        z = self.depth_roi.flatten() / 1000
        x = np.multiply(x,z)
        y = np.multiply(y,z)

        x = x[np.nonzero(z)]
        y = y[np.nonzero(z)]
        z = z[np.nonzero(z)]
    
        return np.vstack((x, y, z)).transpose()

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
        # point cloud roi markerArray
        if self.pointcloud_roi is not None:
            self.pointcloud_roi_publisher.publish(self.np_to_point_cloud())
        
        
    def compute_bbox_on_depth(self):
        # heuristic method: take the center keypoint and hard-code the width and length
        
        if len(self.points_2d) == 0:
            rospy.logwarn_throttle(1,"No 2D keypoints input")
            return False
        
        bbox_center = self.points_2d[0]
        
        width = 140
        height = 150
        lw = 3
        top_left     = np.array([bbox_center.x - width, bbox_center.y-height]).astype(np.int64)
        top_right    = np.array([bbox_center.x + width, bbox_center.y-height]).astype(np.int64)
        bottom_left  = np.array([bbox_center.x - width, bbox_center.y+height]).astype(np.int64)
        bottom_right = np.array([bbox_center.x + width, bbox_center.y+height]).astype(np.int64)
        
        # project the bbox on depth overlay
        self.depth_img_overlay[top_left[0]:top_right[0], top_left[1]-lw:top_left[1]+lw] += 50000
        self.depth_img_overlay[bottom_left[0]:bottom_right[0], bottom_left[1]-lw:bottom_left[1]+lw] += 50000
        self.depth_img_overlay[top_left[0]-lw:top_left[0]+lw, top_left[1]:bottom_left[1]] += 50000
        self.depth_img_overlay[top_right[0]-lw:top_right[0]+lw, top_right[1]:bottom_right[1]] += 50000
        self.depth_img_overlay[ top_left[0]:bottom_right[0] , top_left[1]:bottom_right[1] ] += 10000
        
        # get the bbox pixel idxs and crop the depth roi
        self.top_left_idx = top_left
        self.bottom_right_idx = bottom_right
        self.depth_roi = self.depth_img[self.top_left_idx[0]:self.bottom_right_idx[0], self.top_left_idx[1]:self.bottom_right_idx[1]]
        
        return True
    
    def get_pointcloud_roi(self):
        
        if self.top_left_idx is None:
            rospy.logwarn_throttle(1, "No bbox idxs")
            return False
        
        if self.depth_img is None:
            rospy.logwarn_throttle(1, "No depth img")
            return False

        if self.depth_roi is None:
            rospy.logwarn_throttle(1,"No cropped depth roi")
            return False
        
        # extract from 2D bbox on depth img
        self.pointcloud_roi = self.convert_depth_frame_to_pointcloud() 

        return True

        
        
    def np_to_point_cloud(self):
        """ Creates a point cloud message.
        Args:
            points: N x 7 array of xyz positions (m) and rgba colors (0..1)
            parent_frame: frame in which the point cloud is defined
        Returns:
            sensor_msgs/PointCloud2 message
        """
        ros_dtype = PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = self.pointcloud_roi.astype(dtype).tobytes()

        fields = [PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyz')]

        return PointCloud2(
            header=self.msg_header,
            height=1,
            width=self.pointcloud_roi.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 3),
            row_step=(itemsize * 3 * self.pointcloud_roi.shape[0]),
            data=data
        )

    def update(self):
        
        self._process_depth_img()
        
        if not self.compute_bbox_on_depth():
            rospy.logwarn_throttle(1, "Can not compute bbox on depth img")

        if not self.get_pointcloud_roi():
            rospy.logwarn_throttle(1, "Can not get point cloud")

        self.call_publishers()
        
        
if __name__ == "__main__":
    rospy.init_node("kp_msg_vis_node")
    kp_vis = Keypoints3dVisualizer()
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        kp_vis.update()
        rate.sleep()
   