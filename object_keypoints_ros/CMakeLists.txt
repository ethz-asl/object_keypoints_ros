cmake_minimum_required(VERSION 3.10)
project(object_keypoints_ros)
find_package(catkin REQUIRED)
catkin_python_setup()

set(PACKAGE_DEPENDENCIES
    keypoint_msgs
    )
