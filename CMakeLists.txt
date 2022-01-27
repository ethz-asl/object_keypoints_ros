cmake_minimum_required(VERSION 3.10)
project(object_keypoints_ros)
find_package(catkin REQUIRED COMPONENTS message_generation geometry_msgs sensor_msgs std_msgs)

catkin_python_setup()

add_message_files(DIRECTORY msg FILES Keypoint.msg Keypoints.msg)
add_service_files(DIRECTORY srv FILES KeypointsDetection.srv KeypointsPerception.srv)

generate_messages(DEPENDENCIES std_msgs geometry_msgs sensor_msgs)

catkin_package(CATKIN_DEPENDS message_runtime std_msgs geometry_msgs sensor_msgs)



