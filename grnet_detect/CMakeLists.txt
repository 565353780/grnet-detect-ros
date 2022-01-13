cmake_minimum_required(VERSION 3.0.2)
project(grnet_detect)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  sensor_msgs
  genmsg
)

add_service_files(
  FILES
  PC2ToPC2.srv
)

catkin_python_setup()

generate_messages(
  DEPENDENCIES
  sensor_msgs
)

catkin_package()

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

catkin_install_python(PROGRAMS
  bin/run_inference
  bin/run_train
  bin/run_test
  bin/listener
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

