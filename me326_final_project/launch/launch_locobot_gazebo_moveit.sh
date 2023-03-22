#!/bin/bash

gnome-terminal -x roslaunch interbotix_xslocobot_moveit xslocobot_moveit.launch robot_model:=locobot_wx250s show_lidar:=true use_gazebo:=true dof:=6
#gnome-terminal -x roslaunch interbotix_xslocobot_control xslocobot_control.launch robot_model:=locobot_wx250s show_lidar:=true use_sim:=true
sleep 10
rosservice call /gazebo/unpause_physics

sleep 5
roslaunch me326_locobot_example spawn_cube.launch
gnome-terminal -x roslaunch me326_final_project gazebo_moveit_arm.launch

sleep 5
gnome-terminal -x rosrun me326_final_project matching_ptcld_serv

sleep 5
gnome-terminal -x roslaunch me326_final_project locobot_gazebo_example.launch 


