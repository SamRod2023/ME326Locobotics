#!/usr/bin/env python3
'''
Written by: Monroe Kennedy, Date: 1/2/2023
Docs: http://docs.ros.org/en/melodic/api/moveit_tutorials/html/doc/move_group_python_interface/move_group_python_interface_tutorial.html

Example of using moveit for grasping example
'''

import sys
import rospy
import numpy as np
import scipy as sp
from scipy import linalg
import geometry_msgs
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from visualization_msgs.msg import Marker
from std_msgs.msg import String

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from moveit_commander.conversions import pose_to_list

import sensor_msgs.point_cloud2 as pc2
from sensor_msgs.msg import Image
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

#from interbotix_xs_modules.locobot import InterbotixLocobotXS

import cv2
import time

from std_msgs.msg import Float64


class OrientCamera(object):
	"""docstring for OrientCamera"""
	def __init__(self, tilt_topic = "/locobot/tilt_controller/command", pan_topic = "/locobot/pan_controller/command"):		
		self.orient_pub = rospy.Publisher(tilt_topic, Float64, queue_size=1, latch=True)
		self.pan_pub = rospy.Publisher(pan_topic, Float64, queue_size=1, latch=True)

	def tilt_camera(self,angle=0.5):
		msg = Float64()
		msg.data = angle
		self.orient_pub.publish(msg)
		# print("cause orientation, msg: ", msg)

	def pan_camera(self,angle=0.5):
		msg = Float64()
		msg.data = angle
		self.pan_pub.publish(msg)

class MoveLocobotArm(object):
	"""docstring for MoveLocobotArm"""
	def __init__(self,moveit_commander=None):
		self.moveit_commander = moveit_commander
		self.robot = self.moveit_commander.RobotCommander() #this needs to be launched in the namespace of the robot (in this example, this is done in the launch file using 'group')
		self.scene = self.moveit_commander.PlanningSceneInterface()
		self.gripper_group_name = "interbotix_gripper"
		self.gripper_move_group = self.moveit_commander.MoveGroupCommander(self.gripper_group_name)

		self.arm_group_name = "interbotix_arm" #interbotix_arm and interbotix_gripper (can see in Rviz)
		self.arm_move_group = self.moveit_commander.MoveGroupCommander(self.arm_group_name)
		self.display_trajectory_publisher = rospy.Publisher('/locobot/move_group/display_planned_path',
		                                               moveit_msgs.msg.DisplayTrajectory,
		                                               queue_size=20)
		# We can get the name of the reference frame for this robot:
		self.planning_frame = self.arm_move_group.get_planning_frame()
		# We can also print the name of the end-effector link for this group:
		self.eef_link = self.arm_move_group.get_end_effector_link()
		self.jnt_names = self.arm_move_group.get_active_joints()
		# We can get a list of all the groups in the robot:
		self.group_names = self.robot.get_group_names()
		self.state = "planning grasp"

		self.grasp_complete = rospy.Publisher("/locobot/mobile_base/grasp_complete", String, queue_size=1)

		self.in_range = False



	def display_moveit_info(self):
		# We can get the name of the reference frame for this robot:
		print("============ Planning frame: %s" % self.planning_frame)
		# We can also print the name of the end-effector link for this group:
		print("============ End effector link: %s" % self.eef_link)
		print("============ Armgroup joint names: %s" % self.jnt_names)
		# We can get a list of all the groups in the robot:
		print("============ Available Planning Groups:", self.robot.get_group_names())
		# Sometimes for debugging it is useful to print the entire state of the
		# robot:
		print("============ Printing robot state")
		print(self.robot.get_current_state())
		print("\n")
		
	def close_gripper(self):
		gripper_goal = self.gripper_move_group.get_current_joint_values()
		print("grippers",gripper_goal)
		gripper_goal[0] = 0.016 #0.015 - 0.037
		gripper_goal[1] = -0.016 #-0.015 - 0.037
		print("grippers",gripper_goal)
		self.gripper_move_group.go(gripper_goal, wait=True)
		gripper_goal = self.gripper_move_group.get_current_joint_values()
		print("grippers",gripper_goal)

	def open_gripper(self):
		gripper_goal = self.gripper_move_group.get_current_joint_values()
		print('opening')
		gripper_goal[0] = 0.03
		gripper_goal[1] = -0.03
		#gripper_goal = self.gripper_move_group.get_named_target_values('Open')
		self.gripper_move_group.go(gripper_goal, wait=True)
		self.gripper_move_group.stop()
		gripper_goal = self.gripper_move_group.get_current_joint_values()
		print("grippers",gripper_goal)

	def move_arm_down_for_camera(self):
		#start here
		joint_goal = self.arm_move_group.get_current_joint_values() 
		#['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
		joint_goal[0] = -0.1115207331248822 #waist
		joint_goal[1] = -0.5313552376357276 #shoulder
		joint_goal[2] = 1.058371284458718 #elbow
		joint_goal[3] = -0.05608022936825474 #forearm_roll
		joint_goal[4] = 0.9302728070281328 #wrist_angle
		joint_goal[5] = -0.14247350829385486 #wrist_rotate

		# The go command can be called with joint values, poses, or without any
		# parameters if you have already set the pose or joint target for the group
		self.arm_move_group.go(joint_goal, wait=True)

	def move_complete_pick(self,data):
		self.in_range = True
		print("Don't call me twice")

	def move_gripper_down_to_grasp(self, data):
		EE_dist_to_goal = np.infty
		#if self.state == "planning grasp":
		if True:
			pose_goal = geometry_msgs.msg.Pose()

			#dist = np.linalg.norm(np.array([data.pose.position.z, data.pose.position.x]))

			if self.in_range:
				self.open_gripper()
				pose_goal.position.x = data.pose.position.z - 0.1675
				pose_goal.position.y = -1*data.pose.position.x + 0.02
				pose_goal.position.z = 0.023

				v = np.matrix([0,1,0]) #pitch about y-axis
				th = 80*np.pi/180. #pitch by 45deg
				#note that no rotation is th= 0 deg

				pose_goal.orientation.x = v.item(0)*np.sin(th/2)
				pose_goal.orientation.y = v.item(1)*np.sin(th/2)
				pose_goal.orientation.z = v.item(2)*np.sin(th/2)
				pose_goal.orientation.w = np.cos(th/2)

				self.arm_move_group.set_pose_target(pose_goal)
				print("planning")
				plan = self.arm_move_group.go(wait=True)

				# Calling `stop()` ensures that there is no residual movement
				self.arm_move_group.stop()
				# It is always good to clear your targets after planning with poses.
				# Note: there is no equivalent function for clear_joint_value_targets()
				self.arm_move_group.clear_pose_targets()
				self.in_range = False
				self.state = "grasping"

			#print(dist)					
			
			print('Moving arm')

			current_pose = self.arm_move_group.get_current_pose()
			EE_dist_to_goal = np.linalg.norm([pose_goal.position.x - current_pose.pose.position.x, pose_goal.position.y - current_pose.pose.position.y,
						pose_goal.position.z - current_pose.pose.position.z])

		if EE_dist_to_goal < 0.005 and self.state == "grasping":
				self.close_gripper()
				self.state = "homing arm"

		if self.state == "homing arm":
			self.state = "done"
			self.move_arm_down_for_camera()
			self.grasp_complete.publish(self.state)

	def move_gripper_down_to_place(self, data):
		pose_goal = geometry_msgs.msg.Pose()

		pose_goal.position.x = 0.5
		pose_goal.position.y = 0
		pose_goal.position.z = 0.023

		v = np.matrix([0,1,0]) #pitch about y-axis
		th = 80*np.pi/180. #pitch by 45deg
		#note that no rotation is th= 0 deg

		pose_goal.orientation.x = v.item(0)*np.sin(th/2)
		pose_goal.orientation.y = v.item(1)*np.sin(th/2)
		pose_goal.orientation.z = v.item(2)*np.sin(th/2)
		pose_goal.orientation.w = np.cos(th/2)

		self.arm_move_group.set_pose_target(pose_goal)

		# now we call the planner to compute and execute the plan
		plan = self.arm_move_group.go(wait=True)
		# Calling `stop()` ensures that there is no residual movement
		self.arm_move_group.stop()
		# It is always good to clear your targets after planning with poses.
		# Note: there is no equivalent function for clear_joint_value_targets()
		self.arm_move_group.clear_pose_targets()
		#self.state == "releasing"

		#current_pose = self.arm_move_group.get_current_pose()
		#EE_dist_to_goal = np.linalg.norm([pose_goal.position.x - current_pose.pose.position.x, pose_goal.position.y - current_pose.pose.position.y,
		#				pose_goal.position.z - current_pose.pose.position.z])

		#if EE_dist_to_goal < 0.005 and self.state == "releasing":
		self.open_gripper()
		#self.state = "homing arm"

		#if self.state == "homing arm":
		#self.state = "done"
		self.move_arm_down_for_camera()
		self.grasp_complete.publish(self.state)


def main():


	rospy.init_node('locobot_arm_motion')

	moveit_commander.roscpp_initialize(sys.argv)
	print(sys.argv)

	# Point the camera toward the blocks
	camera_orient_obj = OrientCamera()
	camera_orient_obj.tilt_camera(angle=0.7)

	move_arm_obj = MoveLocobotArm(moveit_commander=moveit_commander)
	move_arm_obj.display_moveit_info()
	move_arm_obj.move_arm_down_for_camera()

	#Uncomment below to move gripper down for grasping (note the frame is baselink; z=0 is the ground (hitting the ground!))
	# move_arm_obj.close_gripper()
	# move_arm_obj.move_gripper_down_to_grasp()
	# move_arm_obj.open_gripper()

	camera_orient_obj.tilt_camera(angle=0.7)

	rospy.Subscriber("/locobot/camera_cube_locator", Marker, move_arm_obj.move_gripper_down_to_grasp)
	rospy.Subscriber("/locobot/mobile_base/move_complete_pick", String, move_arm_obj.move_complete_pick)
	rospy.Subscriber("/locobot/mobile_base/move_complete", String, move_arm_obj.move_gripper_down_to_place)
	rospy.spin()


if __name__ == '__main__':
	main()
