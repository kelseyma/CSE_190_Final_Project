#!/usr/bin/env python

#robot.py implementation goes here

import rospy
import numpy as np
import image_util
from read_config import read_config
from bellman import Bellman

class Robot():
	def __init__(self):
		rospy.init_node("robot")
		self.config = read_config()
		self.do_bellman()
		rospy.sleep(2)
		rospy.signal_shutdown("done")

	def do_bellman(self):
		bellman = Bellman()
		rospy.sleep(2)
		if bellman.find_path():
			print "values: \n", bellman.v_map
			print "policies: \n", bellman.p_map
			print "path: ", bellman.path
			print "policy path: \n", bellman.policies
			image_util.save_image_for_iteration(bellman.policies)
		else:
			print "charging path: ", bellman.cs_path
			print "policy path: \n", bellman.policies
			image_util.save_image_for_iteration(bellman.policies)

if __name__ == '__main__':
	r = Robot()
