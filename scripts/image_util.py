import cv2
import numpy as np
import os
from read_config import read_config

map_x = (read_config()["map_size"][0])
map_y = (read_config()["map_size"][1])
MAP_SHAPE = (((map_x * (20 + 4)) + 4), ((map_y * (20 + 4)) + 4), 3)

up = cv2.imread("/home/thn055/catkin_ws/src/cse_190_final/img/up.jpg")
down = cv2.imread("/home/thn055/catkin_ws/src/cse_190_final/img/down.jpg")
left = cv2.imread("/home/thn055/catkin_ws/src/cse_190_final/img/left.jpg")
right = cv2.imread("/home/thn055/catkin_ws/src/cse_190_final/img/right.jpg")
goal = cv2.imread("/home/thn055/catkin_ws/src/cse_190_final/img/goal.jpg")
wall = cv2.imread("/home/thn055/catkin_ws/src/cse_190_final/img/wall.jpg")
pit = cv2.imread("/home/thn055/catkin_ws/src/cse_190_final/img/pit.jpg")
empty = cv2.imread("/home/thn055/catkin_ws/src/cse_190_final/img/empty.jpg")

img_map = {
	"EMPTY": empty,
	"WALL": wall,
	"PIT": pit,
	"GOAL": goal,
	"N": up,
	"S": down,
	"W": left,
	"E": right
}

height, width, layers = MAP_SHAPE

def save_image_for_iteration(policy_list):
	#Creating an empty map of white spaces
	empty_map = np.zeros(MAP_SHAPE)
	empty_map.fill(255)
	for row in range(len(policy_list)):
		for col in range(len(policy_list[0])):
			new_pos_row = ((row + 1) * 4) + (row * 20)
			new_pos_col = ((col + 1) * 4) + (col * 20)
			empty_map[new_pos_row : new_pos_row + 20, new_pos_col : new_pos_col + 20] = img_map[policy_list[row][col]]
	cv2.imwrite("/home/thn055/catkin_ws/src/cse_190_final/result/path" + ".jpg", empty_map)

