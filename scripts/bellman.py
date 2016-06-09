#!/usr/bin/env python

import rospy
import numpy as np
import math as m
import Queue as Q
from read_config import read_config
from copy import deepcopy

class Bellman():
	def __init__(self):
		self.config = read_config()
		self.cost = 0
		self.is_wall = False
		self.going_back = False
		self.m_size = self.config["map_size"]
		self.c_step = self.config["cost_for_each_step"]
		self.c_wall = self.config["cost_for_hitting_wall"]
		self.c_goal = self.config["cost_for_reaching_goal"]
		self.c_pit = self.config["cost_for_falling_in_pit"]
		self.c_up = self.config["cost_for_up"]
		self.c_down = self.config["cost_for_down"]
		self.c_charge = self.config["cost_for_charge"]
		self.goal = self.config["goal"]
		self.start = self.config["start"]
		self.walls = self.config["walls"]
		self.pits = self.config["pits"]
		self.terrain = self.config["terrain_map"]
		self.charge = self.config["charging_stations"]
		self.fuel = self.config["fuel_capacity"]
		self.path = []
		self.cs_path = []
		self.policies = np.full([self.m_size[0], self.m_size[1]], "EMPTY", dtype=object)
		self.policies[self.goal[0]][self.goal[1]] = "GOAL"
		for wall in self.walls:
			self.policies[wall[0]][wall[1]] = "WALL"
		for pit in self.pits:
			self.policies[pit[0]][pit[1]] = "PIT"
		self.is_cs = False
		self.init_vals()
		self.val_iter(self.goal)

	def init_vals(self):
		"""Initialize all values to 0 except for goal/walls/pits"""
		self.v_map = np.full([self.m_size[0], self.m_size[1]], 0)
		self.old_v = np.full([self.m_size[0], self.m_size[1]], 0)
		self.p_map = np.empty([self.m_size[0], self.m_size[1]], dtype=object)
		self.v_map[self.goal[0]][self.goal[1]] = self.c_goal
		self.old_v[self.goal[0]][self.goal[1]] = self.c_goal
		self.p_map[self.goal[0]][self.goal[1]] = "GOAL"
		for wall in self.walls:
			self.v_map[wall[0]][wall[1]] = self.c_wall
			self.old_v[wall[0]][wall[1]] = self.c_wall
			self.p_map[wall[0]][wall[1]] = "WALL"
		for pit in self.pits:
			self.v_map[pit[0]][pit[1]] = self.c_pit
			self.old_v[pit[0]][pit[1]] = self.c_pit
			self.p_map[pit[0]][pit[1]] = "PIT"
		for station in self.charge:
			self.v_map[station[0]][station[1]] = self.c_charge
			self.old_v[station[0]][station[1]] = self.c_charge

	def val_iter(self, goal):
		"""For all k = 1 -> max_iter"""
		for k in range(1, self.config["max_iterations"]):
			"""For all states other than walls/pits/goal"""
			for x in range(0, self.m_size[0]):
				for y in range(0, self.m_size[1]):
					if self.checks([x,y], goal):
						sums = []
						"""For all actions in (N,S,W,E)"""
						for i in range(0, 4):
							"""Find cost to calculate value"""
							"""Add all vals from the moves to make summation"""
							summation = self.calc_vals([x,y], i, goal)
							sums.append(summation)
						"""Pick action with min sum and update policy map"""
						p = sums.index(min(sums))
						"""Update value map"""
						self.v_map[x][y] = min(sums)
						if p == 0:
							self.p_map[x][y] = "N"
						elif p == 1:
							self.p_map[x][y] = "S"
						elif p == 2:
							self.p_map[x][y] = "W"
						elif p == 3:
							self.p_map[x][y] = "E"
			self.old_v = deepcopy(self.v_map)
	
	def checks(self, state, goal):
		self.is_wall = False
		self.cost = self.c_step
		"""Check if is wall/pit/goal and find cost for state"""
		if state in self.walls:
			self.cost = self.c_wall 
			self.is_wall = True
			return False
		elif state in self.pits:
			return False
		elif state == goal:
			return False
		else:
			self.cost = self.c_step
			return True

	def out_bounds(self, state):
		"""Check if out of bounds"""
		x = state[0]
		y = state[1]
		if x < 0 or x == self.m_size[0] or y < 0 or y == self.m_size[1]:
			self.cost = self.c_wall
			self.is_wall = True
		else:
			"""Add in cost of terrain"""
			terrain = self.terrain[state[0]][state[1]] 
			if terrain == "U":
				t_cost = self.c_up
				if self.going_back:
					t_cost = self.c_down
			elif terrain == "D":
				t_cost = self.c_down
				if self.going_back:
					t_cost = self.c_up
			elif terrain == "F":
				t_cost = 0
			self.cost += t_cost

	def calc_vals(self, state, move, goal):
		x = state[0]
		y = state[1]
		val = 0
		self.going_back = False
		if move == 0:
			"""North"""
			f = [x - 1, y]
		elif move == 1:
			"""South"""
			f = [x + 1, y]
			self.going_back = True
		elif move == 2:
			"""West"""
			f = [x, y - 1]
			self.going_back = True
		elif move == 3:
			"""East"""
			f = [x, y + 1]

		self.checks(f, goal)
		self.out_bounds(f)
		if self.is_wall:
			f = state
		val = self.cost + (self.config["discount_factor"] * self.old_v[f[0]][f[1]])
		return val
	
	def traverse_path(self, state, direc, policy, path):
		while direc != "GOAL":
			self.policies[state[0], state[1]] = direc
			if direc == "N":
				state = [state[0] - 1, state[1]]
			elif direc == "S":
				state = [state[0] + 1, state[1]]
			elif direc == "W":
				state = [state[0], state[1] - 1]
			elif direc == "E":
				state = [state[0], state[1] + 1]
			direc = policy[state[0]][state[1]]
			path.append(state)
	
	def find_path(self):
		f = self.start
		direction = self.p_map[self.start[0]][self.start[1]]
		self.path.append(self.start)
		self.policies[f[0]][f[1]] = direction
		self.traverse_path(f, direction, self.p_map, self.path)
		
		"""Calculate fuel needed to reach goal"""
		c_fuel = 0
		for state in self.path:
			terrain = self.terrain[state[0]][state[1]] 
			t_cost = 0
			if terrain == "U":
				t_cost = self.c_up
			elif terrain == "D":
				t_cost = self.c_down
			c_fuel += self.c_step + t_cost
		"""Account for 1 over step"""
		c_fuel -= self.c_step
		
		"""If have enough fuel, just use the optimal path"""
		if self.fuel > c_fuel:
			return True
		else: 
			"""If car doesn't have enought fuel to get to goal"""
			"""Find optimal path with a charging station"""
			poss_paths = []
			orig_p = deepcopy(self.p_map)
			self.is_cs = True
			for	charge in self.charge:
				self.policies = np.full([self.m_size[0], self.m_size[1]], "EMPTY", dtype=object)
				for wall in self.walls:
					self.policies[wall[0]][wall[1]] = "WALL"
				for pit in self.pits:
					self.policies[pit[0]][pit[1]] = "PIT"
				self.policies[self.goal[0]][self.goal[1]] = "GOAL"
				temp_path = []
				"""Save path from charging station -> goal"""
				cs = charge
				direction = orig_p[cs[0]][cs[1]]
				self.traverse_path(cs, direction, orig_p, temp_path)
			
				self.v_map = np.full([self.m_size[0], self.m_size[1]], 0)
				self.old_v = np.full([self.m_size[0], self.m_size[1]], 0)
				self.p_map = np.empty([self.m_size[0], self.m_size[1]], dtype=object)
				self.v_map[charge[0]][charge[1]] = self.c_goal
				self.old_v[charge[0]][charge[1]] = self.c_goal
				self.p_map[charge[0]][charge[1]] = "GOAL"
				for wall in self.walls:
					self.v_map[wall[0]][wall[1]] = self.c_wall
					self.old_v[wall[0]][wall[1]] = self.c_wall
					self.p_map[wall[0]][wall[1]] = "WALL"
				for pit in self.pits:
					self.v_map[pit[0]][pit[1]] = self.c_pit
					self.old_v[pit[0]][pit[1]] = self.c_pit
					self.p_map[pit[0]][pit[1]] = "PIT"
				
				"""Run Bellman-Ford to find optimal path from start -> charging station"""
				"""And combine with path from station -> goal to get final path"""
				self.val_iter(charge)
				x = self.start
				direction = self.p_map[x[0]][x[1]]
				count = 0
				self.policies[x[0]][x[1]] = direction
				temp_path.insert(count, x)
	
				while direction != "GOAL":
					self.policies[x[0]][x[1]] = direction
					if direction == "N":
						x = [x[0] - 1, x[1]]
					elif direction == "S":
						x = [x[0] + 1, x[1]]
					elif direction == "W":
						x = [x[0], x[1] - 1]
					elif direction == "E":
						x = [x[0], x[1] + 1]
					count += 1
					direction = self.p_map[x[0]][x[1]]
					temp_path.insert(count, x)
				poss_paths.append(temp_path)
			"""Determine which path with a charging station is the shortest"""
			min_path = len(poss_paths[0])
			optimal = poss_paths[0]
			for path in poss_paths:
				if min_path > len(path):
					min_path = len(path)
					optimal = path
			self.cs_path = optimal
			return False
