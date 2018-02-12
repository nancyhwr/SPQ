import operator
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import namedtuple 
import seaborn as sns
import pandas as pd 

## The Soccer Field basic info


####################################  Variables   ###########################################################
Experiment = 1
Trial = 10
Game = 300
Movement = 50
score = 1000


field = (9, 4)
player_goal = [(-1, 1), (-1, 2)]
opponent_goal = [(9, 1), (9, 2)]
field_loc = [(x,y) for x in range(field[0]) for y in range(field[1])]
states = [(s, ball) for s in field_loc+player_goal+opponent_goal for ball in [True, False]]
actions = [ (0, 1), (1, 0),(-1, 0), (0, -1)]

player_start = [(4, 1),(4, 1),(4, 1)]
opponent_start = [(3, 2),(3, 2),(3, 2)]


playerQ = {(state, p_action, o_action): 0 for state in states for p_action in actions for o_action in actions}
opponentQ = {(state, o_action, p_action): 0 for state in states for o_action in actions for p_action in actions}
playerPi = {(state, action): 1/len(actions) for state in states for action in actions}
opponentPi = {(state, action): 1/len(actions) for state in states for action in actions}


partition_r = 30
goal_r = 1000

####################################  Overlaps   ###########################################################
# front_field = [(x, y) for x in np.arange(0, int(field[0]/3)+1, 1) for y in range(field[1])] 
# middle_field = [(x, y) for x in np.arange(int(field[0]/3)-1, int(2*field[0]/3)+1, 1) for y in range(field[1])] 
# back_field = [(x, y) for x in np.arange(int(2*field[0]/3-1), int(field[0]), 1) for y in range(field[1])]  

front_field = [(x, y) for x in np.arange(0, int(field[0]/3), 1) for y in range(field[1])] 
middle_field = [(x, y) for x in np.arange(int(field[0]/3), int(2*field[0]/3)+1, 1) for y in range(field[1])] 
back_field = [(x, y) for x in np.arange(int(2*field[0]/3), int(field[0]), 1) for y in range(field[1])]  

# front_field = [(x, y) for x in np.arange(0, int(field[0]/3)+2, 1) for y in range(field[1])] 
# middle_field = [(x, y) for x in np.arange(int(field[0]/3)-2, int(2*field[0]/3)+2, 1) for y in range(field[1])] 
# back_field = [(x, y) for x in np.arange(int(2*field[0]/3-2), int(field[0]), 1) for y in range(field[1])]  

wholefield = [(x, y) for x in range(field[0]) for y in range(field[1])]
subFields = {'front': front_field, 'middle': middle_field, 'back': back_field}



####################################  Functions   ###########################################################

def getGoal(player):

	if player.current_state in player.goal and player.ball:
		return True
	else:
		return False


def meetUp(player, opponent):
	if player.current_state[0] == opponent.current_state[0]:
		if player.current_state[1] == True:
			player.current_state = tuple((player.current_state[0], False))
			opponent.current_state = tuple((opponent.current_state[0], True))
		else:
			player.current_state = tuple((player.current_state[0], True))
			opponent.current_state = tuple((opponent.current_state[0], False))


def newState(player, action):
	new_s =  (tuple(map(operator.add, player.current_state[0], action)), player.current_state[1])
	if (new_s in states) or ((new_s[0] in player.goal) and (new_s[1] == True)):
		return new_s
	else:
		return player.current_state
		
	
def getpartitionR(player):
	if player.subField != None:
		#if player.current_state in subFields[player.subField]:
		if player.current_state[0] in player.subField:
			return partition_r  
		else:
			return 0
	else: 
		return 0
		

def underlyReward(player, opponent):  ## Return the (player.reward, opponent.reward)
	if (player.current_state[0] in player_goal) and (player.current_state[1] == True) :
		return (goal_r, -goal_r)
	elif (opponent.current_state[0] in opponent_goal) and (opponent.current_state[1] == True):
		return (-goal_r, goal_r)
	else:
		return (0, 0)
