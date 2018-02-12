from player import Player
from supporter import Supporter
import operator
import matplotlib.pyplot as plt
import numpy as np
import csv
from collections import namedtuple 
import seaborn as sns
import pandas as pd 
from field import * 
import random


## Record each independent game's scores for player and opponent respectively.
w_h = 0.6
decay = 0.8
central_alpha = 0.4

def eachMove(player, opponent, w_h, trail_central_alpha):

	# if win, break
	if player.current_state[0] in player_goal:
		return (1, 0)
	elif opponent.current_state[0] in opponent_goal:
		return (0, 1)
	initialState_p = player.current_state
	initialState_o = opponent.current_state

	# take action
	pAction = player.takeAction()
	oAction = opponent.takeAction()

	meetUp(player, opponent)

	finalState_p = newState(player, pAction)
	finalState_o = newState(opponent, oAction)
	# get reward
	r_p = underlyReward(player, opponent)[0]
	r_o = underlyReward(player, opponent)[1]
	h_p = getpartitionR(player)

	player.updateQ(initialState_p, finalState_p, pAction, oAction, r_p, h_p, w_h,trail_central_alpha)
	opponent.updateQ(initialState_o, finalState_o, oAction, pAction, r_o)

	player.current_state = finalState_p
	opponent.current_state = finalState_o
	return (0, 0)
	

############################################################################################################################################
print('no-agents-overlap, pr = 10, SPQ')

data_dict = {}

for k in range(Experiment):
	
	print('[Experiment] = ', k)
	playerWin = [0] * Game 
	opponentWin = [0] * Game 

	for i in range(Trial):
		print('Trial =', i)
		#re-inirialize all players and opponents
		central_Q = playerQ.copy()
		central_Pi = playerPi.copy()
		supporter_balls = [random.choice([True, False])]*3
		supporter_starts = [(player_start[i], supporter_balls[i]) for i in range(3)]
		opponent_starts =  [(opponent_start[i], not supporter_balls[i]) for i in range(3)]
		
		support1 = Supporter(player_goal, states, actions, central_Q, central_Pi, supporter_starts[0])
		support2 = Supporter(player_goal, states, actions, central_Q, central_Pi, supporter_starts[1])
		support3 = Supporter(player_goal, states, actions, central_Q, central_Pi, supporter_starts[2])

		opponent1 = Player(opponent_goal, states, actions, opponentQ.copy(), opponentPi.copy(), opponent_starts[0])
		opponent2 = Player(opponent_goal, states, actions, opponentQ.copy(), opponentPi.copy(), opponent_starts[1])
		opponent3 = Player(opponent_goal, states, actions, opponentQ.copy(), opponentPi.copy(), opponent_starts[2])

		player  = Player(player_goal, states, actions, central_Q, playerPi.copy(), supporter_starts[1])
		opponent = Player(opponent_goal, states, actions, opponentQ.copy(), opponentPi.copy(), opponent_starts[1])

		trial_w_h = w_h   # balance param for Q and H mix
		trail_central_alpha = central_alpha # Annealing the Q value
		
		for j in range(Game):
	
			# renew the start state
			support1.current_state = supporter_starts[0]
			support2.current_state = supporter_starts[1]
			support3.current_state = supporter_starts[2]

			opponent1.current_state = opponent_starts[0]
			opponent2.current_state = opponent_starts[1]
			opponent3.current_state = opponent_starts[2]

			support1.subField = subFields['front']
			support2.subField = subFields['middle']
			support3.subField = subFields['back']

			player.current_state = supporter_starts[1]
			opponent.current_state = opponent_starts[1]

			tag = [1]*4

			for m in range(Movement):

				if tag[0] == 1:
					eachstep = eachMove(support1, opponent1, trial_w_h, trail_central_alpha)

					if max(eachstep) == 1:
						tag[0] = 0
					
				if tag[1] == 1:
					if max(eachMove(support2, opponent2, trial_w_h,trail_central_alpha)) == 1:
						tag[1] = 0

				if tag[2] == 1:
					if max(eachMove(support3, opponent3, trial_w_h,trail_central_alpha)) == 1:
						tag[2] = 0
				# Main group of player and opponent
				if tag[3] == 1:

					if player.current_state[0] in player_goal:
						playerWin[j] = playerWin[j]+ 1
						tag[3] = 0

					elif opponent.current_state[0] in opponent_goal:
						opponentWin[j] = opponentWin[j]+1
						tag[3] = 0
					else:
						initialState_p = player.current_state
						initialState_o = opponent.current_state
						pAction = player.takeAction()
						oAction = opponent.takeAction()
						meetUp(player, opponent)
						finalState_p = newState(player, pAction)
						finalState_o = newState(opponent, oAction)
						r_p = underlyReward(player, opponent)[0]
						r_o = underlyReward(player, opponent)[1]
						player.updateQ(initialState_p, finalState_p, pAction, oAction, r_p)
						opponent.updateQ(initialState_o, finalState_o, oAction, pAction, r_o)
						player.current_state = finalState_p
						opponent.current_state = finalState_o

			trial_w_h = trial_w_h * decay

		del support1, support2, support3, opponent1, opponent2, opponent3, player, opponent
	
	data_dict = {** data_dict, **{k:playerWin}}

df = pd.DataFrame(data_dict) 
df.to_csv('SPQ0_icml.csv')  # 3 agents overlapped; pr = 10



plt.plot(playerWin, label = 'player')
plt.plot(opponentWin, label = 'opponent')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


