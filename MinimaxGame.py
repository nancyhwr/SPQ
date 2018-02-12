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


############################################################################################################################################
print('regular MinimaxQ vs MinimaxQ')

data_dict = {}

for k in range(Experiment):

	playerWin = [0] * Game 
	opponentWin = [0] * Game 

	print('Experiment = ', k)


	for i in range(Trial):
		print('Trial =', i)
	
		ball = random.choice([True, False])
		pstart = (player_start[1], ball)
		ostart = (opponent_start[1], not ball)
		
		player  = Player(player_goal, states, actions, playerQ.copy(), playerPi.copy(), pstart)
		opponent = Player(opponent_goal, states, actions, opponentQ.copy(), opponentPi.copy(), ostart)

		for j in range(Game):

			player.current_state = pstart
			opponent.current_state = ostart

			for m in range(Movement):

				if player.current_state[0] in player_goal and player.current_state[1]:
					playerWin[j] = playerWin[j]+ 1
					break
						
				elif opponent.current_state[0] in opponent_goal and opponent.current_state[1]:
					opponentWin[j] = opponentWin[j]+1
					break
						
					
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

		
		del player, opponent

	data_dict = {**data_dict,**{k: playerWin}}


df = pd.DataFrame(data_dict) 
df.to_csv('minimax-minimax_icml.csv') 

plt.plot(playerWin, label = 'player')
plt.plot(opponentWin, label = 'opponent')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


















