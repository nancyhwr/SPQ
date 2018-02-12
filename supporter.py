from collections import namedtuple 
from math import exp
import random
import operator
from scipy.optimize import linprog
import numpy as np
import math
from ast import literal_eval as make_tuple
from math import exp

# Player file: epsilon-greedy exploration; MinimaxQ Learning


class Supporter:

	
	def __init__(self, goal,states, actions, Q, pi,start_state, alpha=.4, decay = .9999954, gamma = .9, epsilon=.6):  #T, cool_down, subField, partition_r,
	
		self.goal = goal
		self.actions = actions
		self.states = states
		self.Q = Q
		self.localQ = self.Q.copy()
		self.Q_p = self.Q.copy()
		self.pi = pi
		self.V =  {s: 0 for s in self.states}
		for g in self.goal:
			self.V[(g, True)] = 1000
		self.H = {(s, a, o):0 for s in self.states for a in self.actions for o in self.actions}
	
		#######################
		self.alpha = alpha
		self.decay = decay
		self.gamma = gamma
		self.epsilon = epsilon
		#######################
		self.T = 0
		self.cool_down = 0
		#######################
		self.partition_r = 0
		self.subField = []
		#######################
		self.current_state = start_state
		
	######### Take actions following epsilon-greedy policy

	def takeAction(self):

		if self.current_state[0] in self.subField:
			return self.exploration()
		else:
			return self.exploit()


	def exploration(self):

		if random.uniform(0, 1) < self.epsilon:
			return random.choice(self.actions)
		else:
			prob = [self.pi[self.current_state, a] for a in self.actions]
			for i in range(len(prob)):
				if prob[i] < 0:
					prob[i] = 0
			prob = [prob[i]/sum(prob) for i in range(len(prob))]
			action = np.random.choice(len(self.actions),1, replace=False, p = prob) 
			return self.actions[action[0]]


	def exploit(self):
	
		prob = [self.pi[self.current_state, a] for a in self.actions]
		maxp = max(prob)
		count = prob.count(maxp)
		best = [i for i in range(len(self.actions)) if prob[i] == maxp]
		i = random.choice(best)
		return self.actions[i]

	

	def updateQ(self, initialState, finalState, a, o , r_p, h_p, w_h, central_alpha, restrictActions=None):
		
		# if finalState[0] in self.goal or finalState[0] in self.subField:
		self.localQ[initialState, a, o] = (1 - self.alpha) * self.localQ[initialState, a, o] + \
				self.alpha * (r_p + self.gamma * self.V[finalState])
		
		self.H[initialState, a, o] = (1 - self.alpha) * self.H[initialState, a, o] + \
				self.alpha * (h_p + self.gamma * max([self.H[finalState, p_a, o_a] for p_a in self.actions for o_a in self.actions]))

		self.Q_p[initialState, a, o] = (1-w_h)*self.localQ[initialState, a, o] + w_h*self.H[initialState, a, o]
		
		self.Q[initialState,a, o] = (1-central_alpha)* self.Q[initialState, a, o] + central_alpha * self.localQ[initialState, a, o]
	
		self.V[initialState] = self.updatePolicy(initialState)  
		#self.alpha = self.alpha * self.decay

### Get a mix strategy!
	def updatePolicy(self, state, retry=False):

		numActionsB, numActionsA = len(self.actions), len(self.actions)
		c = np.zeros(numActionsA + 1)
		c[0] = -1
		A_ub = np.ones((numActionsB, numActionsA + 1))
		A_ub_local = A_ub.copy()
		A_ub[:, 1:] = [[-1 * self.Q_p[state, a, o] for a in self.actions] for o in self.actions] 
		A_ub_local[:, 1:] = [[-1 * self.localQ[state, a, o] for a in self.actions] for o in self.actions] 
		b_ub = np.zeros(numActionsB)
		A_eq = np.ones((1, numActionsA + 1))
		A_eq[0, 0] = 0
		b_eq = [1]
		bounds = ((None, None),) + ((0, 1),) * numActionsA
		res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
		res_local = linprog(c, A_ub=A_ub_local, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)

		if res.success: # and self.satisfyBound(res.x[1:])
			# le = res.x[1:]
			# self.normalPi(state, le)
			self.updatePi(state, self.modifyProb(res.x[1:], state))#self.modifyProb(res.x[1:], state)
		elif not retry:
			self.updatePolicy(state, retry=True)
			#return self.updatePolicy(state, retry=True)
		if res_local.success:
			return res_local.x[0]
		return self.V[state]
		#return res_local.x[0]
	

	def satisfyBound(self, le):
		for i in range(len(le)):
			if le[i] < 0:
				return False
		return True

	def modifyProb(self, probRes, state):
		start = False
		listLen = 1
		matchList = [math.inf] * len(self.actions)
		result = [0] * len(self.actions)

		if 1 in probRes:
			for i in range(len(probRes)):
				newList = list(self.Q[state, self.actions[i], o] for o in self.actions)
				if start == True and newList == matchList:
					result[i] = 1
				if probRes[i] == 1:
					result[i] = 1
					matchList = list(self.Q[state, self.actions[i], o] for o in self.actions)
					start = True

		for j in range(len(probRes)):
			if result[j] == 1:
				probRes[j] = 1/sum(result)
		return probRes


	def updatePi(self, state, probRes):
		for i in range(len(self.actions)):
			self.pi[state, self.actions[i]] = probRes[i]

	def updateState(self, new_state):
		self.current_state = new_state