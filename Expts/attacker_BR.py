import numpy as np 
from docplex.mp.model import Model
import gurobipy as gp
import time

NUMTYPES = 3 # Attacker Types
NUMATTACKS = 292 # Max no. of attacks
NUMCONFIGS = 4
MAX_ITER = 10
T = 1000
GAMMA = 0.5 # Exploration Parameter
EPSILON = 0.1
ALPHA = 0.1
DISCOUNT_FACTOR = 0.8
M = 1000000
Lmax = 1000
PADDING = 1/(T**(1/3))

FPL_ETA = np.sqrt(np.log(NUMCONFIGS)/(NUMCONFIGS*T)) #FPL Hyperparameter
EXP_ETA = np.sqrt(2*np.log(NUMCONFIGS)/(NUMCONFIGS*T)) #EXP Hyperparameter

strats_list = ["FPLMTD", "FPLMTDLite", "DOBSS", "RANDOM", "RobustRL", "EXP3", "BSSQ", "PaddedExp3", "SwitchingExp3", "FPLGR"]

NUMSTRATS = 10
FPLMTD = 0
FPLMTDLite = 1
DOBSS = 2
RANDOM = 3
RobustRL = 4
EXP3 = 5
BSSQ = 6
PaddedExp3 = 7
SwitchingExp3 = 8
FPLGR = 9
 
# returns defender and attacker utilities
def parse_util():
	global NUMTYPES
	def_util = []
	att_util = []
	f = open("r_utilities.txt", "r")
	y = f.readline()
	NUMTYPES = int(y)
	for t in range(NUMTYPES):
		d = f.readline().split()
		d = [float(item) for item in d]
		def_util.append(d)
		a = f.readline().split()
		a = [float(item) for item in a]
		att_util.append(a)
	def_util = np.array(def_util)
	att_util = np.array(att_util)
	f.close()
	return def_util, att_util

# get number of attacks
def parse_attacks():
	global NUMATTACKS
	f = open("r_attacks.txt", "r")
	y = f.readline()
	NUMATTACKS = int(y)
	f.close()

# returns 0-1 vulnerabilities 2D matrix for (config, attack)
def parse_vulset():
	global NUMCONFIGS, NUMATTACKS
	vul_set = []
	f = open("r_vulnerabilities.txt", "r")
	y = f.readline()
	for c in range(NUMCONFIGS):
		vul = [0]*NUMATTACKS
		y = f.readline()
		lis = f.readline().split()
		lis = [int(item) for item in lis]
		for i in lis:
			vul[i] = 1
		vul_set.append(vul)
	vul_set = np.array(vul_set)
	return vul_set

# returns switching cost 2D matrix
def parse_switching():
	global NUMCONFIGS
	sc = []
	f = open("r_switching.txt", "r")

	s = f.readline().split()
	NUMCONFIGS = len(s)
	s = [float(item) for item in s]
	sc.append(s)

	for c in range(NUMCONFIGS-1):
		s = f.readline().split()
		s = [float(item) for item in s]
		sc.append(s)
	sc = np.array(sc)
	return sc

# returns 3D utilities matrix for (attacker type, config, attack) for attacker
# and corresponding defender ultility matrix
def parse_game_utils(def_util, att_util, vul_set):
	game_def_util = [[[0.0]*NUMATTACKS for i in range(NUMCONFIGS)] for j in range(NUMTYPES)]
	game_att_util = [[[0.0]*NUMATTACKS for i in range(NUMCONFIGS)] for j in range(NUMTYPES)]
	for tau in range(NUMTYPES):
		for c in range(NUMCONFIGS):
			for a in range(NUMATTACKS):
				if(vul_set[c,a] == 1):
					game_def_util[tau][c][a] = def_util[tau][a]
					game_att_util[tau][c][a] = att_util[tau][a]
	return game_def_util, game_att_util

# returns x values if solution to MIQP exists when starting
def getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec):
	X = Model(name = 'dobss_init')
	# rewards
	X.ra = game_att_util
	X.rd = game_def_util
	X.P = Pvec # attacker type prob
	X.sc = sc # switching costs
	X.m = M # large constant

	# defender mixed strategy
	x = {i: X.continuous_var(name = 'x_'+str(i), lb = 0, ub = 1) for i in range(NUMCONFIGS)}
	# pure strategies for (attacker type, attack)
	n = {(i, j): X.binary_var(name = 'n_'+str(i)+'_'+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}
	# value of attacker's pure strategy
	v = {i: X.continuous_var(name = 'v_'+str(i)) for i in range(NUMTYPES)}

	# w_ij = x_i * x_j
	w = {(i, j): X.continuous_var(name = 'w_'+str(i)+'_'+str(j), lb = 0, ub = 1) for i in range(NUMCONFIGS) for j in range(NUMCONFIGS)}
	# Σx = 1
	X.add_constraint(X.sum(x) == 1)
	# Σn = 1
	for tau in range(NUMTYPES):
		X.add_constraint(X.sum(n[tau, a] for a in range(NUMATTACKS)) == 1)

	# value constraints
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			X.add_constraint(v[tau] - X.sum(X.ra[tau][c][a]*x[c] for c in range(NUMCONFIGS)) >= 0)
			X.add_constraint(v[tau] - X.sum(X.ra[tau][c][a]*x[c] for c in range(NUMCONFIGS)) <= (1 - n[tau, a])*X.m)

	# w constraints
	X.add_constraint(X.sum(w) == 1)
	for c in range(NUMCONFIGS):
		for cdash in range(NUMCONFIGS):
			X.add_constraint(w[c, cdash] <= x[c])
			X.add_constraint(w[c, cdash] <= x[cdash])

	for c in range(NUMCONFIGS):
		X.add_constraint(X.sum(w[c, cdash] for cdash in range(NUMCONFIGS)) <= x[c])
		X.add_constraint(X.sum(w[cdash, c] for cdash in range(NUMCONFIGS)) <= x[c])

	# maximise total reward - switching cost
	X.maximize(X.sum(X.P[tau]*X.rd[tau][c][a]*x[c]*n[tau, a] for c in range(NUMCONFIGS) for tau in range(NUMTYPES) for a in range(NUMATTACKS)) - X.sum(X.sc[i, j]*w[i, j] for i in range(NUMCONFIGS) for j in range(NUMCONFIGS)))
	sol = X.solve()
	if(sol == None):
		print("Error: No solution instance")
		exit()
	# sol.display()
	soln_x = []
	# return x values
	for c in range(NUMCONFIGS):
		soln_x.append(x[c].solution_value)
	return soln_x


# returns x values if solution to MIQP exists given a DOBSS strategy
def getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, DOBSS_strat):
	X = Model(name = 'dobss_next')
	# rewards
	X.ra = game_att_util
	X.rd = game_def_util

	X.P = Pvec # attacker type prob
	X.sc = sc # switching costs
	X.m = M # large constant

	PureStrat = [0]*NUMCONFIGS
	PureStrat[DOBSS_strat] = 1 
	X.PS = PureStrat

	# defender mixed strategy
	x = {i: X.continuous_var(name = 'x_'+str(i), lb = 0, ub = 1) for i in range(NUMCONFIGS)}
	# pure strategies for (attacker type, attack)
	n = {(i, j): X.binary_var(name = 'n_'+str(i)+'_'+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}
	# value of attacker's pure strategy
	v = {i: X.continuous_var(name = 'v_'+str(i)) for i in range(NUMTYPES)}

	# Σx = 1
	X.add_constraint(X.sum(x) == 1) 
	# Σn = 1
	for tau in range(NUMTYPES):
		X.add_constraint(X.sum(n[tau, a] for a in range(NUMATTACKS)) == 1)
	# value constraints 
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			X.add_constraint(v[tau] - X.sum(X.ra[tau][c][a]*x[c] for c in range(NUMCONFIGS)) >= 0)
			X.add_constraint(v[tau] - X.sum(X.ra[tau][c][a]*x[c] for c in range(NUMCONFIGS)) <= (1 - n[tau, a])*X.m)

	# maximise total reward - switching cost
	X.maximize(X.sum(X.P[tau]*X.rd[tau][c][a]*x[c]*n[tau, a] for c in range(NUMCONFIGS) for tau in range(NUMTYPES) for a in range(NUMATTACKS)) - X.sum(X.sc[i, j]*X.PS[i]*x[j] for i in range(NUMCONFIGS) for j in range(NUMCONFIGS)))
	sol = X.solve()
	if(sol == None):
		print("Error: No solution instance")
		exit()
	# sol.display()
	# return x values
	soln_x = []
	for c in range(NUMCONFIGS):
		soln_x.append(x[c].solution_value)
	return soln_x

# get strategy from distribution
def getStratFromDist(x):
	y = np.random.random()
	for i in range(len(x)):
		if(y <= x[i]):
			return i
		y -= x[i]
	return len(x) - 1 


# returns BSG equilibrium values
def getSSEq(game_def_qval, game_att_qval):
	m = gp.Model("MIQP")

	m.setParam('OutputFlag', 0)
	m.setParam('LogFile', '')
	m.setParam('LogToConsole', 0)
	
	# defender mixed strategy
	x = {i: m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = 'x_'+str(i)) for i in range(NUMCONFIGS)}
	m.update()

	# Add defender stategy constraints: Σx = 1
	x_sum = gp.LinExpr()
	for i in range(NUMCONFIGS):
		x_sum.add(x[i])
	m.addConstr(x_sum == 1)

	# declare objective function
	obj = gp.QuadExpr()

	# pure strategies for (attacker type, attack)
	q = {i: m.addVar(lb = 0, ub = 1, vtype = gp.GRB.INTEGER, name = 'q_'+str(i)) for i in range(NUMATTACKS)}
	
	# value of attacker's pure strategy
	v_a = m.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype = gp.GRB.CONTINUOUS, name = 'v_a')

	m.update()

	# Update objective function
	for sdash in range(NUMCONFIGS):
		for a in range(NUMATTACKS):
			obj.add( game_def_qval[sdash][a] * x[sdash] * q[a] )

	# Add constraints to make attacker have a pure strategy
	q_sum = gp.LinExpr()
	for a in range(NUMATTACKS):
		q_sum.add(q[a])
	m.addConstr(q_sum==1)

	# Add constraints to make attacker select dominant pure strategy
	for a in range(NUMATTACKS):
		val = gp.LinExpr()
		val.add(v_a)
		for sdash in range(NUMCONFIGS):
			val.add(float(game_att_qval[sdash][a]) * x[sdash], -1.0)
		m.addConstr(val >= 0)
		m.addConstr(val <= (1 - q[a]) * M)

	# set objective funcion
	m.setObjective(obj, gp.GRB.MAXIMIZE)

	# Solve MIQP
	m.optimize()

	# return x, q and values
	soln_x = [0.0 for i in range(NUMCONFIGS)]
	soln_q = [0.0 for i in range(NUMATTACKS)]

	for sdash in range(NUMCONFIGS):
		soln_x[sdash] = x[sdash].X

	for a in range(NUMATTACKS):
		soln_q[a] = q[a].X

	return soln_x, soln_q, m.objVal, v_a.X


def getBSSQStrat(def_util, att_util, sc, p, P, n_episodes):
	x = [[(1/NUMCONFIGS) for i in range(NUMCONFIGS)] for i in range(NUMCONFIGS)]
	q = [[[(1/NUMATTACKS) for a in range(NUMATTACKS)] for i in range(NUMCONFIGS)] for j in range(NUMTYPES)]

	v_def = [[0.0 for i in range(NUMCONFIGS)] for j in range(NUMTYPES)]
	v_att = [[0.0 for i in range(NUMCONFIGS)] for j in range(NUMTYPES)]

	Qval_def = [[[[0.0 for a in range(NUMATTACKS)] for i in range(NUMCONFIGS)] for j in range(NUMCONFIGS)] for k in range(NUMTYPES)]
	Qval_att = [[[[0.0 for a in range(NUMATTACKS)] for i in range(NUMCONFIGS)] for j in range(NUMCONFIGS)] for k in range(NUMTYPES)]

	game_def_reward = np.full((NUMTYPES, NUMCONFIGS, NUMCONFIGS, NUMATTACKS), 0.0)
	game_att_reward = np.full((NUMTYPES, NUMCONFIGS, NUMCONFIGS, NUMATTACKS), 0.0)

	# incorporate switching cost in reward
	for tau in range(NUMTYPES):
		for s in range(NUMCONFIGS):
			for sdash in range(NUMCONFIGS):
				for a in range(NUMATTACKS):
					game_def_reward[tau][s][sdash][a] = def_util[tau][sdash][a] - sc[s][sdash]
					game_att_reward[tau][s][sdash][a] = att_util[tau][sdash][a]

	# epsilon decay from start epsilon value to end epsilon value
	max_eps_len = 10
	start_eps_val = 0.1
	end_eps_val = 0.05
	decay_val = (end_eps_val / start_eps_val) ** (1 / max_eps_len)

	for ep in range(n_episodes):
		print("BSSQ episode:"+str(ep) + "\t", end = "\r")
		# sampling start state
		s = getStratFromDist(p)

		eps_val = start_eps_val

		itr = 0
		while itr <= max_eps_len:
			# sampling attacker type		
			tau = getStratFromDist(P)
			
			# eps-greedy sampling of actions from x, q
			y = np.random.random()

			sdash = None
			a = None
			if(y < eps_val):
				# exploration
				sdash = int(np.random.random()*NUMCONFIGS)
				a = int(np.random.random()*NUMATTACKS)
			else:
				# greedy exploitation
				sdash = np.argmax(x[s])
				a = np.argmax(q[tau][s])

			Qval_def[tau][s][sdash][a] = (1 - ALPHA) * Qval_def[tau][s][sdash][a] + ALPHA * (game_def_reward[tau][s][sdash][a] + DISCOUNT_FACTOR * v_def[tau][sdash])
			Qval_att[tau][s][sdash][a] = (1 - ALPHA) * Qval_att[tau][s][sdash][a] + ALPHA * (game_att_reward[tau][s][sdash][a] + DISCOUNT_FACTOR * v_att[tau][sdash])

			# get BSMG SSEquilibrium values
			for typ in range(NUMTYPES):
				x[s], q[typ][s], v_def[typ][s], v_att[typ][s] = getSSEq(Qval_def[tau][s], Qval_att[typ][s])

			# epsilon decay
			eps_val = eps_val * decay_val
			s = sdash
			itr += 1
	return x

# returns FPL strategy
def getFPLMTDStrat(r, s, old_strat, vulset, P, t):
	# exploration
	gamma = np.random.random()
	if(gamma <= GAMMA):
		return int(np.random.random()*NUMCONFIGS)

	# reward estimates
	rhat = r.copy()
	# switching costs
	shat = s.copy()

	# adding perturbation
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			rhat[tau, a] = rhat[tau, a] - np.random.exponential(1/FPL_ETA)

	# considering utility for attacks that give minimum reward
	# while considering attacker type probability
	u = np.array([0.0]*NUMCONFIGS)
	for c in range(NUMCONFIGS):
		for tau in range(NUMTYPES):
			min1 = 1000
			for a in range(NUMATTACKS):
				if((vulset[c, a] == 1) & (rhat[tau, a] < min1)):
					min1 = rhat[tau, a]
			u[c] += P[tau]*min1
	if(old_strat!= -1):
		u[old_strat]/=np.exp(-ALPHA)
	if(t!=0):
		u = u/(t)
	# net reward
	new_u = [u[c] - shat[old_strat, c] for c in range(NUMCONFIGS)]
	# return the best / leader strategy
	return np.argmax(new_u)


# returns FPL strategy without considering P and vulnerability data
def getFPLMTDLiteStrat(r, s, old_strat, t):
	# exploration
	gamma = np.random.random()
	if(gamma <= GAMMA):
		return int(np.random.random()*NUMCONFIGS)
	rhat = r.copy()
	shat = s.copy()

	# adding perturbation
	for c in range(NUMCONFIGS):
		rhat[c] -= np.random.exponential(1/FPL_ETA)

	if(old_strat != -1):
		rhat[old_strat] /= np.exp(-ALPHA)
	if(t!=0):
		rhat = rhat/(t)
	# net reward
	new_u = [rhat[c] - shat[old_strat, c] for c in range(NUMCONFIGS)]
	# return the best / leader strategy
	return np.argmax(new_u)

# returns FPL+GR strategy
def getFPLGRStrat(r):
	rhat = r.copy()
	for c in range(NUMCONFIGS):
		rhat[c] -= np.random.exponential(1/FPL_ETA)

	return np.argmax(rhat)

# update reward estimates using GR for FPL
def FPLMTD_GR(r, old_strat, strat, vulset, P, util, attack, tau, switch_costs, t):
	rhat = np.copy(r)
	l = 1
	Kr = Lmax
	strat_list = []

	while(l < Lmax):
		strat2 = getFPLMTDStrat(rhat, switch_costs, old_strat, vulset, P, t)
		strat_list.append(strat2)
		if(((vulset[strat2, attack] == 1) & (Kr > l))|(util == 0)):
			Kr = l
		l+=1
		if((Kr < l)):
			break

	rhat[tau, attack] += Kr*util/P[tau]

	return rhat

# update reward estimates using GR for FPL lite
def FPLMTDLite_GR(r, old_strat, strat, util, switch_costs, t):
	rhat = np.copy(r)
	l = 1
	while(l < Lmax):
		strat2 = getFPLMTDLiteStrat(rhat, switch_costs, old_strat, t)
		if(strat2 == strat):
			break
		l+=1
	rhat[strat]+= util*l
	return rhat

def FPL_GR(r, strat, util):
	rhat = np.copy(r)
	l = 1
	while(l < Lmax):
		strat2 = getFPLGRStrat(rhat)
		if(strat2 == strat):
			break
		l+=1
	rhat[strat]+= util*l
	return rhat

# returns RobustRL strategy
def getRobustRLStrat(movelist, utillist):
	if(np.random.random()< EPSILON):
		return int(np.random.random()*NUMCONFIGS)
	l = len(movelist)
	max_util = [-M]*NUMCONFIGS
	for i in range(l):
		if(utillist[i] > max_util[movelist[i]]):
			utillist[i] > max_util[movelist[i]]
	return np.argmin(max_util)

# samples strategy sequentially using EXP3 from p values
def getEXP3Strat(p):
	y = np.random.random()
	for c in range(NUMCONFIGS):
		if(y < p[c]):
			return c
		y -= p[c]
	return NUMCONFIGS - 1

def getPaddedExp3Strat(p):
	z = np.random.random()
	if(z < PADDING):
		y = np.random.random()
		for c in range(NUMCONFIGS):
			if(y < p[c]):
				return c
			y -= p[c]
		return NUMCONFIGS - 1

	return int(np.random.random()*NUMCONFIGS)


# using GR to update attacker reward estimates for FPL-UE
def Attacker_GR(rhat, vdash, util):
	r = rhat.copy()
	i = 1
	l = Lmax # cap value
	while(i < Lmax):
		y = np.random.random()
		if(y < EPSILON):
			# exploration
			v = int(np.random.random()*NUMCONFIGS)
		else:
			# FPL
			rdash = r - np.random.exponential(FPL_ETA, NUMATTACKS)
			v = np.argmax(rdash)
		if(vdash == v):
			l = i
			break
		i+=1
	r[vdash] += l*util
	return r


# getting attack that gives best attack utility
def getAttackBestResponse(def_util, att_util, strat, P, vulset, Mixed_Strat, t):
	y = np.random.random()
	tau = NUMTYPES -1
	for i in range(NUMTYPES):
		if(y < P[i]):
			tau = i
			break
		else:
			y -= P[i]

	# get utilities for attacks
	util_vec = [0.0]*NUMATTACKS
	for a in range(NUMATTACKS):
		u = 0
		for c in range(NUMCONFIGS):
			if(vulset[c, a] == 0):
				u += att_util[tau, a]*Mixed_Strat[a]
		util_vec[a] = u 
	# max attack utility
	attack = np.argmax(util_vec)
	# get corresponding defender utility
	util = 0
	if(vulset[strat, attack] == 1):
		util = def_util[tau, attack]

	# assigning weights according to outcome
	MS = Mixed_Strat.copy()
	for c in range(NUMCONFIGS):
		MS[i] = MS[i]*(t)/(t+1)
		if(i == strat):
			MS[i] += 1/(t+1)

	return util, tau, attack, MS


epsvec = [0.5, 0.6, 0.7, 0.8, 0.9]
Pvec = None

if __name__ == "__main__":
	FPLMTD_runtime = FPLMTDLite_runtime = DOBSS_runtime = RANDOM_runtime = RobustRL_runtime = 0
	EXP3_runtime = BSSQ_runtime = PaddedExp3_runtime = SwitchingExp3_runtime = FPLGR_runtime = 0

	# get switching costs, utilities, and vulnerabilities
	sc = parse_switching()
	parse_attacks()
	def_util, att_util = parse_util()
	vulset = parse_vulset()

	# attacker type probability
	Pvec = [1/NUMTYPES for i in range(NUMTYPES)]

	game_def_util, game_att_util = parse_game_utils(def_util, att_util, vulset)

	DOBSS_mixed_strat_list = []

	start = time.time()
	# get defender mixed strategies when system is in config c
	for c in range(NUMCONFIGS):
		DOBSS_mixed_strat_list.append(getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, c))
	# initialising DOBSS strategy
	DOBSS_mixed_strat_list.append(getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec))
	end = time.time()
	DOBSS_runtime = end - start

	FPLMTD_switch = FPLMTDLite_switch = DOBSS_switch = RANDOM_switch = RobustRL_switch = 0
	EXP3_switch = BSSQ_switch = PaddedExp3_switch = SwitchingExp3_switch = FPLGR_switch = 0

	utility = np.array([[0.0]*T for i in range(NUMSTRATS)])

	for iter1 in range(MAX_ITER):
		FPLMTD_rhat = np.array([[0.0]*NUMATTACKS for i in range(NUMTYPES)])
		FPLMTDLite_rhat = np.array([0.0]*NUMCONFIGS)

		FPLGR_rhat = np.array([0.0]*NUMCONFIGS)

		Mixed_Strat = [[0.0]*NUMATTACKS for i in range(NUMSTRATS)]

		EXP3_p = [1/NUMCONFIGS]*NUMCONFIGS
		EXP3_L = [0.0]*NUMCONFIGS

		PaddedExp3_p = [1/NUMCONFIGS]*NUMCONFIGS
		PaddedExp3_L = [0.0]*NUMCONFIGS

		SwitchingExp3_p = [1/NUMCONFIGS]*NUMCONFIGS
		SwitchingExp3_L = [0.0]*NUMCONFIGS
		SwitchingExp3_util = 0.0

		RobustRL_maxvalue = [-10000]*NUMCONFIGS
		strat_old = [-1]*NUMSTRATS
		#DOBSS_mixed_strat = getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec)
		strat = [0]*NUMSTRATS
		DOBSS_mixed_strat = DOBSS_mixed_strat_list[-1]

		# get BSSQ x value
		num_episodes = 10

		start = time.time()
		BSSQ_mixed_strat_list = getBSSQStrat(game_def_util, game_att_util, sc, [1/NUMCONFIGS]*NUMCONFIGS, Pvec, num_episodes)
		end = time.time()
		# print(BSSQ_mixed_strat_list)	
		BSSQ_runtime += (end - start)

		BSSQ_mixed_strat = BSSQ_mixed_strat_list[-1]

		strat[RobustRL] = int(np.random.random()*NUMCONFIGS)

		for t in range(T):
			print(str(iter1)+":"+str(t) + "\t", end = "\r")
			
			# get strategies (configs) from each method
			strat[DOBSS] = getStratFromDist(DOBSS_mixed_strat)
			strat[RANDOM] = int(np.random.random()*NUMCONFIGS)

			start = time.time()
			strat[FPLMTD] = getFPLMTDStrat(FPLMTD_rhat, sc, strat_old[FPLMTD], vulset, Pvec, t)
			end = time.time()
			FPLMTD_runtime += (end - start)

			start = time.time()
			strat[FPLMTDLite] = getFPLMTDLiteStrat(FPLMTDLite_rhat, sc, strat_old[FPLMTDLite], t)
			end = time.time()
			FPLMTDLite_runtime += (end - start)

			start = time.time()
			strat[EXP3] = getEXP3Strat(EXP3_p)
			end = time.time()
			EXP3_runtime += (end - start)

			strat[PaddedExp3] = getPaddedExp3Strat(PaddedExp3_p)

			if(t%2 == 0):
				strat[SwitchingExp3] = getEXP3Strat(SwitchingExp3_p)

			strat[FPLGR] = getFPLGRStrat(FPLGR_rhat)

			# get mixed strategy for BSSQ
			strat[BSSQ] = getStratFromDist(BSSQ_mixed_strat)

			if(strat[FPLMTD] != strat_old[FPLMTD]):
				FPLMTD_switch += 1
			if(strat[FPLMTDLite] != strat_old[FPLMTDLite]):
				FPLMTDLite_switch += 1
			if(strat[DOBSS] != strat_old[DOBSS]):
				DOBSS_switch += 1 
			if(strat[RANDOM] != strat_old[RANDOM]):
				RANDOM_switch += 1 
			if(strat[RobustRL] != strat_old[RobustRL]):
				RobustRL_switch += 1 
			if(strat[EXP3] != strat_old[EXP3]):
				EXP3_switch += 1
			if(strat[BSSQ] != strat_old[BSSQ]):
				BSSQ_switch += 1 
			if(strat[PaddedExp3] != strat_old[PaddedExp3]):
				PaddedExp3_switch += 1 
			if(strat[SwitchingExp3] != strat_old[SwitchingExp3]):
				SwitchingExp3_switch += 1 
			if(strat[FPLGR] != strat_old[FPLGR]):
				FPLGR_switch += 1 

			# calculate ultilities using strategy from each method by simulating attack
			util, typ, attack, scosts = [0.0]*NUMSTRATS, [0]*NUMSTRATS, [0]*NUMSTRATS, [0.0]*NUMSTRATS
			for i in range(NUMSTRATS):
				util[i], typ[i], attack[i], Mixed_Strat[i] = getAttackBestResponse(def_util, att_util, strat[i], Pvec, vulset, Mixed_Strat[i], t)
				
				if(strat_old[i]!=-1):
					scosts[i] = sc[strat_old[i], strat[i]]
				utility[i, t] += (util[i] - scosts[i])


			#print(util[0])
			#DOBSS_mixed_strat = getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, strat[DOBSS])
			DOBSS_mixed_strat = DOBSS_mixed_strat_list[strat[DOBSS]]

			BSSQ_mixed_strat = BSSQ_mixed_strat_list[strat[BSSQ]]


			# Reward estimates using Geometric Resampling
			start = time.time()
			FPLMTD_rhat = FPLMTD_GR(FPLMTD_rhat, strat_old[FPLMTD], strat[FPLMTD], vulset, Pvec, util[FPLMTD], attack[FPLMTD], typ[FPLMTD], sc, t)
			end = time.time()
			FPLMTD_runtime += (end - start)

			start = time.time()
			FPLMTDLite_rhat = FPLMTDLite_GR(FPLMTDLite_rhat, strat_old[FPLMTDLite], strat[FPLMTDLite], util[FPLMTDLite], sc, t)
			end = time.time()
			FPLMTDLite_runtime += (end - start)

			FPLGR_rhat = FPL_GR(FPLGR_rhat, strat[FPLGR], util[FPLGR])


			# EXP3 update using utilities and last EXP3_p
			start = time.time()
			EXP3_L[strat[EXP3]] += (util[EXP3] - scosts[EXP3])/EXP3_p[strat[EXP3]]
			temp = np.sum([np.exp(EXP_ETA*EXP3_L[c]) for c in range(NUMCONFIGS)])
			for c in range(NUMCONFIGS):
				EXP3_p[c] = np.exp(EXP_ETA*EXP3_L[c])/temp 
			end = time.time()
			EXP3_runtime += (end - start)

			# Switching Exp3 update rule
			start = time.time()
			SwitchingExp3_util += util[SwitchingExp3] - scosts[SwitchingExp3]
			if(t%2 == 1):
				SwitchingExp3_L[strat[SwitchingExp3]] += (SwitchingExp3_util)/SwitchingExp3_p[strat[SwitchingExp3]]
				temp = np.sum([np.exp(EXP_ETA*SwitchingExp3_L[c]) for c in range(NUMCONFIGS)])
				for c in range(NUMCONFIGS):
					SwitchingExp3_p[c] = np.exp(EXP_ETA*SwitchingExp3_L[c])/temp 
				SwitchingExp3_util = 0.0
			end = time.time()
			SwitchingExp3_runtime += (end - start)

			# PaddedExp3 Update Rule
			start = time.time()
			updated_p = [0.0]*NUMCONFIGS
			for c in range(NUMCONFIGS):
				updated_p[c] += PADDING*PaddedExp3_p[c]
			updated_p[strat_old[PaddedExp3]] += (1-PADDING)
			PaddedExp3_L[strat[PaddedExp3]] += (util[PaddedExp3] - scosts[PaddedExp3])/updated_p[strat[PaddedExp3]]
			temp = np.sum([np.exp(EXP_ETA*PaddedExp3_L[c]) for c in range(NUMCONFIGS)])
			for c in range(NUMCONFIGS):
				PaddedExp3_p[c] = np.exp(EXP_ETA*PaddedExp3_L[c])/temp
			end = time.time() 
			PaddedExp3_runtime += (end - start)

			for i in range(NUMSTRATS):
				strat_old[i] = strat[i]

			start = time.time()
			RobustRL_maxvalue[strat[RobustRL]] = max(util[RobustRL] - scosts[RobustRL], RobustRL_maxvalue[strat[RobustRL]])
			strat[RobustRL] = np.argmin(RobustRL_maxvalue)
			end = time.time()
			RobustRL_runtime += (end - start)

	print("FPLMTD_Switch = ", FPLMTD_switch/MAX_ITER)
	print("FPLMTDLite_Switch = ", FPLMTDLite_switch/MAX_ITER)
	print("DOBSS_Switch = ", DOBSS_switch/MAX_ITER)
	print("RANDOM_Switch = ", RANDOM_switch/MAX_ITER)
	print("RobustRL_Switch = ", RobustRL_switch/MAX_ITER)
	print("EXP3_Switch = ", EXP3_switch/MAX_ITER)
	print("BSSQ_Switch = ", BSSQ_switch/MAX_ITER)
	print("PaddedExp3_Switch = ", PaddedExp3_switch/MAX_ITER)
	print("SwitchingExp3_Switch = ", SwitchingExp3_switch/MAX_ITER)
	print("FPLGR_Switch = ", FPLGR_switch/MAX_ITER)
	print("\n")

	print("FPLMTD Run-time = ", FPLMTD_runtime/MAX_ITER)
	print("FPLMTDLite Run-time = ", FPLMTDLite_runtime/MAX_ITER)
	print("DOBSS Run-time = ", DOBSS_runtime/MAX_ITER)
	print("RANDOM Run-time = ", RANDOM_runtime/MAX_ITER)
	print("RobustRL Run-time = ", RobustRL_runtime/MAX_ITER)
	print("EXP3 Run-time = ", EXP3_runtime/MAX_ITER)
	print("BSSQ Run-time = ", BSSQ_runtime/MAX_ITER)
	print("PaddedExp3 Run-time = ", PaddedExp3_runtime/MAX_ITER)
	print("SwitchingExp3 Run-time = ", SwitchingExp3_runtime/MAX_ITER)
	print("FPLGR Run-time = ", FPLGR_runtime/MAX_ITER)
	print("\n")

	# FPLMTD, FPLMTDLite, DOBSS, RANDOM, RobustRL, EXP3, BSSQ sum of utilities
	f_out = open(str(NUMCONFIGS) + "output_BestResponse.txt", "w")
	for i in range(NUMSTRATS):
		# print(np.sum(utility[i, :])/MAX_ITER)
		print(strats_list[i]+" Utility = " + str(np.sum(utility[i, :])/MAX_ITER))
		for t in range(T):
			f_out.write(str(utility[i, t]/MAX_ITER) + " ")
		f_out.write("\n")
	print("\n")








