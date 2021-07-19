import numpy as np 
from docplex.mp.model import Model
import gurobipy as gp

NUMTYPES = 3 # Attacker Types
NUMATTACKS = 292 # Max no. of attacks
NUMCONFIGS = 4
MAX_ITER = 10
T = 1000
GAMMA = 0.5 # Exploration Parameter
ETA = 0.25
EPSILON = 0.1
ALPHA = 0.1
DISCOUNT_FACTOR = 0.8
M = 1000000
Lmax = 1000

NUMSTRATS = 7
FPLMTD = 0
FPLMTDLite = 1
DOBSS = 2
RANDOM = 3
RobustRL = 4
EXP3 = 5

 
# returns defender and attacker utilities
def parse_util():
	def_util = []
	att_util = []
	f = open("utilities.txt", "r")
	y = f.readline()
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


# returns 0-1 vulnerabilities 2D matrix for (config, attack)
def parse_vulset():
	vul_set = []
	f = open("vulnerabilities.txt", "r")
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
	sc = []
	f = open("switching.txt", "r")
	for c in range(NUMCONFIGS):
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
			rhat[tau, a] -= np.random.exponential(ETA)

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
		rhat[c] -= np.random.exponential(ETA)

	if(old_strat != -1):
		rhat[old_strat] /= np.exp(-ALPHA)
	if(t!=0):
		rhat = rhat/(t)
	# net reward
	new_u = [rhat[c] - shat[old_strat, c] for c in range(NUMCONFIGS)]
	# return the best / leader strategy
	return np.argmax(new_u)

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
			rdash = r - np.random.exponential(ETA, NUMATTACKS)
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
Pvec = [0.15, 0.35, 0.5]

# get utilities, vulnerabilities and switching costs
def_util, att_util = parse_util()
vulset = parse_vulset()
sc = parse_switching()

game_def_util, game_att_util = parse_game_utils(def_util, att_util, vulset)


# get defender mixed strategies when system is in config c
DOBSS_mixed_strat_list = []
for c in range(NUMCONFIGS):
	DOBSS_mixed_strat_list.append(getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, c))
# initialising DOBSS strategy
DOBSS_mixed_strat_list.append(getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec))


FPLMTD_switch = 0
FPLMTDLite_switch = 0
DOBSS_switch = 0
utility = np.array([[0.0]*T for i in range(NUMSTRATS)])

for iter1 in range(MAX_ITER):
	FPLMTD_rhat = np.array([[0.0]*NUMATTACKS for i in range(NUMTYPES)])
	FPLMTDLite_rhat = np.array([0.0]*NUMCONFIGS)

	Mixed_Strat = [[0.0]*NUMATTACKS for i in range(NUMSTRATS)]

	EXP3_p = [1/NUMCONFIGS]*NUMCONFIGS
	EXP3_L = [0.0]*NUMCONFIGS

	RobustRL_maxvalue = [-10000]*NUMCONFIGS
	strat_old = [-1]*NUMSTRATS
	#DOBSS_mixed_strat = getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec)
	strat = [0]*NUMSTRATS
	DOBSS_mixed_strat = DOBSS_mixed_strat_list[-1]

	strat[RobustRL] = int(np.random.random()*NUMCONFIGS)

	for t in range(T):
		print(str(iter1)+":"+str(t) + "\t", end = "\r")
		
		# get strategies (configs) from each method
		strat[DOBSS] = getStratFromDist(DOBSS_mixed_strat)
		strat[RANDOM] = int(np.random.random()*NUMCONFIGS)
		strat[FPLMTD] = getFPLMTDStrat(FPLMTD_rhat, sc, strat_old[FPLMTD], vulset, Pvec, t)
		strat[FPLMTDLite] = getFPLMTDLiteStrat(FPLMTDLite_rhat, sc, strat_old[FPLMTDLite], t)
		strat[EXP3] = getEXP3Strat(EXP3_p)


		if(strat[FPLMTD] != strat_old[FPLMTD]):
			FPLMTD_switch += 1
		if(strat[FPLMTDLite] != strat_old[FPLMTDLite]):
			FPLMTDLite_switch += 1
		if(strat[DOBSS] != strat_old[DOBSS]):
			DOBSS_switch += 1 

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

		# Reward estimates using Geometric Resampling
		FPLMTD_rhat = FPLMTD_GR(FPLMTD_rhat, strat_old[FPLMTD], strat[FPLMTD], vulset, Pvec, util[FPLMTD], attack[FPLMTD], typ[FPLMTD], sc, t)
		FPLMTDLite_rhat = FPLMTDLite_GR(FPLMTDLite_rhat, strat_old[FPLMTDLite], strat[FPLMTDLite], util[FPLMTDLite], sc, t)

		# EXP3 update using utilities and last EXP3_p
		EXP3_L[strat[EXP3]] += (util[EXP3] - scosts[EXP3])/EXP3_p[strat[EXP3]]
		temp = np.sum([np.exp(ETA*EXP3_L[c]) for c in range(NUMCONFIGS)])
		for c in range(NUMCONFIGS):
			EXP3_p[c] = np.exp(ETA*EXP3_L[c])/temp 

		for i in range(NUMSTRATS):
			strat_old[i] = strat[i]

		RobustRL_maxvalue[strat[RobustRL]] = max(util[RobustRL] - scosts[RobustRL], RobustRL_maxvalue[strat[RobustRL]])
		strat[RobustRL] = np.argmin(RobustRL_maxvalue)

print("FPLMTD_Switch = ", FPLMTD_switch/MAX_ITER)
print("FPLMTDLite_Switch = ", FPLMTDLite_switch/MAX_ITER)
print("DOBSS_Switch = ", DOBSS_switch/MAX_ITER)
# print(EXP3_L)
f_out = open(str(NUMCONFIGS) + "output_BestResponse.txt", "w")
for i in range(NUMSTRATS):
	print(np.sum(utility[i, :])/MAX_ITER)
	for t in range(T):
		f_out.write(str(utility[i, t]/MAX_ITER) + " ")
	f_out.write("\n")
print("\n")








