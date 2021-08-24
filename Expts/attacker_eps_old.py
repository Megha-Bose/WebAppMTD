import sys
import numpy as np
import time

from BSSQ import *
from DOBSS import *

IN_DIR = "../Data/input/"
OUT_DIR = "../Data/output/"

NUMTYPES = 3 # Attacker Types
NUMATTACKS = 292 # Max no. of attacks
NUMCONFIGS = 4
MAX_ITER = 10
T = 1000
GAMMA = 0.5 # Exploration Parameter
EPSILON = 0.1
ALPHA = 0.1
BSSQ_ALPHA = 0.1
DISCOUNT_FACTOR = 0.8
M = 1000000
Lmax = 1000
PADDING = 1/(T**(1/3))

FPL_ETA = np.sqrt(np.log(NUMCONFIGS)/(NUMCONFIGS*T)) # FPL Hyperparameter
EXP_ETA = np.sqrt(2*np.log(NUMCONFIGS)/(NUMCONFIGS*T)) # EXP Hyperparameter


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
def parse_util(dataset_num):
	global NUMTYPES
	def_util = []
	att_util = []
	f = open(IN_DIR + str(dataset_num) + "utilities.txt", "r")
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
def parse_attacks(dataset_num):
	global NUMATTACKS
	f = open(IN_DIR + str(dataset_num) + "attacks.txt", "r")
	y = f.readline()
	NUMATTACKS = int(y)
	f.close()

# returns 0-1 vulnerabilities 2D matrix for (config, attack)
def parse_vulset(dataset_num):
	global NUMCONFIGS, NUMATTACKS
	vul_set = []
	f = open(IN_DIR + str(dataset_num) + "vulnerabilities.txt", "r")
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
def parse_switching(dataset_num):
	global NUMCONFIGS
	sc = []
	f = open(IN_DIR + str(dataset_num) + "switching.txt", "r")

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


# returns FPLMTD strategy
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


def getAttackEps(def_util, att_util, strat, P, eps, vulset):
	y = np.random.random()
	tau = NUMTYPES -1
	for i in range(NUMTYPES):
		if(y < P[i]):
			tau = i
			break
		else:
			y -= P[i]
	att = [att_util[tau, i]*vulset[strat, i] for i in range(NUMATTACKS)]
	max_util = np.max(att)
	att_list = []
	for a in range(NUMATTACKS):
		if(att[a] >= eps*max_util):
			att_list.append(a)
	random_ind = int(np.random.random()*len(att_list))
	attack = att_list[random_ind]
	util = 0
	if(vulset[strat, attack] == 1):
		util = def_util[tau, attack]

	return util, tau, attack


epsvec = [0.5, 0.6, 0.7, 0.8, 0.9]
Pvec = [0.15, 0.35, 0.5]

if __name__ == "__main__":
	for dataset_num in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
		print("Dataset: " + str(dataset_num))
		FPLMTD_runtime = FPLMTDLite_runtime = DOBSS_runtime = RANDOM_runtime = RobustRL_runtime = 0
		EXP3_runtime = BSSQ_runtime = PaddedExp3_runtime = SwitchingExp3_runtime = FPLGR_runtime = 0

		# get switching costs, utilities, and vulnerabilities
		sc = parse_switching(dataset_num)
		parse_attacks(dataset_num)
		def_util, att_util = parse_util(dataset_num)
		vulset = parse_vulset(dataset_num)

		# attacker type probability
		Pvec = [1/NUMTYPES for i in range(NUMTYPES)]

		game_def_util, game_att_util = parse_game_utils(def_util, att_util, vulset)

		DOBSS_mixed_strat_list = []
		start = time.time()
		# get defender mixed strategies when system is in config c
		for c in range(NUMCONFIGS):
			DOBSS_mixed_strat_list.append(getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, c, NUMCONFIGS, NUMATTACKS, NUMTYPES, M))
		# initialising DOBSS strategy
		DOBSS_mixed_strat_list.append(getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, M))
		end = time.time()
		DOBSS_runtime = end - start
		
		for eps in epsvec:
			print("Eps: " + str(eps))
			FPLMTD_switch = FPLMTDLite_switch = DOBSS_switch = RANDOM_switch = RobustRL_switch = 0
			EXP3_switch = BSSQ_switch = PaddedExp3_switch = SwitchingExp3_switch = FPLGR_switch = 0

			utility = np.array([[0.0]*T for i in range(NUMSTRATS)])

			for iter in range(MAX_ITER):
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
				#DOBSS_mixed_strat = getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, M)
				strat = [0]*NUMSTRATS
				DOBSS_mixed_strat = DOBSS_mixed_strat_list[-1]

				# get BSSQ x value
				start = time.time()
				BSSQ_mixed_strat_list = getBSSQStrat(game_def_util, game_att_util, sc, [1/NUMCONFIGS]*NUMCONFIGS, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, BSSQ_ALPHA, DISCOUNT_FACTOR, M)
				end = time.time()
				# print(BSSQ_mixed_strat_list)	
				BSSQ_runtime += (end - start)

				BSSQ_mixed_strat = BSSQ_mixed_strat_list[-1]

				start = time.time()
				strat[RobustRL] = int(np.random.random()*NUMCONFIGS)
				end = time.time()
				RobustRL_runtime += (end - start)
				
				for t in range(T):
					print(str(iter)+":"+str(t) + " "*10, end = "\r")
				
					# get strategies (configs) from each method
					start = time.time()
					strat[DOBSS] = getStratFromDist(DOBSS_mixed_strat)
					end = time.time()
					DOBSS_runtime += (end - start)

					start = time.time()
					strat[RANDOM] = int(np.random.random()*NUMCONFIGS)
					end = time.time()
					RANDOM_runtime += (end - start)

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

					start = time.time()
					strat[PaddedExp3] = getPaddedExp3Strat(PaddedExp3_p)
					end = time.time()
					PaddedExp3_runtime += (end - start)

					if(t%2 == 0):
						start = time.time()
						strat[SwitchingExp3] = getEXP3Strat(SwitchingExp3_p)
						end = time.time()
						SwitchingExp3_runtime += (end - start)

					start = time.time()
					strat[FPLGR] = getFPLGRStrat(FPLGR_rhat)
					end = time.time()
					FPLMTD_runtime += (end - start)

					# get mixed strategy for BSSQ
					start = time.time()
					strat[BSSQ] = getStratFromDist(BSSQ_mixed_strat)			
					end = time.time()
					BSSQ_runtime += (end - start)

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
						util[i], typ[i], attack[i] = getAttackEps(def_util, att_util, strat[i], Pvec, eps, vulset)
						
						if(strat_old[i]!=-1):
							scosts[i] = sc[strat_old[i], strat[i]]
						utility[i, t] += (util[i] - scosts[i])

					#print(util[0])
					#DOBSS_mixed_strat = getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, strat[DOBSS], NUMCONFIGS, NUMATTACKS, NUMTYPES, M)
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

					start = time.time()
					FPLGR_rhat = FPL_GR(FPLGR_rhat, strat[FPLGR], util[FPLGR])
					end = time.time()
					FPLGR_runtime += (end - start)


					# EXP3 update using utilities and last EXP3_p
					start = time.time()
					EXP3_L[strat[EXP3]] += (util[EXP3] - scosts[EXP3])/EXP3_p[strat[EXP3]]
					temp = np.sum([np.exp(EXP_ETA * EXP3_L[c]) for c in range(NUMCONFIGS)])
					for c in range(NUMCONFIGS):
						EXP3_p[c] = np.exp(EXP_ETA * EXP3_L[c]) / temp 
					end = time.time()
					EXP3_runtime += (end - start)

					# Switching Exp3 update rule
					start = time.time()
					SwitchingExp3_util += util[SwitchingExp3] - scosts[SwitchingExp3]
					if(t % 2 == 1):
						SwitchingExp3_L[strat[SwitchingExp3]] += (SwitchingExp3_util) / SwitchingExp3_p[strat[SwitchingExp3]]
						temp = np.sum([np.exp(EXP_ETA*SwitchingExp3_L[c]) for c in range(NUMCONFIGS)])
						for c in range(NUMCONFIGS):
							SwitchingExp3_p[c] = np.exp(EXP_ETA * SwitchingExp3_L[c])/temp 
						SwitchingExp3_util = 0.0
					end = time.time()
					SwitchingExp3_runtime += (end - start)

					# PaddedExp3 Update Rule
					start = time.time()
					updated_p = [0.0] * NUMCONFIGS
					for c in range(NUMCONFIGS):
						updated_p[c] += (PADDING * PaddedExp3_p[c])
					updated_p[strat_old[PaddedExp3]] += (1 - PADDING)
					PaddedExp3_L[strat[PaddedExp3]] += (util[PaddedExp3] - scosts[PaddedExp3]) / updated_p[strat[PaddedExp3]]
					temp = np.sum([np.exp(EXP_ETA * PaddedExp3_L[c]) for c in range(NUMCONFIGS)])
					for c in range(NUMCONFIGS):
						PaddedExp3_p[c] = np.exp(EXP_ETA*PaddedExp3_L[c]) / temp
					end = time.time() 
					PaddedExp3_runtime += (end - start)

					for i in range(NUMSTRATS):
						strat_old[i] = strat[i]

					start = time.time()
					RobustRL_maxvalue[strat[RobustRL]] = max(util[RobustRL] - scosts[RobustRL], RobustRL_maxvalue[strat[RobustRL]])
					strat[RobustRL] = np.argmin(RobustRL_maxvalue)
					end = time.time()
					RobustRL_runtime += (end - start)

				print("Iteration " + str(iter+1) + " over.")

			print("\n")
			stdout = sys.stdout
			f_ov_out = open(OUT_DIR + str(dataset_num) + "overall_out_eps_"+str(int(eps*100))+".txt", 'w')
			sys.stdout = f_ov_out

			print(NUMCONFIGS)
			print(NUMATTACKS)
			print(NUMTYPES)

			print("\n")
			print("Average switches per iteration:")
			print(FPLMTD_switch/MAX_ITER)
			print(FPLMTDLite_switch/MAX_ITER)
			print(DOBSS_switch/MAX_ITER)
			print(RANDOM_switch/MAX_ITER)
			print(RobustRL_switch/MAX_ITER)
			print(EXP3_switch/MAX_ITER)
			print(BSSQ_switch/MAX_ITER)
			print(PaddedExp3_switch/MAX_ITER)
			print(SwitchingExp3_switch/MAX_ITER)
			print(FPLGR_switch/MAX_ITER)
			print("\n")

			print("Average run-times per iteration:")
			print(FPLMTD_runtime/MAX_ITER)
			print(FPLMTDLite_runtime/MAX_ITER)
			print(DOBSS_runtime/MAX_ITER)
			print(RANDOM_runtime/MAX_ITER)
			print(RobustRL_runtime/MAX_ITER)
			print(EXP3_runtime/MAX_ITER)
			print(BSSQ_runtime/MAX_ITER)
			print(PaddedExp3_runtime/MAX_ITER)
			print(SwitchingExp3_runtime/MAX_ITER)
			print(FPLGR_runtime/MAX_ITER)
			print("\n")

			print("Average utilities per iteration:")
			# FPLMTD, FPLMTDLite, DOBSS, RANDOM, RobustRL, EXP3, BSSQ sum of utilities
			f_out = open(OUT_DIR + str(dataset_num) + "output_eps_"+str(int(eps*100))+".txt", "a")
			for i in range(NUMSTRATS):
				# print(np.sum(utility[i, :])/MAX_ITER)
				print(str(np.sum(utility[i, :])/MAX_ITER))
				for t in range(T):
					f_out.write(str(utility[i, t]/MAX_ITER) + " ")
				f_out.write("\n")
			print("\n")

			sys.stdout = stdout
			f_ov_out.close()
			f_out.close()








