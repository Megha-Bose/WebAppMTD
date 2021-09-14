import sys
import numpy as np
import time

from BSSQ import *
from DOBSS import *

IN_DIR = "../Data/input/"
OUT_DIR = "../Data/output/"

SEED = 2021
MAX_ITER = 5
T = 500

EPSILON = 0.1
M = 1000000
Lmax = 1000

NUMSTRATS = 2
FPLMaxMin = 0
FPLMTD = 1

# returns defender and attacker utilities
def parse_util(dataset_num):
	global NUMTYPES, IN_DIR
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
	global NUMATTACKS, IN_DIR
	f = open(IN_DIR + str(dataset_num) + "attacks.txt", "r")
	y = f.readline()
	NUMATTACKS = int(y)
	f.close()

# returns 0-1 vulnerabilities 2D matrix for (config, attack)
def parse_vulset(dataset_num):
	global NUMCONFIGS, NUMATTACKS, IN_DIR
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
	global NUMCONFIGS, IN_DIR
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
def getStratFromDist(x, rng):
	y = rng.random()
	for i in range(len(x)):
		if(y <= x[i]):
			return i
		y -= x[i]
	return len(x) - 1

 
# returns FPL strategy
def getFPLMaxMinStrat(r, n_vec, s, old_strat, vulset, P, t, rng, gamma, eta):
	# exploration
	gamma1 = rng.random()
	if(gamma1 <= gamma):
		return int(rng.random()*NUMCONFIGS)

	# reward estimates
	rhat = r.copy()
	# switching costs
	shat = s.copy()

	for a in range(NUMATTACKS):
		if(n_vec[a]!=0):
			rhat[:, a] = rhat[:, a]/n_vec[a]

	

	# adding perturbation
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			rhat[tau, a] = rhat[tau, a] - rng.exponential(eta)

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

	# net reward
	new_u = [u[c] - shat[old_strat, c] for c in range(NUMCONFIGS)]
	# return the best / leader strategy
	return np.argmax(new_u)


# returns FPLMTD strategy
def getFPLMTDStrat(r, s, old_strat, t, rng, gamma, eta):
	# exploration
	gamma1 = rng.random()
	if(gamma1 <= gamma):
		# print("Random")
		return int(rng.random()*NUMCONFIGS)
	rhat = r.copy()
	shat = s.copy()

	if(t != 0):
		rhat = rhat/t
	# adding perturbation
	for c in range(NUMCONFIGS):
		rhat[c] -= rng.exponential(eta)
	# net reward
	new_u = [rhat[c] - shat[old_strat, c] for c in range(NUMCONFIGS)]
	# return the best / leader strategy
	# print(new_u)
	# print(np.argmax(new_u))
	return np.argmax(new_u)

# update reward estimates using GR for FPL
def FPLMaxMin_MSE(r, attack_vec, type_vec, util_vec, P, t):
	rhat = np.copy(r)
	# print(type_vec[-1])
	last_type = type_vec[-1]
	p_vec = np.array([0.0]*NUMATTACKS)
	count = 0
	for tprime in range(t+1):
		if(type_vec[tprime] == last_type):
			p_vec[attack_vec[tprime]] += 1
			count += 1
	p_vec = P[last_type]*p_vec/count

	rhat[last_type, :] = 0.0
	for tprime in range(t+1):
		if(type_vec[tprime] == last_type):
			rhat[last_type, attack_vec[tprime]] += util_vec[tprime]/p_vec[attack_vec[tprime]]

	return rhat

# update reward estimates using GR for FPL lite
def FPLMTD_GR(r, old_strat, strat, util, switch_costs, t, rng, gamma, eta):
	rhat = np.copy(r)
	l = 1
	while(l < Lmax):
		strat2 = getFPLMTDStrat(rhat, switch_costs, old_strat, t, rng, gamma, eta)
		if(strat2 == strat):
			break
		l+=1
	rhat[strat] += util*l
	return rhat


def getAttackBestResponse(def_util, att_util, strat, P, vulset, Mixed_Strat, t, rng):
	y = rng.random()
	tau = NUMTYPES - 1
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

if __name__ == "__main__":
	
	zero_sum_flag = 0
	if len(sys.argv) > 3 and sys.argv[3] == '0':
		zero_sum_flag = 1

	IN_DIR = "../Data/input/"
	OUT_DIR = "../Data/output/"
	case = ''
	if zero_sum_flag == 1:
		case = 'zero_sum/'
	else:
		case = 'general_sum/'
	IN_DIR = IN_DIR + case
	OUT_DIR = OUT_DIR + case

	

	MTD_gamma = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02]
	MTD_eta = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]

	hyper_value_mtd = np.array([0.0]*(len(MTD_gamma)*len(MTD_eta)))
	hyper_value_maxmin = np.array([0.0]*(len(MTD_gamma)*len(MTD_eta)))

	for hyper_iter in range(len(MTD_gamma)*len(MTD_eta)):
		print("Hyperparameter Combination "+str(hyper_iter))
		gamma = MTD_gamma[int(hyper_iter/len(MTD_eta))]
		eta = MTD_eta[int(hyper_iter % len(MTD_eta))]

		for dataset_num in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
			# print("Dataset: " + str(dataset_num))

			# seeding random number generator  for reproducability
			rng = np.random.default_rng(SEED)

			FPLMTD_runtime, FPLMaxMin_runtime = 0, 0

			# get switching costs, utilities, and vulnerabilities
			sc = parse_switching(dataset_num)
			parse_attacks(dataset_num)
			def_util, att_util = parse_util(dataset_num)
			vulset = parse_vulset(dataset_num)
			for i in range(NUMCONFIGS):
				sc[i, i] = 0

			# attacker type probability
			Pvec = []
			psum = 0.0
			for tau in range(NUMTYPES):
				Pvec.append(rng.random())
				psum += Pvec[tau]
			for tau in range(NUMTYPES):
				Pvec[tau] = Pvec[tau] / psum

			# Pvec = [1/NUMTYPES for i in range(NUMTYPES)]

			game_def_util, game_att_util = parse_game_utils(def_util, att_util, vulset)
			FPLMTD_switch, FPLMaxMin_switch = 0, 0

			utility = np.array([[0.0]*T for i in range(NUMSTRATS)])


			for iter in range(MAX_ITER):
				FPLMaxMin_rhat = np.array([[0.0]*NUMATTACKS for i in range(NUMTYPES)])
				FPLMaxMin_n = [0]*NUMATTACKS
				FPLMaxMin_attack, FPLMaxMin_type, FPLMaxMin_util = [], [], []

				FPLMTD_rhat = np.array([0.0]*NUMCONFIGS)

				Mixed_Strat = [[0.0]*NUMATTACKS for i in range(NUMSTRATS)]

				strat_old = [-1]*NUMSTRATS
				strat = [0]*NUMSTRATS

				for t in range(T):
					# print(str(iter)+":"+str(t) + " "*10, end = "\r")

					start = time.time()
					strat[FPLMaxMin] = getFPLMaxMinStrat(FPLMaxMin_rhat, FPLMaxMin_n, sc, strat_old[FPLMaxMin], vulset, Pvec, t, rng, gamma, eta)
					end = time.time()
					FPLMaxMin_runtime += (end - start)
					# print(strat[FPLMaxMin])
					start = time.time()
					strat[FPLMTD] = getFPLMTDStrat(FPLMTD_rhat, sc, strat_old[FPLMTD], t, rng, gamma, eta)
					end = time.time()
					FPLMTD_runtime += (end - start)
					# print(strat[FPLMTD])
					if(strat[FPLMTD] != strat_old[FPLMTD]):
						FPLMTD_switch += 1
					if(strat[FPLMaxMin] != strat_old[FPLMaxMin]):
						FPLMaxMin_switch += 1

					# calculate ultilities using strategy from each method by simulating attack
					util, typ, attack, scosts = [0.0]*NUMSTRATS, [0]*NUMSTRATS, [0]*NUMSTRATS, [0.0]*NUMSTRATS
					for i in range(NUMSTRATS):
						util[i], typ[i], attack[i], Mixed_Strat[i] = getAttackBestResponse(def_util, att_util, strat[i], Pvec, vulset, Mixed_Strat[i], t, rng)
						
						if(strat_old[i]!=-1):
							scosts[i] = sc[strat_old[i], strat[i]]
						utility[i, t] += (util[i] - scosts[i])

					# print(typ[FPLMaxMin])
					# Updating reward estimates using Mixed Strategy Estimation
					start = time.time()
					FPLMaxMin_attack.append(attack[FPLMaxMin])
					FPLMaxMin_type.append(typ[FPLMaxMin])
					FPLMaxMin_util.append(util[FPLMaxMin])
					for a in range(NUMATTACKS):
						if(vulset[strat[FPLMaxMin], a]==1):
							FPLMaxMin_n[a] += 1
					FPLMaxMin_rhat = FPLMaxMin_MSE(FPLMaxMin_rhat, FPLMaxMin_attack, FPLMaxMin_type, FPLMaxMin_util, Pvec, t)
					end = time.time()
					FPLMaxMin_runtime += (end - start)

					# Updating reward estimates using Geometric Resampling
					start = time.time()
					FPLMTD_rhat = FPLMTD_GR(FPLMTD_rhat, strat_old[FPLMTD], strat[FPLMTD], util[FPLMTD], sc, t, rng, gamma, eta)
					end = time.time()
					FPLMTD_runtime += (end - start)

					for i in range(NUMSTRATS):
						strat_old[i] = strat[i]


			# print("\n")
			# print("Average switches per iteration:")
			# print(FPLMaxMin_switch/MAX_ITER)
			# print(FPLMTD_switch/MAX_ITER)
			# print("\n")

			# print("Average run-times per iteration:")
			# print(FPLMaxMin_runtime/MAX_ITER)
			# print(FPLMTD_runtime/MAX_ITER)
			# print("\n")

			# print("Average utilities per iteration:")
			# for i in range(NUMSTRATS):

			# 	print(str(np.sum(utility[i, :])/MAX_ITER))
			# print("\n")

			hyper_value_maxmin[hyper_iter] += np.sum(utility[FPLMaxMin, :])/MAX_ITER
			hyper_value_mtd[hyper_iter] += np.sum(utility[FPLMTD, :])/MAX_ITER


	print("Hyperparameter Selection for FPLMaxMin:")
	hyper_iter = np.argmax(hyper_value_maxmin)
	print("eta = " + str(MTD_eta[int(hyper_iter % len(MTD_eta))]))
	print("gamma = " + str(MTD_gamma[int(hyper_iter/len(MTD_eta))]))	

	print("Hyperparameter Selection for FPLMTD:")
	hyper_iter = np.argmax(hyper_value_mtd)
	print("eta = " + str(MTD_eta[int(hyper_iter % len(MTD_eta))]))
	print("gamma = " + str(MTD_gamma[int(hyper_iter/len(MTD_eta))]))


