import sys
import numpy as np
import time

from BSSQ_gurobi import *
from DOBSS_gurobi import *

IN_DIR = "../Data/input/"
OUT_DIR = "../Data/output/"

SEED = 2022
NUMTYPES = 1 
NUMATTACKS = 1 
NUMCONFIGS = 1
MAX_ITER = 50
T = 1000
MaxMin_GAMMA = 0.006 #MaxMin Hyper-parameters
MaxMin_ETA = 0.03
MTD_GAMMA = 0.007 #MTD Parameters
MTD_ETA = 0.1
EPSILON = 0.1
RobustRL_EPSILON = 0.01
BSSQ_ALPHA = 0.2
DISCOUNT_FACTOR = 0.8
M = 1000000
Lmax = 1000
PADDING = 1/(T**(1/3))


NUMSTRATS = 2
FPLMaxMin = 0
RANDOM = 1

# returns defender and attacker utilities
def parse_util(dataset_num):
	global NUMTYPES, IN_DIR
	def_util = []
	att_util = []
	f = open(IN_DIR + str(dataset_num) + "utilities.txt", "r")
	y = f.readline()
	NUMTYPES = int(y)
	p_type = None

	p = f.readline().split()
	p_type = [float(item) for item in p]

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
	return def_util, att_util, p_type

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
def getFPLMaxMinStrat(r, n_vec, s, old_strat, vulset, P, t, rng):
	# exploration
	gamma1 = rng.random()
	if(gamma1 <= MaxMin_GAMMA):
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
			rhat[tau, a] = rhat[tau, a] - rng.exponential(MaxMin_ETA)

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
def getFPLMTDStrat(r, s, old_strat, t, rng):
	# exploration
	gamma1 = rng.random()
	if(gamma1 <= MTD_GAMMA):
		# print("Random")
		return int(rng.random()*NUMCONFIGS)
	rhat = r.copy()
	shat = s.copy()

	if(t != 0):
		rhat = rhat/t
	# adding perturbation
	for c in range(NUMCONFIGS):
		rhat[c] -= rng.exponential(MTD_ETA)
	# net reward
	new_u = [rhat[c] - shat[old_strat, c] for c in range(NUMCONFIGS)]
	# return the best / leader strategy
	# print(new_u)
	# print(np.argmax(new_u))
	return np.argmax(new_u)

# returns FPL+GR strategy
def getFPLGRStrat(r, rng):
	rhat = r.copy()
	for c in range(NUMCONFIGS):
		rhat[c] -= (1/FPL_ETA)*rng.exponential(1)
	return np.argmax(rhat)

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
def FPLMTD_GR(r, old_strat, strat, util, switch_costs, t, rng):
	rhat = np.copy(r)
	l = 1
	while(l < Lmax):
		strat2 = getFPLMTDStrat(rhat, switch_costs, old_strat, t, rng)
		if(strat2 == strat):
			break
		l+=1
	rhat[strat] += util*l
	return rhat

def FPL_GR(r, strat, util, rng):
	rhat = np.copy(r)
	l = 1
	while(l < Lmax):
		strat2 = getFPLGRStrat(rhat, rng)
		if(strat2 == strat):
			break
		l+=1
	rhat[strat]+= util*l
	return rhat

# returns RobustRL strategy
def getRobustRLStrat(RobustRL_maxvalue, rng):
	maxval = np.max(RobustRL_maxvalue)
	notmax_array = []
	max_array = []
	for c in range(NUMCONFIGS):
		if(RobustRL_maxvalue[c] != maxval):
			notmax_array.append(c)
		else:
			max_array.append(c)
	if((rng.random() < EPSILON) & (len(notmax_array) > 0)):
		return notmax_array[int(rng.random()*len(notmax_array))]
	return max_array[int(rng.random()*len(max_array))]

# samples strategy sequentially using EXP3 from p values
def getEXP3Strat(p, rng):
	y = rng.random()
	for c in range(NUMCONFIGS):
		if(y < p[c]):
			return c
		y -= p[c]
	return NUMCONFIGS - 1

def getPaddedExp3Strat(p, rng, old):
	z = rng.random()
	if(z < PADDING):
		y = rng.random()
		for c in range(NUMCONFIGS):
			if(y < p[c]):
				return c
			y -= p[c]
		return NUMCONFIGS - 1
	return old


# using GR to update attacker reward estimates for FPL-UE
def Attacker_GR(rhat, vdash, util, rng):
	r = rhat.copy()
	i = 1
	l = Lmax # cap value
	while(i < Lmax):
		y = rng.random()
		if(y < EPSILON):
			# exploration
			v = int(rng.random()*NUMCONFIGS)
		else:
			# FPL
			rdash = r - rng.exponential(FPL_ETA, NUMATTACKS)
			v = np.argmax(rdash)
		if(vdash == v):
			l = i
			break
		i+=1
	r[vdash] += l*util
	return r


# getting attack that gives best attack utility
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
			if(vulset[c, a] == 1):
				u += att_util[tau, a]*Mixed_Strat[c]
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
		MS[c] = MS[c]*(t)/(t+1)
		if(c == strat):
			MS[c] += 1/(t+1)

	return util, tau, attack, MS

def computeCVec(r, vulset, P):
	c_vec = []
	for c in range(NUMCONFIGS):
		val = 0
		for tau in range(NUMTYPES):
			val += P[tau]*np.min([r[tau, a]*vulset[c, a] for a in range(NUMATTACKS)])
		c_vec.append(val)
	return c_vec

def fixGreedyVulnerabilities(r, vulset, k, P):
	vulset_new = np.copy(vulset)
	for iter1 in range(k):
		c_vec = computeCVec(r, vulset_new, P)
		cmin = np.argmin(c_vec)
		maxval = -1000000
		maxvul = 0
		for a in range(NUMATTACKS):
			if(np.sum(vulset_new[:, a]) == 0):
				continue
			vulset_prime = np.copy(vulset_new)
			vulset_prime[:, a] = 0
			temp = computeCVec(r, vulset_prime, P)
			val = temp[cmin] - c_vec[cmin]
			if(val > maxval):
				maxval = val
				maxvul = a
		# print(maxvul, vulset_new[:, maxvul])
		vulset_new[:, maxvul] = 0
	return vulset_new

def fixRandomVulnerabilities(k, vulset, rng):
	vulset_new = np.copy(vulset)
	X = rng.permutation(NUMATTACKS)
	count = 0
	for i in range(NUMATTACKS):
		if(np.sum(vulset_new[:, X[i]]) == 0):
			continue
		vulset_new[:, X[i]] = 0
		count += 1
		if(count == k):
			break
	return vulset_new


epsvec = [0.5, 0.6, 0.7, 0.8, 0.9]
Pvec = None

if __name__ == "__main__":
	
	zero_sum_flag = 0
	if len(sys.argv) > 3 and sys.argv[3] == '0':
		zero_sum_flag = 1

	K = int(sys.argv[3])
	# print(K)

	IN_DIR = "../Data/input/"
	OUT_DIR = "../Data/output/"
	case = ''
	if zero_sum_flag == 1:
		case = 'zero_sum/'
	else:
		case = 'general_sum/'
	IN_DIR = IN_DIR + case
	OUT_DIR = OUT_DIR + case

	for dataset_num in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
		# print("Dataset: " + str(dataset_num))

		# seeding random number generator  for reproducability
		rng = np.random.default_rng(SEED)

		# get switching costs, utilities, and vulnerabilities
		sc = parse_switching(dataset_num)
		parse_attacks(dataset_num)
		def_util, att_util, Pvec = parse_util(dataset_num)
		vulset = parse_vulset(dataset_num)

		for c in range(NUMCONFIGS):
			sc[c, c] = 0


		game_def_util, game_att_util = parse_game_utils(def_util, att_util, vulset)


		switch = [0]*NUMSTRATS

		utility = [np.array([[0.0]*T for i in range(MAX_ITER)]) for iter in range(NUMSTRATS)]
		utility_greedy = [np.array([[0.0]*T for i in range(MAX_ITER)]) for iter in range(NUMSTRATS)]
		utility_random = [np.array([[0.0]*T for i in range(MAX_ITER)]) for iter in range(NUMSTRATS)]
		utility_untouched = [np.array([[0.0]*T for i in range(MAX_ITER)]) for iter in range(NUMSTRATS)]

		for iter in range(MAX_ITER):
			# print(str(iter), end = "\r")
			FPLMaxMin_rhat = np.array([[0.0]*NUMATTACKS for i in range(NUMTYPES)])
			FPLMaxMin_n = [0]*NUMATTACKS
			FPLMaxMin_attack, FPLMaxMin_type, FPLMaxMin_util = [], [], []

			Mixed_Strat = [[0.0]*NUMCONFIGS for i in range(NUMSTRATS)]

			strat_old = [-1]*NUMSTRATS
			# DOBSS_mixed_strat = getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, M)
			strat = [0]*NUMSTRATS

			for t in range(T):
				print(str(iter)+":"+str(t) + " "*10, end = "\r")
				
				# get strategies (configs) from each method
				strat[FPLMaxMin] = getFPLMaxMinStrat(FPLMaxMin_rhat, FPLMaxMin_n, sc, strat_old[FPLMaxMin], vulset, Pvec, t, rng)

				strat[RANDOM] = int(rng.random()*NUMCONFIGS)

				for i in range(NUMSTRATS):
					if(strat[i] != strat_old[i]):
						switch[i]+=1

				# calculate ultilities using strategy from each method by simulating attack
				util, typ, attack, scosts = [0.0]*NUMSTRATS, [0]*NUMSTRATS, [0]*NUMSTRATS, [0.0]*NUMSTRATS
				for i in range(NUMSTRATS):
					util[i], typ[i], attack[i], Mixed_Strat[i] = getAttackBestResponse(def_util, att_util, strat[i], Pvec, vulset, Mixed_Strat[i], t, rng)
					
					if(strat_old[i]!=-1):
						scosts[i] = sc[strat_old[i], strat[i]]
					utility[i][iter, t] = (util[i] - scosts[i])

				# Reward estimates using Geometric Resampling
				FPLMaxMin_attack.append(attack[FPLMaxMin])
				FPLMaxMin_type.append(typ[FPLMaxMin])
				FPLMaxMin_util.append(util[FPLMaxMin])
				for a in range(NUMATTACKS):
					if(vulset[strat[FPLMaxMin], a]==1):
						FPLMaxMin_n[a] += 1
				FPLMaxMin_rhat = FPLMaxMin_MSE(FPLMaxMin_rhat, FPLMaxMin_attack, FPLMaxMin_type, FPLMaxMin_util, Pvec, t)				

				for i in range(NUMSTRATS):
					strat_old[i] = strat[i]

			reward_estimates = np.copy(FPLMaxMin_rhat)
			for tau in range(NUMTYPES):
				for a in range(NUMATTACKS):
					if(FPLMaxMin_n[a]!=0):
						reward_estimates[tau, a] = reward_estimates[tau, a]/FPLMaxMin_n[a]
			vulset_new = fixGreedyVulnerabilities(reward_estimates, vulset, K, Pvec)

			FPLMaxMin_rhat_old = np.copy(FPLMaxMin_rhat)
			FPLMaxMin_n_old = FPLMaxMin_n.copy()
			FPLMaxMin_attack_old = FPLMaxMin_attack.copy()
			FPLMaxMin_type_old = FPLMaxMin_type.copy()
			FPLMaxMin_util_old = FPLMaxMin_util.copy()

			Mixed_Strat_old = Mixed_Strat.copy()
			strat_old_old = strat_old.copy()

			for t in range(T):
				print(str(iter)+":"+str(t) + " "*10, end = "\r")
				
				# get strategies (configs) from each method
				strat[FPLMaxMin] = getFPLMaxMinStrat(FPLMaxMin_rhat, FPLMaxMin_n, sc, strat_old[FPLMaxMin], vulset_new, Pvec, T+t, rng)

				strat[RANDOM] = int(rng.random()*NUMCONFIGS)

				for i in range(NUMSTRATS):
					if(strat[i] != strat_old[i]):
						switch[i]+=1

				# calculate ultilities using strategy from each method by simulating attack
				util, typ, attack, scosts = [0.0]*NUMSTRATS, [0]*NUMSTRATS, [0]*NUMSTRATS, [0.0]*NUMSTRATS
				for i in range(NUMSTRATS):
					util[i], typ[i], attack[i], Mixed_Strat[i] = getAttackBestResponse(def_util, att_util, strat[i], Pvec, vulset_new, Mixed_Strat[i], T+t, rng)
					
					if(strat_old[i]!=-1):
						scosts[i] = sc[strat_old[i], strat[i]]
					utility_greedy[i][iter, t] = (util[i] - scosts[i])

				# Reward estimates using Geometric Resampling
				FPLMaxMin_attack.append(attack[FPLMaxMin])
				FPLMaxMin_type.append(typ[FPLMaxMin])
				FPLMaxMin_util.append(util[FPLMaxMin])
				for a in range(NUMATTACKS):
					if(vulset[strat[FPLMaxMin], a]==1):
						FPLMaxMin_n[a] += 1
				FPLMaxMin_rhat = FPLMaxMin_MSE(FPLMaxMin_rhat, FPLMaxMin_attack, FPLMaxMin_type, FPLMaxMin_util, Pvec, T+t)				

				for i in range(NUMSTRATS):
					strat_old[i] = strat[i]

			vulset_new = fixRandomVulnerabilities(K, vulset, rng)

			FPLMaxMin_rhat = np.copy(FPLMaxMin_rhat_old)
			FPLMaxMin_n = FPLMaxMin_n_old.copy()
			FPLMaxMin_attack, FPLMaxMin_type, FPLMaxMin_util = FPLMaxMin_attack_old.copy(), FPLMaxMin_type_old.copy(), FPLMaxMin_util_old.copy()

			Mixed_Strat = Mixed_Strat_old.copy()

			strat_old = strat_old_old.copy()
			# DOBSS_mixed_strat = getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, M)
			strat = [0]*NUMSTRATS

			for t in range(T):
				print(str(iter)+":"+str(t) + " "*10, end = "\r")
				
				# get strategies (configs) from each method
				strat[FPLMaxMin] = getFPLMaxMinStrat(FPLMaxMin_rhat, FPLMaxMin_n, sc, strat_old[FPLMaxMin], vulset_new, Pvec, T+t, rng)

				strat[RANDOM] = int(rng.random()*NUMCONFIGS)

				for i in range(NUMSTRATS):
					if(strat[i] != strat_old[i]):
						switch[i]+=1

				# calculate ultilities using strategy from each method by simulating attack
				util, typ, attack, scosts = [0.0]*NUMSTRATS, [0]*NUMSTRATS, [0]*NUMSTRATS, [0.0]*NUMSTRATS
				for i in range(NUMSTRATS):
					util[i], typ[i], attack[i], Mixed_Strat[i] = getAttackBestResponse(def_util, att_util, strat[i], Pvec, vulset_new, Mixed_Strat[i], T+t, rng)
					
					if(strat_old[i]!=-1):
						scosts[i] = sc[strat_old[i], strat[i]]
					utility_random[i][iter, t] = (util[i] - scosts[i])

				# Reward estimates using Geometric Resampling
				FPLMaxMin_attack.append(attack[FPLMaxMin])
				FPLMaxMin_type.append(typ[FPLMaxMin])
				FPLMaxMin_util.append(util[FPLMaxMin])
				for a in range(NUMATTACKS):
					if(vulset[strat[FPLMaxMin], a]==1):
						FPLMaxMin_n[a] += 1
				FPLMaxMin_rhat = FPLMaxMin_MSE(FPLMaxMin_rhat, FPLMaxMin_attack, FPLMaxMin_type, FPLMaxMin_util, Pvec, T+t)				

				for i in range(NUMSTRATS):
					strat_old[i] = strat[i]

			FPLMaxMin_rhat = np.copy(FPLMaxMin_rhat_old)
			FPLMaxMin_n = FPLMaxMin_n_old.copy()
			FPLMaxMin_attack, FPLMaxMin_type, FPLMaxMin_util = FPLMaxMin_attack_old.copy(), FPLMaxMin_type_old.copy(), FPLMaxMin_util_old.copy()

			Mixed_Strat = Mixed_Strat_old.copy()

			strat_old = strat_old_old.copy()
			# DOBSS_mixed_strat = getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, M)
			strat = [0]*NUMSTRATS

			for t in range(T):
				print(str(iter)+":"+str(t) + " "*10, end = "\r")
				
				# get strategies (configs) from each method
				strat[FPLMaxMin] = getFPLMaxMinStrat(FPLMaxMin_rhat, FPLMaxMin_n, sc, strat_old[FPLMaxMin], vulset, Pvec, T+t, rng)

				strat[RANDOM] = int(rng.random()*NUMCONFIGS)

				for i in range(NUMSTRATS):
					if(strat[i] != strat_old[i]):
						switch[i]+=1

				# calculate ultilities using strategy from each method by simulating attack
				util, typ, attack, scosts = [0.0]*NUMSTRATS, [0]*NUMSTRATS, [0]*NUMSTRATS, [0.0]*NUMSTRATS
				for i in range(NUMSTRATS):
					util[i], typ[i], attack[i], Mixed_Strat[i] = getAttackBestResponse(def_util, att_util, strat[i], Pvec, vulset, Mixed_Strat[i], T+t, rng)
					
					if(strat_old[i]!=-1):
						scosts[i] = sc[strat_old[i], strat[i]]
					utility_untouched[i][iter, t] = (util[i] - scosts[i])

				# Reward estimates using Geometric Resampling
				FPLMaxMin_attack.append(attack[FPLMaxMin])
				FPLMaxMin_type.append(typ[FPLMaxMin])
				FPLMaxMin_util.append(util[FPLMaxMin])
				for a in range(NUMATTACKS):
					if(vulset[strat[FPLMaxMin], a]==1):
						FPLMaxMin_n[a] += 1
				FPLMaxMin_rhat = FPLMaxMin_MSE(FPLMaxMin_rhat, FPLMaxMin_attack, FPLMaxMin_type, FPLMaxMin_util, Pvec, T+t)				

				for i in range(NUMSTRATS):
					strat_old[i] = strat[i]






			# print("Iteration " + str(iter+1) + " over.")

		# print("\n")

		greedy_vec = []
		random_vec = []
		average_vec = []
		for i in range(MAX_ITER):
			greedy_vec.append(np.sum(utility_greedy[0][i, :]) - np.sum(utility_untouched[0][i, :]))
			random_vec.append(np.sum(utility_random[0][i, :]) - np.sum(utility_untouched[0][i, :]))
			average_vec.append(np.sum(utility_greedy[0][i, :])/np.sum(utility_untouched[0][i, :]))

		print("Statistics for K = %f" % K)
		print("Greedy Improvement, mean : %.2f, std : %.2f " % (np.mean(greedy_vec), np.std(greedy_vec)/np.sqrt(MAX_ITER)))
		print("Random Improvement, mean : %.2f, std : %.2f " % (np.mean(random_vec), np.std(random_vec)/np.sqrt(MAX_ITER)))
		print("Greedy Untouched Ratio, mean : %.4f, std : %.4f " % (np.mean(average_vec), np.std(average_vec)/np.sqrt(MAX_ITER)))










