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
MAX_ITER = 10
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
lambda_QR = 0.75


NUMSTRATS = 11
FPLMaxMin = 0
FPLMTD = 1
DOBSS = 2
RANDOM = 3
RobustRL = 4
EXP3 = 5
BSSQ = 6
PaddedExp3 = 7
SwitchingExp3 = 8
FPLGR = 9
BiasedASLR = 10

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


# getting attack that according to QR
def getAttackQuantalResponse(def_util, att_util, strat, P, vulset, Mixed_Strat, t, rng):
	y = rng.random()
	tau = NUMTYPES - 1
	for i in range(NUMTYPES):
		if(y < P[i]):
			tau = i
			break
		else:
			y -= P[i]

	# get probabilities for attacks according to quantal response
	att_prob = [1/NUMATTACKS]*NUMATTACKS
	tot = 0
	for a in range(NUMATTACKS):
		att_prob[a] = 1
		for c in range(NUMCONFIGS):
			if(vulset[c, a] == 1):
				att_prob[a] = att_prob[a] * np.exp(lambda_QR * Mixed_Strat[c] * att_util[tau, c])
		# if att_prob[a] == 1:
		# 	att_prob[a] = 0
		tot += att_prob[a]
		
	if tot != 0:
		for a in range(NUMATTACKS):
			att_prob[a] = att_prob[a] / tot

	y = rng.random()
	attack = NUMATTACKS - 1
	for i in range(NUMATTACKS):
		if(y < att_prob[i]):
			attack = i
			break
		else:
			y -= att_prob[i]

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


epsvec = [0.5, 0.6, 0.7, 0.8, 0.9]
Pvec = None

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

	for dataset_num in range(int(sys.argv[1]), int(sys.argv[2]) + 1):
		# print("Dataset: " + str(dataset_num))

		# seeding random number generator  for reproducability
		rng = np.random.default_rng(SEED)

		FPLMaxMin_runtime = FPLMTD_runtime = DOBSS_runtime = RANDOM_runtime = RobustRL_runtime = 0
		EXP3_runtime = BSSQ_runtime = PaddedExp3_runtime = SwitchingExp3_runtime = FPLGR_runtime = 0
		BiasedASLR_runtime = 0

		# get switching costs, utilities, and vulnerabilities
		sc = parse_switching(dataset_num)
		parse_attacks(dataset_num)
		def_util, att_util, Pvec = parse_util(dataset_num)
		vulset = parse_vulset(dataset_num)

		for c in range(NUMCONFIGS):
			sc[c, c] = 0

		# Pvec = [1/NUMTYPES for i in range(NUMTYPES)]
		FPL_ETA = np.sqrt(np.log(NUMCONFIGS)/(NUMCONFIGS*T)) # FPL Hyperparameter
		EXP_ETA = np.sqrt(2*np.log(NUMCONFIGS)/(NUMCONFIGS*T)) # EXP Hyperparameter

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

		switch = [0]*NUMSTRATS

		utility = [np.array([[0.0]*T for i in range(MAX_ITER)]) for iter in range(NUMSTRATS)]

		for iter in range(MAX_ITER):
			# print(str(iter), end = "\r")
			FPLMaxMin_rhat = np.array([[0.0]*NUMATTACKS for i in range(NUMTYPES)])
			FPLMaxMin_n = [0]*NUMATTACKS
			FPLMaxMin_attack, FPLMaxMin_type, FPLMaxMin_util = [], [], []

			FPLMTD_rhat = np.array([0.0]*NUMCONFIGS)

			FPLGR_rhat = np.array([0.0]*NUMCONFIGS)

			Mixed_Strat = [[0.0]*NUMCONFIGS for i in range(NUMSTRATS)]

			EXP3_p = [1/NUMCONFIGS]*NUMCONFIGS
			EXP3_L = [0.0]*NUMCONFIGS

			PaddedExp3_p = [1/NUMCONFIGS]*NUMCONFIGS
			PaddedExp3_L = [0.0]*NUMCONFIGS

			SwitchingExp3_p = [1/NUMCONFIGS]*NUMCONFIGS
			SwitchingExp3_L = [0.0]*NUMCONFIGS
			SwitchingExp3_util = 0.0

			RobustRL_maxvalue = [T]*NUMCONFIGS

			config_hit_count = [1]*NUMCONFIGS

			strat_old = [-1]*NUMSTRATS
			# DOBSS_mixed_strat = getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, M)
			strat = [0]*NUMSTRATS
			DOBSS_mixed_strat = DOBSS_mixed_strat_list[-1]

			# get BSSQ x value
			start = time.time()
			BSSQ_mixed_strat_list = getBSSQStrat(game_def_util, game_att_util, sc, [1/NUMCONFIGS]*NUMCONFIGS, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, BSSQ_ALPHA, DISCOUNT_FACTOR, M, rng)
			end = time.time()
			# print(BSSQ_mixed_strat_list)	
			BSSQ_runtime += (end - start)

			BSSQ_mixed_strat = BSSQ_mixed_strat_list[-1]

			start = time.time()
			strat[RobustRL] = int(rng.random()*NUMCONFIGS)
			end = time.time()
			RobustRL_runtime += (end - start)

			for t in range(T):
				print(str(iter)+":"+str(t) + " "*10, end = "\r")
				
				# get strategies (configs) from each method
				start = time.time()
				strat[FPLMaxMin] = getFPLMaxMinStrat(FPLMaxMin_rhat, FPLMaxMin_n, sc, strat_old[FPLMaxMin], vulset, Pvec, t, rng)
				end = time.time()
				FPLMaxMin_runtime += (end - start)

				start = time.time()
				strat[FPLMTD] = getFPLMTDStrat(FPLMTD_rhat, sc, strat_old[FPLMTD], t, rng)
				end = time.time()
				FPLMTD_runtime += (end - start)

				start = time.time()
				strat[DOBSS] = getStratFromDist(DOBSS_mixed_strat, rng)
				end = time.time()
				DOBSS_runtime += (end - start)

				start = time.time()
				strat[RANDOM] = int(rng.random()*NUMCONFIGS)
				end = time.time()
				RANDOM_runtime += (end - start)

				start = time.time()
				strat[EXP3] = getEXP3Strat(EXP3_p, rng)
				end = time.time()
				EXP3_runtime += (end - start)

				start = time.time()
				strat[PaddedExp3] = getPaddedExp3Strat(PaddedExp3_p, rng, strat_old[PaddedExp3])
				end = time.time()
				PaddedExp3_runtime += (end - start)

				if(t%2 == 0):
					start = time.time()
					strat[SwitchingExp3] = getEXP3Strat(SwitchingExp3_p, rng)
					end = time.time()
					SwitchingExp3_runtime += (end - start)

				start = time.time()
				strat[FPLGR] = getFPLGRStrat(FPLGR_rhat, rng)
				end = time.time()
				FPLMTD_runtime += (end - start)

				# get mixed strategy for BSSQ
				start = time.time()
				strat[BSSQ] = getStratFromDist(BSSQ_mixed_strat, rng)			
				end = time.time()
				BSSQ_runtime += (end - start)

				# get strategy for BiasedASLR
				start = time.time()
				probs_config = [1/NUMCONFIGS]*NUMCONFIGS
				total_probs_config = 0
				for config in range(NUMCONFIGS):
					total_probs_config += (1.0/config_hit_count[config])
				for config in range(NUMCONFIGS):
					probs_config[config] = (1.0/config_hit_count[config])/total_probs_config
				strat[BiasedASLR] = getStratFromDist(probs_config, rng)
				end = time.time()
				BiasedASLR_runtime += (end - start)

				for i in range(NUMSTRATS):
					if(strat[i] != strat_old[i]):
						switch[i]+=1

				# calculate ultilities using strategy from each method by simulating attack
				util, typ, attack, scosts = [0.0]*NUMSTRATS, [0]*NUMSTRATS, [0]*NUMSTRATS, [0.0]*NUMSTRATS
				for i in range(NUMSTRATS):
					util[i], typ[i], attack[i], Mixed_Strat[i] = getAttackQuantalResponse(def_util, att_util, strat[i], Pvec, vulset, Mixed_Strat[i], t, rng)
					
					if(strat_old[i]!=-1):
						scosts[i] = sc[strat_old[i], strat[i]]
					utility[i][iter, t] = (util[i] - scosts[i])



				#print(util[0])
				#DOBSS_mixed_strat = getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, strat[DOBSS], NUMCONFIGS, NUMATTACKS, NUMTYPES, M)
				DOBSS_mixed_strat = DOBSS_mixed_strat_list[strat[DOBSS]]

				BSSQ_mixed_strat = BSSQ_mixed_strat_list[strat[BSSQ]]

				#Updating Biased ASLR
				start = time.time()
				for cprime in range(NUMCONFIGS):
					if((vulset[cprime, attack[BiasedASLR]] == 1) & (def_util[typ[BiasedASLR], attack[BiasedASLR]] < 0)):
						config_hit_count[cprime] += 1
				end = time.time()
				BiasedASLR_runtime += (end - start)


				# Reward estimates using Geometric Resampling
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
				FPLMTD_rhat = FPLMTD_GR(FPLMTD_rhat, strat_old[FPLMTD], strat[FPLMTD], util[FPLMTD], sc, t, rng)
				end = time.time()
				FPLMTD_runtime += (end - start)

				start = time.time()
				FPLGR_rhat = FPL_GR(FPLGR_rhat, strat[FPLGR], util[FPLGR] - scosts[FPLGR], rng)
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
				PaddedExp3_L[strat[PaddedExp3]] += (util[PaddedExp3]) / updated_p[strat[PaddedExp3]]
				temp = np.sum([np.exp(EXP_ETA * PaddedExp3_L[c]) for c in range(NUMCONFIGS)])
				for c in range(NUMCONFIGS):
					PaddedExp3_p[c] = np.exp(EXP_ETA*PaddedExp3_L[c]) / temp
				end = time.time() 
				PaddedExp3_runtime += (end - start)

				for i in range(NUMSTRATS):
					strat_old[i] = strat[i]

				start = time.time()
				RobustRL_maxvalue[strat[RobustRL]] = min(util[RobustRL] - scosts[RobustRL], RobustRL_maxvalue[strat[RobustRL]])
				strat[RobustRL] = getRobustRLStrat(RobustRL_maxvalue, rng)
				end = time.time()
				RobustRL_runtime += (end - start)

			# print("Iteration " + str(iter+1) + " over.")

		print("\n")
		stdout = sys.stdout
		f_ov_out = open(OUT_DIR + str(dataset_num) + "overall_out_QuantalResponse.txt", 'w')
		sys.stdout = f_ov_out

		print(NUMCONFIGS)
		print(NUMATTACKS)
		print(NUMTYPES)

		print("\n")
		print("Average switches per iteration:")
		for i in range(NUMSTRATS):
			print(switch[i]/MAX_ITER)
		print("\n")

		print("Average run-times per iteration:")
		print(FPLMaxMin_runtime/MAX_ITER)
		print(FPLMTD_runtime/MAX_ITER)
		print(DOBSS_runtime/MAX_ITER)
		print(RANDOM_runtime/MAX_ITER)
		print(RobustRL_runtime/MAX_ITER)
		print(EXP3_runtime/MAX_ITER)
		print(BSSQ_runtime/MAX_ITER)
		print(PaddedExp3_runtime/MAX_ITER)
		print(SwitchingExp3_runtime/MAX_ITER)
		print(FPLGR_runtime/MAX_ITER)
		print(BiasedASLR_runtime/MAX_ITER)
		print("\n")

		print("Average utilities per iteration:")
		# FPLMTD, FPLMTDLite, DOBSS, RANDOM, RobustRL, EXP3, BSSQ sum of utilities
		f_out = open(OUT_DIR + str(dataset_num) + "output_QuantalResponse.txt", "w")
		for i in range(NUMSTRATS):
			# print(np.sum(utility[i, :])/MAX_ITER)
			print(str(np.sum(utility[i])/MAX_ITER))
			f_out.write("Strat "+str(i) + "\n")
			for iter in range(MAX_ITER):
				f_out.write("iter " + str(iter) + "\n")
				for t in range(T):
					f_out.write(str(utility[i][iter, t]/MAX_ITER) + " ")
				f_out.write("\n")
		print("\n")

		sys.stdout = stdout
		f_ov_out.close()
		f_out.close()








