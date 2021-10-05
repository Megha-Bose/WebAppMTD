# import gurobipy as gp
import numpy as np
from docplex.mp.model import Model

n_episodes = 10
max_eps_len = 10
start_eps_val = 0.1
end_eps_val = 0.05

# select value from distribution
def getValFromDist(x, rng):
	y = rng.random()
	for i in range(len(x)):
		if(y <= x[i]):
			return i
		y -= x[i]
	return len(x) - 1 

# returns BSG equilibrium values
def getSSEqMIQP(s, game_def_qval, game_att_qval, NUMCONFIGS, NUMATTACKS, NUMTYPES, M, P):
	X = Model(name = 'BSSQ_MIQP')
	
	# defender mixed strategy
	x = {i: X.continuous_var(name = 'x_'+str(i), lb = 0, ub = 1) for i in range(NUMCONFIGS)}

	# Add defender stategy constraints: Σx = 1
	X.add_constraint(X.sum(x) == 1)
	
	# pure strategies for (attacker type, attack)
	q = {(i, j): X.binary_var(name = 'q_'+str(i)+'_'+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}
	
	# value of attacker's pure strategy
	v_a = {i: X.continuous_var(name = 'v_'+str(i)) for i in range(NUMTYPES)}

	# Add constraints to make attacker have a pure strategy
	for tau in range(NUMTYPES):
		X.add_constraint(X.sum(q[tau, a] for a in range(NUMATTACKS)) == 1)

	# Add constraints to make attacker select dominant pure strategy
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			X.add_constraint(v_a[tau] - X.sum(game_att_qval[tau][s][sdash][a]*x[sdash] for sdash in range(NUMCONFIGS)) >= 0)
			X.add_constraint(v_a[tau] - X.sum(game_att_qval[tau][s][sdash][a]*x[sdash] for sdash in range(NUMCONFIGS)) <= (1 - q[tau, a]) * M)

	# Maximize objective function
	X.maximize(X.sum( P[tau] * game_def_qval[tau][s][sdash][a] * x[sdash] * q[tau, a] for tau in range(NUMTYPES) for sdash in range(NUMCONFIGS) for a in range(NUMATTACKS)))

	# Solve MIQP
	sol = X.solve()
	if(sol == None):
		print("Error: No solution instance")
		exit()

	# return x, q and values
	soln_x = [0.0 for i in range(NUMCONFIGS)]
	soln_q = [[0.0 for i in range(NUMATTACKS)] for j in range(NUMTYPES)]
	soln_v_a = [0.0 for i in range(NUMTYPES)]

	for sdash in range(NUMCONFIGS):
		soln_x[sdash] = x[sdash].solution_value

	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			soln_q[tau][a] = q[tau, a].solution_value

	for tau in range(NUMTYPES):
		soln_v_a[tau] = v_a[tau].solution_value

	return soln_x, soln_q, X.objective_value, soln_v_a


# returns BSG equilibrium values using decomposed MILP formulation (DOBSS)
def getSSEqMILP(s, game_def_qval, game_att_qval, NUMCONFIGS, NUMATTACKS, NUMTYPES, M, P):
	X = Model(name = 'BSSQ_MILP')

	# declare objective function
	x = []

	# pure strategies for (attacker type, attack)
	q = {(i, j): X.binary_var(name = 'q_'+str(i)+'_'+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}
	
	# value of attacker's pure strategy
	v_a = {i: X.continuous_var(name = 'v_'+str(i)) for i in range(NUMTYPES)}

	# Add constraints to make attacker have the dominant pure strategy
	for tau in range(NUMTYPES):
		X.add_constraint(X.sum(q[tau, a] for a in range(NUMATTACKS)) == 1)

	# z[tau][sdash][a] = x[sdash] * q[tau][a] to linearlize
	z = []
	x = []
	for tau in range(NUMTYPES):
		zdash = []
		for sdash in range(NUMCONFIGS):
			zsdash = {a: X.continuous_var(name = 'z_' + str(tau) + '_' + str(sdash) + '_' + str(a), lb = 0, ub = 1) for a in range(NUMATTACKS)}
			zdash.append(zsdash)
		z.append(zdash)

		# z constraints
		for a in range(NUMATTACKS):
			X.add_constraint(X.sum(z[tau][sdash][a] for sdash in range(NUMCONFIGS)) <= 1)
			X.add_constraint(X.sum(z[tau][sdash][a] for sdash in range(NUMCONFIGS)) >= q[tau, a])

		for sdash in range(NUMCONFIGS):
			X.add_constraint(X.sum(z[tau][sdash][a] for a in range(NUMATTACKS)) <= 1)
			if(tau == 0):
				x.append(X.sum(z[tau][sdash][a] for a in range(NUMATTACKS)))
			else:
				X.add_constraint(X.sum(z[tau][sdash][a] for a in range(NUMATTACKS)) == x[sdash])
		X.add_constraint(X.sum(z[tau][sdash][a] for sdash in range(NUMCONFIGS) for a in range(NUMATTACKS)) == 1)

		# Add constraints to make attacker select dominant pure strategy
		xsdash = X.sum(z[tau][sdash][a] for a in range(NUMATTACKS) for sdash in range(NUMCONFIGS))
		X.add_constraint(v_a[tau] - X.sum(game_att_qval[tau][s][sdash][a]*xsdash) >= 0)
		X.add_constraint(v_a[tau] - X.sum(game_att_qval[tau][s][sdash][a]*xsdash) <= (1 - q[tau, a]) * M)

	X.maximize(X.sum(P[tau] * game_def_qval[tau][s][sdash][a] * z[tau][sdash][a] for tau in range(NUMTYPES) for sdash in range(NUMCONFIGS) for a in range(NUMATTACKS)))

	# Solve MILP
	sol = X.solve()
	if(sol == None):
		print("Error: No solution instance")
		exit()

	# return x, q and values
	soln_x = [0.0 for i in range(NUMCONFIGS)]
	soln_q = [[0.0 for i in range(NUMATTACKS)] for j in range(NUMTYPES)]
	soln_v_a = [0.0 for i in range(NUMTYPES)]

	for sdash in range(NUMCONFIGS):
		soln_x[sdash] = x[sdash].solution_value

	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			soln_q[tau][a] = q[tau, a].solution_value

	for tau in range(NUMTYPES):
		soln_v_a[tau] = v_a[tau].solution_value

	return soln_x, soln_q, X.objective_value, soln_v_a


def getBSSQStrat(def_util, att_util, sc, p, P, NUMCONFIGS, NUMATTACKS, NUMTYPES, ALPHA, DISCOUNT_FACTOR, M, rng):
	x = [[(1/NUMCONFIGS) for i in range(NUMCONFIGS)] for i in range(NUMCONFIGS)]
	q = [[[(1/NUMATTACKS) for a in range(NUMATTACKS)] for i in range(NUMCONFIGS)] for j in range(NUMTYPES)]

	v_def = [0.0 for i in range(NUMCONFIGS)]
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
	
	decay_val = (end_eps_val / start_eps_val) ** (1 / max_eps_len)

	for ep in range(n_episodes):
		# print("BSSQ episode:"+str(ep), end = "\r")
		# sampling start state
		s = getValFromDist(p, rng)

		eps_val = start_eps_val

		itr = 0
		while itr <= max_eps_len:
			# sampling attacker type		
			tau = getValFromDist(P, rng)

			sdash = None
			a = None
			
			# eps-greedy sampling of defender action from x
			y = rng.random()

			if(y < eps_val):
				# exploration
				sdash = rng.integers(low=0, high=NUMCONFIGS, size=1)[0]
			else:
				# greedy exploitation
				sdash = np.argmax(x[s])

			# eps-greedy sampling of attacker action from q
			y = rng.random()

			if(y < eps_val):
				# exploration
				a = rng.integers(low=0, high=NUMATTACKS, size=1)[0]
			else:
				# greedy exploitation
				a = np.argmax(q[tau][s])

			Qval_def[tau][s][sdash][a] = (1 - ALPHA) * Qval_def[tau][s][sdash][a] + ALPHA * (game_def_reward[tau][s][sdash][a] + DISCOUNT_FACTOR * v_def[sdash])
			Qval_att[tau][s][sdash][a] = (1 - ALPHA) * Qval_att[tau][s][sdash][a] + ALPHA * (game_att_reward[tau][s][sdash][a] + DISCOUNT_FACTOR * v_att[tau][sdash])

			# get BSMG SSEquilibrium values
			# x[s], ret_q, v_def[s], ret_v_att = getSSEqMILP(s, Qval_def, Qval_att, NUMCONFIGS, NUMATTACKS, NUMTYPES, M, P)
			x[s], ret_q, v_def[s], ret_v_att = getSSEqMIQP(s, Qval_def, Qval_att, NUMCONFIGS, NUMATTACKS, NUMTYPES, M, P)

			for tau in range(NUMTYPES):
				for a in range(NUMATTACKS):
					q[tau][s][a] = ret_q[tau][a]

			for tau in range(NUMTYPES):
				v_att[tau][s] = ret_v_att[tau]

			# epsilon decay
			eps_val = eps_val * decay_val
			s = sdash
			itr += 1
	return x