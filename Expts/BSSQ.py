import gurobipy as gp
import numpy as np

n_episodes = 100
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
	q = {(i, j): m.addVar(lb = 0, ub = 1, vtype = gp.GRB.INTEGER, name = 'q_'+str(i)+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}
	
	# value of attacker's pure strategy
	v_a = {i: m.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype = gp.GRB.CONTINUOUS, name = 'v_a'+str(i)) for i in range(NUMTYPES)}

	m.update()

	# Update objective function
	for tau in range(NUMTYPES):
		for sdash in range(NUMCONFIGS):
			for a in range(NUMATTACKS):
				obj.add( P[tau] * game_def_qval[tau][s][sdash][a] * x[sdash] * q[tau, a] )

	# Add constraints to make attacker have a pure strategy
	for tau in range(NUMTYPES):
		q_sum = gp.LinExpr()
		for a in range(NUMATTACKS):
			q_sum.add(q[tau, a])
		m.addConstr(q_sum==1)

	m.update()

	# Add constraints to make attacker select dominant pure strategy
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			val = gp.LinExpr()
			val.add(v_a[tau])
			for sdash in range(NUMCONFIGS):
				val.add(float(game_att_qval[tau][s][sdash][a]) * x[sdash], -1.0)
			m.addConstr(val >= 0)
			m.addConstr(val <= (1 - q[tau, a]) * M)

	m.update()

	# set objective funcion
	m.setObjective(obj, gp.GRB.MAXIMIZE)

	# Solve MIQP
	m.optimize()

	# return x, q and values
	soln_x = [0.0 for i in range(NUMCONFIGS)]
	soln_q = [[0.0 for i in range(NUMATTACKS)] for j in range(NUMTYPES)]
	soln_v_a = [0.0 for i in range(NUMTYPES)]

	for sdash in range(NUMCONFIGS):
		soln_x[sdash] = x[sdash].X

	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			soln_q[tau][a] = q[tau, a].X

	for tau in range(NUMTYPES):
		soln_v_a[tau] = v_a[tau].X

	return soln_x, soln_q, m.objVal, soln_v_a


# returns BSG equilibrium values using decomposed MILP formulation (DOBSS)
def getSSEqMILP(s, game_def_qval, game_att_qval, NUMCONFIGS, NUMATTACKS, NUMTYPES, M, P):
	m = gp.Model("MILP")

	m.setParam('OutputFlag', 0)
	m.setParam('LogFile', '')
	m.setParam('LogToConsole', 0)
	
	# m.update()

	# declare objective function
	obj = gp.QuadExpr()
	x = []

	# pure strategies for (attacker type, attack)
	q = {(i, j): m.addVar(lb = 0, ub = 1, vtype = gp.GRB.INTEGER, name = 'q_'+str(i)+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}
	
	# value of attacker's pure strategy
	v_a = {i: m.addVar(lb = -gp.GRB.INFINITY, ub = gp.GRB.INFINITY, vtype = gp.GRB.CONTINUOUS, name = 'v_a'+str(i)) for i in range(NUMTYPES)}

	m.update()

	# Add constraints to make attacker have the dominant pure strategy
	for tau in range(NUMTYPES):
		q_sum = gp.LinExpr()
		for a in range(NUMATTACKS):
			q_sum.add(q[tau, a])
		m.addConstr(q_sum == 1)

	# z[sdash][a] = x[sdash] * q[tau][a] to linearlize
	z = []
	for tau in range(NUMTYPES):
		for sdash in range(NUMCONFIGS):
			zsdash = []
			for a in range(NUMATTACKS):
				zsdash.append(m.addVar(lb = 0, ub = 1, vtype = gp.GRB.CONTINUOUS, name = 'z_' + str(sdash) + str(a)))
			z.append(zsdash)

		m.update()

		# z constraints
		for a in range(NUMATTACKS):
			zsdash_sum = gp.LinExpr()
			for sdash in range(NUMCONFIGS):
				zsdash_sum.add(z[sdash][a])
			m.addConstr(zsdash_sum <= 1)
			m.addConstr(zsdash_sum >= q[tau, a])

		zsum = gp.LinExpr()
		for sdash in range(NUMCONFIGS):
			zq_sum = gp.LinExpr()
			for a in range(NUMATTACKS):
				zsum.add(z[sdash][a])
				zq_sum.add(z[sdash][a])
			if(tau == 0):
				x.append(zq_sum)
			else:
				m.addConstr(zq_sum == x[sdash])
			m.addConstr(zq_sum <= 1)
		m.addConstr(zsum == 1)

		# Update objective function
		for sdash in range(NUMCONFIGS):
			for a in range(NUMATTACKS):
				obj.add(P[tau] * game_def_qval[tau][s][sdash][a] * z[sdash][a])

		m.update()

	# Add constraints to make attacker select dominant pure strategy
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			val = gp.LinExpr()
			val.add(v_a[tau])
			for sdash in range(NUMCONFIGS):
				xsdash = gp.LinExpr()
				for a in range(NUMATTACKS):
					xsdash.add(z[sdash][a])
				val.add(float(game_att_qval[tau][s][sdash][a]) * xsdash, -1.0)
			m.addConstr(val >= 0)
			m.addConstr(val <= (1 - q[tau, a]) * M)
	
	m.update()

	# set objective funcion
	m.setObjective(obj, gp.GRB.MAXIMIZE)

	# Solve MILP
	m.optimize()

	# return x, q and values
	soln_x = [0.0 for i in range(NUMCONFIGS)]
	soln_q = [[0.0 for i in range(NUMATTACKS)] for j in range(NUMTYPES)]
	soln_v_a = [0.0 for i in range(NUMTYPES)]

	for sdash in range(NUMCONFIGS):
		soln_x[sdash] = x[sdash].getValue()

	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			soln_q[tau][a] = q[tau, a].X

	for tau in range(NUMTYPES):
		soln_v_a[tau] = v_a[tau].X

	return soln_x, soln_q, m.objVal, soln_v_a


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