from docplex.mp.model import Model

# returns x values if solution to MIQP exists when starting
def getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, M):
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
def getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, DOBSS_strat, NUMCONFIGS, NUMATTACKS, NUMTYPES, M):
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

# returns n values if solution to MIQP exists given a DOBSS strategy
def getStackelbergSolution(game_def_util, game_att_util, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, M):
	X = Model(name = 'dobss_attack')
	# rewards
	X.ra = game_att_util
	X.rd = game_def_util

	X.P = Pvec # attacker type prob
	X.m = M # large constant

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

	# Update objective function
	X.maximize(X.sum(X.P[tau]*X.rd[tau][c][a]*x[c]*n[tau, a] for c in range(NUMCONFIGS) for tau in range(NUMTYPES) for a in range(NUMATTACKS)))
	sol = X.solve()
	if(sol == None):
		print("Error: No solution instance")
		exit()

	# return attacker action values
	soln_a = []
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			if(n[tau, a].solution_value > 0.9):
				soln_a.append(a)
	# print(soln_a, NUMTYPES)
	return soln_a
