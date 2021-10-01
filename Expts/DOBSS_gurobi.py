import gurobipy as gp
# from docplex.mp.model import Model

# returns x values if solution to MIQP exists when starting
def getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, M):
	m = gp.Model("dobss_init")

	m.setParam('OutputFlag', 0)
	m.setParam('LogFile', '')
	m.setParam('LogToConsole', 0)

	obj = gp.QuadExpr()

	# defender mixed strategy
	x = {i: m.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name='x_'+str(i)) for i in range(NUMCONFIGS)}
	m.update()
	
	# Σx = 1
	x_sum = gp.LinExpr()
	for i in range(NUMCONFIGS):
		x_sum.add(x[i])
	m.addConstr(x_sum == 1)
	m.update()

	# pure strategies for (attacker type, attack)
	n = {(i, j): m.addVar(lb=0, ub=1, vtype=gp.GRB.INTEGER, name='n_'+str(i)+'_'+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}

	# Σn = 1
	for tau in range(NUMTYPES):
		n_sum = gp.LinExpr()
		for a in range(NUMATTACKS):
			n_sum.add(n[tau, a])
		m.addConstr(n_sum == 1)
	m.update()

	# value of attacker's pure strategy
	v = {i: m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='v_'+str(i)) for i in range(NUMTYPES)}
	
	m.update()

	# w_ij = x_i * x_j
	w = {(i, j): m.addVar(vtype=gp.GRB.CONTINUOUS, name='w_'+str(i)+'_'+str(j)) for i in range(NUMCONFIGS) for j in range(NUMCONFIGS)}
	for c in range(NUMCONFIGS):
		for cdash in range(NUMCONFIGS):
			# using McCormick envelopes to get bounds for the non-convex function w_ij = x_i * x_j
			m.addConstr(w[c, cdash] >= 0)
			m.addConstr(w[c, cdash] <= x[c])
			m.addConstr(w[c, cdash] <= x[cdash])

	for c in range(NUMCONFIGS):
		from_constr = gp.LinExpr()
		to_constr = gp.LinExpr()
		for cdash in range(NUMCONFIGS):
			from_constr.add(w[c, cdash])
			to_constr.add(w[cdash, c])
		m.addConstr(from_constr <= x[c])
		m.addConstr(to_constr <= x[c])	
	
	m.update()
	
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			val = gp.LinExpr()
			val.add(v[tau])
			for c in range(NUMCONFIGS):
				val.add(game_att_util[tau][c][a] * x[c], -1.0)
			m.addConstr(val >= 0, n[tau, a].getAttr("VarName") + "lb")
			m.addConstr(val <= (1 - n[tau, a]) * M, n[tau, a].getAttr("VarName") + "ub")

	# maximise total reward - switching cost
	# Update objective function
	for tau in range(NUMTYPES):
		for c in range(NUMCONFIGS):
			for a in range(NUMATTACKS):
				obj.add(Pvec[tau] * game_def_util[tau][c][a] * x[c] * n[tau, a])

	# McCormick envelope approximation
	w_sum = gp.LinExpr()
	for i in range(NUMCONFIGS):
		for j in range(NUMCONFIGS):
			obj.add(sc[i][j] * w[i, j], -1)
			w_sum.add(w[i, j])
	m.addConstr(w_sum == 1)

	m.update()

	# set objective funcion
	m.setObjective(obj, gp.GRB.MAXIMIZE)

	# solve MIQP
	m.optimize()
	
	soln_x = []
	# return x values
	for c in range(NUMCONFIGS):
		soln_x.append(x[c].X)
	return soln_x


# returns x values if solution to MIQP exists given a DOBSS strategy
def getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, DOBSS_strat, NUMCONFIGS, NUMATTACKS, NUMTYPES, M):
	m = gp.Model("dobss_next")

	m.setParam('OutputFlag', 0)
	m.setParam('LogFile', '')
	m.setParam('LogToConsole', 0)

	obj = gp.QuadExpr()

	PureStrat = [0]*NUMCONFIGS
	PureStrat[DOBSS_strat] = 1 

	# defender mixed strategy
	x = {i: m.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name='x_'+str(i)) for i in range(NUMCONFIGS)}
	m.update()

	# pure strategies for (attacker type, attack)
	n = {(i, j): m.addVar(lb=0, ub=1, vtype=gp.GRB.INTEGER, name='n_'+str(i)+'_'+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}
	m.update()

	# value of attacker's pure strategy
	v = {i: m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='v_'+str(i)) for i in range(NUMTYPES)}
	m.update()

	# Σx = 1
	x_sum = gp.LinExpr()
	for i in range(NUMCONFIGS):
		x_sum.add(x[i])
	m.addConstr(x_sum == 1)
	m.update()

	# Σn = 1
	for tau in range(NUMTYPES):
		n_sum = gp.LinExpr()
		for a in range(NUMATTACKS):
			n_sum.add(n[tau, a])
		m.addConstr(n_sum == 1)
	m.update()

	# value constraints 
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			val = gp.LinExpr()
			val.add(v[tau])
			for c in range(NUMCONFIGS):
				val.add(game_att_util[tau][c][a] * x[c], -1.0)
			m.addConstr(val >= 0, n[tau, a].getAttr("VarName") + "lb")
			m.addConstr(val <= (1 - n[tau, a]) * M, n[tau, a].getAttr("VarName") + "ub")


	# maximise total reward - switching cost

	# Update objective function
	for tau in range(NUMTYPES):
		for c in range(NUMCONFIGS):
			for a in range(NUMATTACKS):
				obj.add(Pvec[tau] * game_def_util[tau][c][a] * x[c] * n[tau, a])

	for i in range(NUMCONFIGS):
		for j in range(NUMCONFIGS):
			obj.add(sc[i][j] * PureStrat[i] * x[j], -1)

	m.update()

	# set objective funcion
	m.setObjective(obj, gp.GRB.MAXIMIZE)

	# solve MIQP
	m.optimize()

	# return x values
	soln_x = []
	for c in range(NUMCONFIGS):
		soln_x.append(x[c].X)
	return soln_x

# returns n values if solution to MIQP exists given a DOBSS strategy
def getStackelbergSolution(game_def_util, game_att_util, Pvec, NUMCONFIGS, NUMATTACKS, NUMTYPES, M):
	m = gp.Model("dobss_attack")

	m.setParam('OutputFlag', 0)
	m.setParam('LogFile', '')
	m.setParam('LogToConsole', 0)

	obj = gp.QuadExpr()

	# defender mixed strategy
	x = {i: m.addVar(lb=0, ub=1, vtype=gp.GRB.CONTINUOUS, name='x_'+str(i)) for i in range(NUMCONFIGS)}
	m.update()

	# pure strategies for (attacker type, attack)
	n = {(i, j): m.addVar(lb=0, ub=1, vtype=gp.GRB.INTEGER, name='n_'+str(i)+'_'+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}
	m.update()

	# value of attacker's pure strategy
	v = {i: m.addVar(lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS, name='v_'+str(i)) for i in range(NUMTYPES)}
	m.update()

	# Σx = 1
	x_sum = gp.LinExpr()
	for i in range(NUMCONFIGS):
		x_sum.add(x[i])
	m.addConstr(x_sum == 1)
	m.update()

	# Σn = 1
	for tau in range(NUMTYPES):
		n_sum = gp.LinExpr()
		for a in range(NUMATTACKS):
			n_sum.add(n[tau, a])
		m.addConstr(n_sum == 1)
	m.update()

	# value constraints 
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			val = gp.LinExpr()
			val.add(v[tau])
			for c in range(NUMCONFIGS):
				val.add(game_att_util[tau][c][a] * x[c], -1.0)
			m.addConstr(val >= 0)
			m.addConstr(val <= (1 - n[tau, a]) * M)


	# maximise total reward - switching cost

	# Update objective function
	for tau in range(NUMTYPES):
		for c in range(NUMCONFIGS):
			for a in range(NUMATTACKS):
				obj.add(Pvec[tau] * game_def_util[tau][c][a] * x[c] * n[tau, a])

	m.update()

	# set objective funcion
	m.setObjective(obj, gp.GRB.MAXIMIZE)

	# solve MIQP
	m.optimize()

	# return x values
	soln_a = []
	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			if(n[tau, a].X > 0.9):
				soln_a.append(a)
	# print(soln_a, NUMTYPES)
	return soln_a

