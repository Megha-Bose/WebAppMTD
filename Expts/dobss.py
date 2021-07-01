import numpy as np 
from docplex.mp.model import Model

NUMTYPES = 1
NUMCONFIGS = 2
NUMATTACKS = 2
M = 10000000

def getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec):
	X = Model(name = 'dobss_init')

	X.ra = game_att_util
	X.rd = game_def_util
	X.P = Pvec
	X.sc = sc
	X.m = M

	x = {i: X.continuous_var(name = 'x_'+str(i), lb = 0, ub = 1) for i in range(NUMCONFIGS)}
	n = {(i, j): X.binary_var(name = 'n_'+str(i)+'_'+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}
	v = {i: X.continuous_var(name = 'v_'+str(i)) for i in range(NUMTYPES)}
	w = {(i, j): X.continuous_var(name = 'w_'+str(i)+'_'+str(j), lb = 0, ub = 1) for i in range(NUMCONFIGS) for j in range(NUMCONFIGS)}

	X.add_constraint(X.sum(x) == 1)
	for tau in range(NUMTYPES):
		X.add_constraint(X.sum(n[tau, a] for a in range(NUMATTACKS)) == 1)

	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			X.add_constraint(v[tau] - X.sum(X.ra[tau][c][a]*x[c] for c in range(NUMCONFIGS)) >= 0)
			X.add_constraint(v[tau] - X.sum(X.ra[tau][c][a]*x[c] for c in range(NUMCONFIGS)) <= (1 - n[tau, a])*X.m)

	X.add_constraint(X.sum(w) == 1)
	for c in range(NUMCONFIGS):
		for cdash in range(NUMCONFIGS):
			X.add_constraint(w[c, cdash] <= x[c])
			X.add_constraint(w[c, cdash] <= x[cdash])

	for c in range(NUMCONFIGS):
		X.add_constraint(X.sum(w[c, cdash] for cdash in range(NUMCONFIGS)) <= x[c])
		X.add_constraint(X.sum(w[cdash, c] for cdash in range(NUMCONFIGS)) <= x[c])

	X.maximize(X.sum(X.P[tau]*X.rd[tau][c][a]*x[c]*n[tau, a] for c in range(NUMCONFIGS) for tau in range(NUMTYPES) for a in range(NUMATTACKS)) - X.sum(X.sc[i, j]*w[i, j] for i in range(NUMCONFIGS) for j in range(NUMCONFIGS)))
	sol = X.solve()
	if(sol == None):
		print("Error: No solution instance")
		exit()
	# sol.display()
	soln_x = []
	for c in range(NUMCONFIGS):
		soln_x.append(x[c].solution_value)
	return soln_x


def getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, DOBSS_strat):
	X = Model(name = 'dobss_next')

	X.ra = game_att_util
	X.rd = game_def_util
	X.P = Pvec
	X.sc = sc
	X.m = M
	PureStrat = [0]*NUMCONFIGS
	PureStrat[DOBSS_strat] = 1 
	X.PS = PureStrat

	x = {i: X.continuous_var(name = 'x_'+str(i), lb = 0, ub = 1) for i in range(NUMCONFIGS)}
	n = {(i, j): X.binary_var(name = 'n_'+str(i)+'_'+str(j)) for i in range(NUMTYPES) for j in range(NUMATTACKS)}
	v = {i: X.continuous_var(name = 'v_'+str(i)) for i in range(NUMTYPES)}

	X.add_constraint(X.sum(x) == 1)
	for tau in range(NUMTYPES):
		X.add_constraint(X.sum(n[tau, a] for a in range(NUMATTACKS)) == 1)

	for tau in range(NUMTYPES):
		for a in range(NUMATTACKS):
			X.add_constraint(v[tau] - X.sum(X.ra[tau][c][a]*x[c] for c in range(NUMCONFIGS)) >= 0)
			X.add_constraint(v[tau] - X.sum(X.ra[tau][c][a]*x[c] for c in range(NUMCONFIGS)) <= (1 - n[tau, a])*X.m)


	X.maximize(X.sum(X.P[tau]*X.rd[tau][c][a]*x[c]*n[tau, a] for c in range(NUMCONFIGS) for tau in range(NUMTYPES) for a in range(NUMATTACKS)) - X.sum(X.sc[i, j]*X.PS[i]*x[j] for i in range(NUMCONFIGS) for j in range(NUMCONFIGS)))
	sol = X.solve()
	if(sol == None):
		print("Error: No solution instance")
		exit()
	# sol.display()
	soln_x = []
	for c in range(NUMCONFIGS):
		soln_x.append(x[c].solution_value)
	return soln_x


sc = np.array([[0, 3], [3, 0]])
game_def_util = [[[1, -1], [-1, 1]]]
game_att_util = [[[-1, 1], [1, -1]]]
Pvec = [1]

DOBSS_mixed_strat_list = []
for c in range(NUMCONFIGS):
	DOBSS_mixed_strat_list.append(getDOBSSStrat(game_def_util, game_att_util, sc, Pvec, c))
DOBSS_mixed_strat_list.append(getInitDOBSSStrat(game_def_util, game_att_util, sc, Pvec))
print(DOBSS_mixed_strat_list)




