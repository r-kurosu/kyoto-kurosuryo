#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# this is for fuhctions to separate plane

import pulp, time

#　Self-made
from src import constant

####################################################

solver_type = constant.solver_type
CPLEX_PATH = constant.CPLEX_PATH
CPLEX_MSG = constant.CPLEX_MSG
CPLEX_TIMELIMIT = constant.CPLEX_TIMELIMIT

####################################################

def y_f(y, y_obs):
    ans = abs(y - y_obs) ** 2
    # ans = abs(y - y_obs) ** (0.1)
    return ans

def solve_MILP(x, y, y_obs):

    MILP = pulp.LpProblem("find_hyperplane", pulp.LpMinimize)

    K = len(x[0])

    w = {i: pulp.LpVariable(f"w({i})", cat=pulp.LpContinuous) for i in range(K)}
    b = pulp.LpVariable("b", cat=pulp.LpContinuous)
    eps = pulp.LpVariable("epsilon", 0, cat=pulp.LpContinuous)

    MILP += eps, "target"

    for (_x, _y) in zip(x, y):
        if _y <= y_obs:
            MILP += pulp.lpSum(w[i] * _x[i] for i in range(K)) - b <= -y_f(_y, y_obs) + eps
        else:
            MILP += pulp.lpSum(w[i] * _x[i] for i in range(K)) - b >= y_f(_y, y_obs) - eps

    # MILP.writeLP("test.lp")

    # Solve MILP
    solve_begin = time.time()
    if solver_type == 1:
        if CPLEX_TIMELIMIT > 0:
            CPLEX = pulp.CPLEX(path = CPLEX_PATH,
                               msg = CPLEX_MSG,
                               timeLimit = CPLEX_TIMELIMIT)
        else:
            CPLEX = pulp.CPLEX(path = CPLEX_PATH,
                               msg = CPLEX_MSG)
        # print("Start Solving Using CPLEX...")
        MILP.solve(CPLEX)
        solve_end = time.time()
    else:
        # print("Start Solving Using Coin-OR...")
        solver = pulp.COIN_CMD(msg = CPLEX_MSG)
        MILP.solve(solver)
        solve_end = time.time()

    w_value = {i: w[i].value() if w[i].value() is not None else 0 for i in range(K)}
    b_value = b.value()

    eps_value = eps.value()

    return w_value, b_value, eps_value, solve_end - solve_begin

def check(w, b, x):
    K = len(x)
    return sum(w[i] * x[i] for i in range(K)) - b

def split_D1_D2(x, y, w, b):
    D_1_ind = list()
    D_2_ind = list()

    for i, (_x, _y) in enumerate(zip(x, y)):
        if check(w, b, _x) <= 0:
            D_1_ind.append(i)
        else:
            D_2_ind.append(i)

    return x[D_1_ind], y[D_1_ind], x[D_2_ind], y[D_2_ind]