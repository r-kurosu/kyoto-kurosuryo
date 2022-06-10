import statistics
import numpy as np
import pandas as pd
import pulp
import time, copy, itertools, math, warnings, os, sys
import openpyxl as excel
import datetime
import pathlib
from scipy.spatial import distance
from math import dist
# import cplex
# import gurobipy as gp

import sys
# sys.path.append("/Applications/CPLEX_Studio221/cplex/python/3.9/x86-64_osx")
# print(sys.path)
# import cplex

CPLEX_PATH = "/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex"
GUROBI_PATH = "/Users/kurosuryou/gurobi.lic"


def read_dataset(data_csv, value_txt):
    ### read files ###
    # read the csv and the observed values
    x = pd.read_csv(data_csv, index_col=0)
    value = pd.read_csv(value_txt)

    ### prepare data set ###
    # prepare CIDs
    CIDs = np.array(x.index)
    # prepare target, train, test arrays
    target = np.array(value['a'])
    # construct dictionary: CID to feature vector
    fv_dict = {}
    for cid, row in zip(CIDs, x.values):
        fv_dict[cid] = row
    # construct dictionary: CID to target value
    target_dict = {}
    for cid, val in zip(np.array(value['CID']), np.array(value['a'])):
        target_dict[cid] = val
    # check CIDs: target_values_filename should contain all CIDs that appear in descriptors_filename
    for cid in CIDs:
        if cid not in target_dict:
            # sys.stderr.write('error: {} misses the target value of CID {}\n'.format(target_values_filename, cid))
            exit(1)

    y = np.array([target_dict[cid] for cid in CIDs])
    return x, y


def pre_problem(x, y, D, K):
    st_time1 = time.time()
    index_A = []
    index_B = []
    temp_max = 0

    for i in range(D):
        if y.loc[i] == 0:
            index_A.append(i)
        else:
            index_B.append(i)

    if len(index_A) == 0:
        for j in index_B:
            temp = distance.euclidean(0, x.loc[j])
            if temp_max <= temp:
                temp_max = temp
                x_a = [0]*K
                x_b = x.loc[j]
                return x_a, x_b
    if len(index_B) == 0:
        for i in index_A:
            temp = distance.euclidean(x.loc[i], 0)
            if temp_max <= temp:
                temp_max = temp
                x_a = x.loc[i]
                x_b = [0]*K
                return x_a, x_b

    for i in index_A:
        for j in index_B:
            temp = distance.euclidean(x.loc[i], x.loc[j])
            if temp_max <= temp:
                temp_max = temp
                x_a = x.loc[i]
                x_b = x.loc[j]

    ed_time1 = time.time()
    print("pre_proc_time = {:.1f}".format(ed_time1-st_time1))

    return x_a, x_b


def find_separator(x_df, y, D, K, w_p, b_p, x_a, x_b):
    model = pulp.LpProblem("Linear_Separator", pulp.LpMinimize)
    print(f"K={K}")
    print(f"n={D}")
    param = 1

    # 変数定義
    b = pulp.LpVariable("b", cat=pulp.LpContinuous)
    # b = pulp.LpVariable("b", -1, 1, cat=pulp.LpContinuous)
    # b = pulp.LpVariable("b", -(10**param), 10**param, cat=pulp.LpContinuous)
    # b = pulp.LpVariable("b", lowBound=0,  cat=pulp.LpContinuous)
    # b_ = pulp.LpVariable("b_", lowBound=0, cat=pulp.LpContinuous)

    w = [pulp.LpVariable("w_{}".format(i), cat=pulp.LpContinuous) for i in range(K)]
    # w = [pulp.LpVariable("w_{}".format(i), -1, 1, cat=pulp.LpContinuous) for i in range(K)]
    # w = [pulp.LpVariable("w_{}".format(i), -(10**param), 10**param, cat=pulp.LpContinuous) for i in range(K)]
    # w = [pulp.LpVariable("w_{}".format(i), lowBound=0, cat=pulp.LpContinuous) for i in range(K)]
    # w_ = [pulp.LpVariable("w__{}".format(i), lowBound=0, cat=pulp.LpContinuous) for i in range(K)]

    eps = pulp.LpVariable('eps', lowBound=0, cat=pulp.LpContinuous)

    # 目的関数
    model += eps

    # 制約条件
    delta = 0.0001
    model += pulp.lpDot(w, x_a) - b <= -1
    model += pulp.lpDot(w, x_b) - b >= 1

    for i in range(D):
        if y.loc[i] == 0:
            model += pulp.lpDot(w, x_df.loc[i]) - b <= -1 + eps
            # model += pulp.lpDot(w, x_df.loc[i]) - pulp.lpDot(w_, x_df.loc[i]) - b <= -1 + eps
        else:
            model += pulp.lpDot(w, x_df.loc[i]) - b >= 1 - eps
            # model += pulp.lpDot(w, x_df.loc[i]) - pulp.lpDot(w_, x_df.loc[i]) - b >= 1 - eps

    # model += pulp.lpSum(w) + pulp.lpSum(w_) >= delta

    status = model.solve(pulp.CPLEX_CMD(path=CPLEX_PATH, msg=0))
    # status = model.solve(pulp.GUROBI(path=GUROBI_PATH, msg=0))

    # 出力

    if status == pulp.LpStatusOptimal:
        w_ast = [w[i].value() for i in range(len(w))]
        # w_ast = [w[i].value() - w_[i].value() for i in range(len(w))]
        b_ast = b.value()
        # b_ast = b.value() - b_.value()
        eps_ast = eps.value()
        w_p.append(w_ast)
        b_p.append(b_ast)
        for i in range(len(w_ast)):
            if w_ast[i] is None:
                w_ast[i] = 0

        return w_ast, b_ast, eps_ast
    else:
        print('FindSeparator infeasible')
        return None, None, None


def find_separator_by_CPLEX(x_df, y, D, K, w_p, b_p, CIDs):
    model = cplex.Cplex()
    model.set_problem_type(model.problem_type.MILP)
    model.objective.set_sense(model.objective.sense.minimize)
    model.set_problem_name("MILP")
    
    # 変数定義
    model.variables.add(names="b", types = "C")
    [model.variables.add(names=f"w_{i}", types = "C") for i in range(K)]
    model.variables.add(names="eps", lb=0, types="C")

    # 目的関数
    model.objective.set_linear("eps", 1)

    # 制約条件

    return


def find_separator_by_GUROBI(x_df, y, D, K, w_p, b_p, CIDs):
    model = gp.Model(name = "MILP")
    
    # ノーマルバージョン -----------------------------------------------------------------------
    # w = [0]*K
    # for i in range(K):
    #     # w[i] = model.addVar(vtype=gp.GRB.CONTINUOUS)
    #     w[i] = model.addVar(lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS)
    # # b = model.addVar(vtype=gp.GRB.CONTINUOUS)
    # b = model.addVar(lb=-1, ub=1, vtype=gp.GRB.CONTINUOUS)
    # eps = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)

    # c0 = []
    # c1 = []
    # for i in range(D):
    #     if y.loc[i] == 0:
    #         c0.append(model.addConstr(naiseki(w, x_df.loc[i]) - b <= -1 + eps))
    #     elif y.loc[i] == 1:
    #         c1.append(model.addConstr(naiseki(w, x_df.loc[i]) -b >= 1 - eps))
    # c = model.addConstr(gp.quicksum(w[i] for i in range(K)) > 0)
    # ----------------------------------------------------------------------------------------
    
    
    # 絶対値バージョン-------------------------------------------------------------------------
    # w = [0]*K
    # w_ = [0]*K
    # for i in range(K):
    #     w[i] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)
    #     w_[i] = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)
    # b = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)
    # b_ = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)
    # eps = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS)
    # c0 = []
    # c1 = []
    # for i in range(D):
    #     if y.loc[i] == 0:
    #         c0.append(model.addConstr(naiseki(w, x_df.loc[i]) - naiseki(w_, x_df.loc[i]) - b <= -1 + eps))
    #     elif y.loc[i] == 1:
    #         c1.append(model.addConstr(naiseki(w, x_df.loc[i]) - naiseki(w_, x_df.loc[i]) - b >= 1 - eps))
    # c = model.addConstr(gp.quicksum(w[i] for i in range(K)) + gp.quicksum(w_[i] for i in range(K)) >= 0.1)
    # ----------------------------------------------------------------------------------------


    # 新しい定式化---
    w = [0]*K
    for i in range(K):
        # w[i] = model.addVar(vtype=gp.GRB.CONTINUOUS)
        w[i] = model.addVar(vtype=gp.GRB.CONTINUOUS)
    # b = model.addVar(vtype=gp.GRB.CONTINUOUS)
    b = model.addVar(vtype=gp.GRB.CONTINUOUS)
    
    # 目的関数
    # model.setObjective(eps, sense=gp.GRB.MINIMIZE)
    model.setObjective(sum((naiseki(w, x_df.loc[i])-b - y.loc[i])**2 for i in range(D)))

    # 求解
    model.update
    model.params.NonConvex = 2
    model.Params.NumericFocus = 3
    model.Params.OutputFlag == 0
    model.Params.MIPFocus = 3
    model.optimize()
    
    if model.Status == gp.GRB.OPTIMAL:
        w_ast = [w[i].X for i in range(len(w))]
        b_ast = b.X
        # eps_ast = eps.X
        eps_ast = 1
        w_p.append(w_ast)
        b_p.append(b_ast)
        for i in range(len(w_ast)):
            if w_ast[i] is None:
                w_ast[i] = 0

        return w_ast, b_ast, eps_ast
    else:
        print('FindSeparator infeasible')
        return None, None, None
    return


def find_separator_b(x, y, D, K, w_p, b_p, CIDs):
    y_obs = 1/2
    # x_df, y, D, K, w_p, b_p, CIDs
    MILP = pulp.LpProblem("find_hyperplane", pulp.LpMinimize)


    # w = {i: pulp.LpVariable(f"w({i})", cat=pulp.LpContinuous) for i in range(K)}
    # w = {i: pulp.LpVariable(f"w({i})", -1, 1, cat=pulp.LpContinuous) for i in range(K)}
    w = [pulp.LpVariable(f"w_{i}", cat=pulp.LpContinuous) for i in range(K)]
    b = pulp.LpVariable("b", cat=pulp.LpContinuous)
    eps = pulp.LpVariable("epsilon", 0, cat=pulp.LpContinuous)

    MILP += eps

    # for (_x, _y) in zip(x, y):
    for i in range(D):
        _x = x.loc[i]
        _y = y.loc[i]
        if _y <= y_obs:
            # MILP += pulp.lpSum(w[i] * _x[i] for i in range(K)) - b <= 1/4 + eps
            MILP += pulp.lpDot(w, _x) - b <= -1 + eps
        else:
            # MILP += pulp.lpSum(w[i] * _x[i] for i in range(K)) - b >= -0.25 - eps
            MILP += pulp.lpDot(w, _x) - b >= 1 - eps

    # MILP.writeLP("test.lp")

    # Solve MILP
    solve_begin = time.time()
    CPLEX = pulp.CPLEX(path = CPLEX_PATH, msg = 0)
        # else:
        #     CPLEX = pulp.CPLEX(path = CPLEX_PATH,
        #                        msg = 0)
        # print("Start Solving Using CPLEX...")
    status = MILP.solve(CPLEX)
    solve_end = time.time()
    # else:
    #     # print("Start Solving Using Coin-OR...")
    #     solver = pulp.COIN_CMD(msg = CPLEX_MSG)
    #     MILP.solve(solver)
    #     solve_end = time.time()
    if status == pulp.LpStatusOptimal:

        w_value = {i: w[i].value() if w[i].value() is not None else 0 for i in range(K)}
        b_value = b.value()

        eps_value = eps.value()

    return w_value, b_value, eps_value


def count_s(y):
    s, s_p = 0, 0
    for a in y:
        if a == 0:
            s += 1
        else:
            s_p += 1

    return s, s_p


def sort_dataset(x_df, y, w, b):
    z, z_p = [], []
    for index, a in y.iteritems():
        if a == 0:
            z.append(naiseki(w, x_df.loc[index]) - b)
        elif a == 1:
            z_p.append(naiseki(w, x_df.loc[index]) - b)
    if y is None:
        return z, z_p
    z.sort()
    z_p.sort(reverse=True)
    # print(z[:5])
    # print(z_p[:5])

    return z, z_p


def naiseki(a, b):
    result = 0
    for i in range(len(a)):
        result += a[i] * b[i]

    return result


def find_r(z, z_p):
    r, r_p = 0, 0
    for i in range(len(z)):
        if z[i] < 0:
            r = i
        else:
            # print('all z is grater than 0')
            break
    for i in range(len(z_p)):
        if z_p[i] > 0:
            r_p = i
        else:
            # print('all z is less than 0')
            break

    return r, r_p


def find_index_l(z, r, z_p, r_p, s, s_p, rho_arg, theta_arg):
    rho = rho_arg
    theta = theta_arg
    for l in range(r, 0, -1):
        if siki_1(l, z, z_p, r, r_p, s, s_p, rho) == True:
            # print(f"l={l}")
            # print(f"r={r}")
            c_A = -z[l]
            break
        c_A = -z[math.floor(r * theta)]

    for l in range(r_p, 0, -1):
        if siki_2(l, z, z_p, r, r_p, s, s_p, rho) == True:
            # print(f"l={l}")
            # print(f"r\'={r_p}")
            c_B = -z_p[l]
            break
        c_B = -z_p[math.floor(r_p * theta)]

    if r <= 0:
        c_A = 0
    if r_p <= 0:
        c_B = 0

    return c_A, c_B


def siki_1(l, z, z_p, r, r_p, s, s_p, rho):
    temp = 0  # |l|
    # print(r_p, s_p)
    # print(z_p)
    # print(z)
    for j in range(r_p + 1, s_p + 1):
        if z_p[j] <= z[l]:
            temp += 1
    if temp / l <= rho * s_p / s:
        return True

    return False


def siki_2(l, z, z_p, r, r_p, s, s_p, rho):
    temp = 0
    for j in range(r + 1, s + 1):
        if z[j] >= z_p[l]:
            temp += 1
    if temp / l <= rho * s / s_p:
        return True

    return False


def redefine_func(x_df, y, w, b, c_A, c_B, CIDs, a_score):
    for index, vector_x in x_df.iterrows():
        # ver2. 正解を気にしない
        if naiseki(w, vector_x) - b <= -c_A:
            x_df.drop(index, axis=0, inplace=True)
            y.drop(index, inplace=True)
            a_score[CIDs.loc[index]] = 0
            CIDs.drop(index, inplace=True)

        elif naiseki(w, vector_x) - b >= c_B:
            x_df.drop(index, axis=0, inplace=True)
            y.drop(index, inplace=True)
            a_score[CIDs.loc[index]] = 1
            CIDs.drop(index, inplace=True)

    D = len(y)

    return D, x_df, y


def set_a_q(x_df, y, CIDs, a_score):
    countA, countB = 0, 0
    countB = y.sum()
    countA = len(y) - countB
    if countA >= countB:
        value = 0
    else:
        value = 1

    for index, vector_x in x_df.iterrows():
        a_score.replace(-1, value, inplace=True)

    return a_score


def experiment_test(x_test, y_test, w, b, CIDs_test, a_score_test, rho_arg, theta_arg):
    # 3.1 s, s'を数える
    s, s_p = count_s(y_test)
    # 3.2 ソートする
    z, z_p = sort_dataset(x_test, y_test, w, b)
    # 3.3 r, r'を探す
    r, r_p = find_r(z, z_p)
    # 3.4 index(l)を探す
    c_A, c_B = find_index_l(z, r, z_p, r_p, s, s_p, rho_arg, theta_arg)
    # 3.5 振り分け
    for index, vector_x in x_test.iterrows():
        if naiseki(w, vector_x) - b <= -c_A:
            x_test.drop(index, axis=0, inplace=True)
            y_test.drop(index, inplace=True)
            a_score_test[CIDs_test.loc[index]] = 0
            CIDs_test.drop(index, inplace=True)

        elif naiseki(w, vector_x) - b >= c_B:
            x_test.drop(index, axis=0, inplace=True)
            y_test.drop(index, inplace=True)
            a_score_test[CIDs_test.loc[index]] = 1
            CIDs_test.drop(index, inplace=True)

    new_D = len(x_test)
    return new_D, x_test, y_test


def constructing_DT_based_HP(x_df, y, D, K, w_p, b_p, c_p_A, c_p_B, CIDs, a_score, rho_arg, theta_arg):
    # 2. 超平面（hyper plane）を探す
    x_a, x_b = pre_problem(x_df, y, D, K)

    w, b, eps = find_separator(x_df, y, D, K, w_p, b_p, x_a, x_b)

    print(f"w={w}")
    print(f"b={b}")
    print(f"eps={eps}")
    # for w_i in w:
    #     if w_i != 0:
    #         print(w_i)

    # 3. 決定木の実装
    # 3.1 s, s'を数える
    s, s_p = count_s(y)

    # 3.2 ソートする
    z, z_p = sort_dataset(x_df, y, w, b)
    # print(s, len(z))
    # print(s_p, len(z_p))

    # 3.3 r, r'を探す
    r, r_p = find_r(z, z_p)
    # print(r, r_p)

    # 3.4 index(l)を探す
    c_A, c_B = find_index_l(z, r, z_p, r_p, s, s_p, rho_arg, theta_arg)
    c_p_A.append(c_A)
    c_p_B.append(c_B)
    # print(c_A, c_B)

    # 3.5 関数Φとデータセットの再定義
    D, x_df, y = redefine_func(x_df, y, w, b, c_A, c_B, CIDs, a_score)

    return D, x_df, y


def output_xlx(ws, i, data_name, ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score, depth):
    ws["A"+str(i+2)] = data_name
    ws["B"+str(i+2)] = ROCAUC_train_score
    ws["C"+str(i+2)] = ROCAUC_test_score
    ws["D"+str(i+2)] = BACC_train_score
    ws["E"+str(i+2)] = BACC_test_score
    ws["F"+str(i+2)] = depth

    return


def wright_columns(ws):
    ws["B1"] = "train score ROC/AOC"
    ws["C1"] = "test score ROC/AOC"
    ws["D1"] = "train score BACC"
    ws["E1"] = "test score BACC"
    ws["F1"] = "max depth"

    return


def wright_parameter(ws, rho_arg, theta_arg, n_arg):
    ws["G2"] = f"rho = {rho_arg}"
    ws["G3"] = f"theta = {theta_arg}"
    ws["G4"] = f"n_least = {n_arg}"

    return


def make_dir(now_time):
    y_m_d = now_time.strftime('%Y-%m-%d')
    p_file = pathlib.Path("outputfile/CV/" + y_m_d)
    p_file_ht = pathlib.Path("outputfile/CV/" + y_m_d + "/hyper_turning")
    p_file_ht_memo = pathlib.Path("outputfile/CV/" + y_m_d + "/ht_memo")
    p_file_for_test = pathlib.Path("outputfile/TEST/" + y_m_d)

    if not p_file.exists():
        p_file.mkdir()
    if not p_file_ht.exists():
        p_file_ht.mkdir()
    if not p_file_ht_memo.exists():
        p_file_ht_memo.mkdir()
    if not p_file_for_test.exists():
        p_file_for_test.mkdir()
    return y_m_d


def make_dir_for_test(now_time):
    y_m_d = now_time.strftime('%Y-%m-%d')
    p_file = pathlib.Path("outputfile/TEST/" + y_m_d)
    p_file_ht = pathlib.Path("outputfile/TEST/" + y_m_d + "/hyper_turning")
    p_file_ht_memo = pathlib.Path("outputfile/TEST/" + y_m_d + "/ht_memo")

    if not p_file.exists():
        p_file.mkdir()
    if not p_file_ht.exists():
        p_file_ht.mkdir()
    if not p_file_ht_memo.exists():
        p_file_ht_memo.mkdir()

    return y_m_d


def prepare_output_file_for_ht_memo():
    # 出力用のファイルを準備
    now_time = datetime.datetime.now()
    y_m_d = make_dir(now_time)
    date_time = now_time.strftime('%Y%m%d-%H%M%S')

    file_name = f"outputfile/CV/{y_m_d}/ht_memo/{date_time}.xlsx"

    return file_name


def prepare_output_file_for_sum():
    # 出力用のファイルを準備
    now_time = datetime.datetime.now()
    y_m_d = make_dir(now_time)
    date_time = now_time.strftime('%Y%m%d-%H%M%S')

    file_name = f"outputfile/CV/{y_m_d}/ht_memo/{date_time}_sum.xlsx"

    return file_name


def prepare_output_file_for_test():
    # 出力用のファイルを準備
    now_time = datetime.datetime.now()
    y_m_d = make_dir(now_time)
    date_time = now_time.strftime('%Y%m%d-%H%M%S')

    file_name = f"outputfile/TEST/{y_m_d}/{date_time}.xlsx"

    return file_name


def check_exist_dataset_for_test(fail1, fail2, fail3, fail4):
    if os.path.exists(fail1) == False or os.path.exists(fail2) == False or os.path.exists(
            fail3) == False or os.path.exists(fail4) == False:
        return False

    return True


def check_exist_dataset_for_cv(fail1, fail2):
    if os.path.exists(fail1) == False or os.path.exists(fail2) == False:
        return False

    return True
