import statistics

import numpy as np
import pandas as pd
import pulp
import time, sys, copy, itertools, math, warnings, os

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score


warnings.simplefilter('ignore')
CPLEX_PATH = "/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex"

# マクロ定義
RHO = 0.05
THETA = 0.1
N_LEAST = 10
# CSV_DATA = "dataset/AhR_large_var0_quadratic_h5000_desc_norm.csv"
# VALUE_DATA = "dataset/AhR_large_values.txt"
CSV_DATA = "dataset/AhR_large_var0_quadratic_h25000_desc_norm.csv"
VALUE_DATA = "dataset/AhR_large_values.txt"

TIMES = 1 # CVの回数（実験は10で行う）



def read_data_list():
    df = pd.read_csv("dataset.csv")
    INPUT_CSV = []
    INPUT_TXT = []
    l_or_s = ["large", "small"]
    h_list = [50, 100, 200]
    for i in range(len(df)):
        for size in l_or_s:
            for h in h_list:
                INPUT_CSV.append("dataset/classification_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_var0_quadratic_h" + str(h) + "_desc_norm.csv")
                INPUT_TXT.append("dataset/classification_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_values.txt")

    return INPUT_CSV, INPUT_TXT


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
            sys.stderr.write('error: {} misses the target value of CID {}\n'.format(target_values_filename, cid))
            exit(1)

    y = np.array([target_dict[cid] for cid in CIDs])
    return x, y


def find_separator(x_df, y, D, K, w_p, b_p, CIDs):
    model = pulp.LpProblem("Linear_Separator", pulp.LpMinimize)
    # 変数定義
    b = pulp.LpVariable("b", -1, 1, cat=pulp.LpContinuous)
    w = [pulp.LpVariable("w_{}".format(i), -1, 1, cat=pulp.LpContinuous) for i in range(K)]
    eps = pulp.LpVariable('eps', cat=pulp.LpContinuous)
    # 目的関数
    model += eps
    # 制約条件
    for i in range(D):
        if y.loc[i] == 0:
            model += pulp.lpDot(w, x_df.loc[i]) - b <= -1 + eps
        else:
            model += pulp.lpDot(w, x_df.loc[i]) - b >= 1 - eps

    status = model.solve(pulp.CPLEX(path=CPLEX_PATH, msg=0))

    # 出力
    if status == pulp.LpStatusOptimal:
        w_ast = [w[i].value() for i in range(len(w))]
        b_ast = b.value()
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


def use_sklearn(x, y):
    # 2.1データ整理
    columns_list = x.columns
    X = x[columns_list]
    Y = pd.DataFrame(y)

    # # 2.2学習
    model = LinearRegression()
    model.fit(X, Y)
    w = []
    for i in range(len(columns_list)):
        w.append(model.coef_[0][i])
    b = model.intercept_
    print('coefficient = ', model.coef_[0])  # 説明変数の係数を出力
    print('intercept = ', model.intercept_)  # 切片を出力
    print(w)

    return w, b


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
    # print(x_df)
    # print(y)
    # print(w)
    #TODO: eps<1の時のバグ対処----
    # if len(x_df) == 0:
    #     print(x_df)
    #     return z, z_p
    # -------------------------
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
    r, r_p = -1, -1
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
            # print('all z is smaller than 0')
            break

    return r, r_p


def find_index_l(z, r, z_p, r_p, s, s_p):
    rho = RHO
    theta = THETA
    for l in range(r, 0, -1):
        if siki_1(l, z, z_p, r, r_p, s, s_p, rho) == True:
            # print(f"l={l}")
            # print(f"r={r}")
            c_A = -z[l]
            break
        c_A = -z[math.floor(r*theta)]

    for l in range(r_p, 0, -1):
        if siki_2(l, z, z_p, r, r_p, s, s_p, rho) == True:
            # print(f"l={l}")
            # print(f"r\'={r_p}")
            c_B = -z_p[l]  # TODO: type miss???
            break
        c_B = -z_p[math.floor(r_p*theta)]

    if r <= 0:
        c_A = 0
    if r_p <= 0:
        c_B = 0

    return c_A, c_B


def siki_1(l, z, z_p, r, r_p, s, s_p, rho):
    temp = 0 # |l|
    for j in range(r_p + 1, s_p):
        if z_p[j] <= z[l]:
            temp += 1
    if temp / l <= rho * s_p / s:
        return True

    return False


def siki_2(l, z, z_p, r, r_p, s, s_p, rho):
    temp = 0
    for j in range(r + 1, s):
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


def experiment_test(x_test, y_test, w, b, CIDs_test, a_score_test):
    # 3.1 s, s'を数える
    s, s_p = count_s(y_test)
    # 3.2 ソートする
    z, z_p = sort_dataset(x_test, y_test, w, b)
    # 3.3 r, r'を探す
    r, r_p = find_r(z, z_p)
    # 3.4 index(l)を探す
    c_A, c_B = find_index_l(z, r, z_p, r_p, s, s_p)
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


def calc_a_q(D, K, a, x):
    a_q = [0]*K
    for d in range(K):
        temp_A = 0
        temp_B = 0
        # for i in range(D):
        #     if a[i] == 0 and x[i][d] isinD_q:
        #         temp_A += 1
        #     elif a[i] == 1 and x[i][d] isinD_q:
        #         temp_B += 1
        if temp_A > temp_B:
            a_q[d] = 0
        elif temp_A < temp_B:
            a_q[d] = 1

    return a_q


def constructing_DT_based_HP(x_df, y, D, K, w_p, b_p, c_p_A, c_p_B, CIDs, a_score):
    # 2. 超平面（hyper plane）を探す
    # w, b = use_sklearn(x, y)
    w, b, eps = find_separator(x_df, y, D, K, w_p, b_p, CIDs)
    # print(f"w={w}")
    # print(f"b={b}")
    # print(f"eps={eps}")

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
    c_A, c_B = find_index_l(z, r, z_p, r_p, s, s_p)
    c_p_A.append(c_A)
    c_p_B.append(c_B)
    # print(c_A, c_B)

    # 3.5 関数Φとデータセットの再定義
    D, x_df, y = redefine_func(x_df, y, w, b, c_A, c_B, CIDs, a_score)

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
        a_score[CIDs.loc[index]] = value

    return


def test_main(INPUT_CSV, INPUT_TXT):
    # 1. read data set
    data_csv = INPUT_CSV
    value_text = INPUT_TXT

    x, y = read_dataset(data_csv, value_text)
    x_df = x.reset_index(drop=True)
    y = pd.Series(y)
    CIDs = pd.Series(list(x.index))
    a_score = pd.Series([-1]*len(CIDs), index=list(CIDs))
    # print(x_df)
    # print(x)
    # print(CIDs)

    # (TIMES)回 5-fold回す
    test_scores = []
    train_scores = []
    st_time = time.time()
    for times in range(TIMES):
        # print("-----------------------------------------------")
        # print(f"{times+1}回目の交差実験")
        # 5-foldCVによる分析
        kf = KFold(n_splits=5, shuffle=True)
        ROC_AOC_scores_train = []
        ROC_AOC_scores_test = []

        for train_id, test_id in kf.split(x_df):
            x_train, x_test = x_df.iloc[train_id], x_df.iloc[test_id]
            y_train, y_test = y.iloc[train_id], y.iloc[test_id]
            CIDs_train, CIDs_test = CIDs.iloc[train_id], CIDs.iloc[test_id]
            a_score_train, a_score_test = a_score.iloc[train_id], a_score.iloc[test_id]

            # 2. construct a decision tree using hyper plane
            x_train = x_train.reset_index(drop=True)
            y_true_train = copy.deepcopy(y_train)
            y_train = y_train.reset_index(drop=True)
            CIDs_train.reset_index(drop=True, inplace=True)

            # マクロ変数
            D = len(y_train)  # データの数
            K = len(x_df.columns)  # ベクトルサイズ（記述示の数）
            N_least = N_LEAST  # 許容
            p = 0

            w_p = []
            b_p = []
            c_p_A = []
            c_p_B = []

            while D > N_least:
                p += 1
                print(f"|D|={D}", f"p={p}")
                new_D, new_x_df, new_y = constructing_DT_based_HP(x_train, y_train, D, K, w_p, b_p, c_p_A, c_p_B, CIDs_train, a_score_train)
                D = new_D
                x_train = new_x_df.reset_index(drop=True)
                y_train = new_y.reset_index(drop=True)
                CIDs_train.reset_index(drop=True, inplace=True)
            q = p+1

            set_a_q(x_train, y_train, CIDs_train, a_score_train)

            # 3. test ---------------------------------------------
            D = len(x_test)
            x_test = x_test.reset_index(drop=True)
            y_true_test = copy.deepcopy(y_test)
            y_test = y_test.reset_index(drop=True)
            CIDs_test.reset_index(drop=True, inplace=True)

            # print(len(b_p))
            for p in range(len(b_p)):
                new_D, new_x_df, new_y = experiment_test(x_test, y_test, w_p[p], b_p[p], CIDs_test, a_score_test)
                D = new_D
                #TODO train->testに変えた。
                x_test = new_x_df.reset_index(drop=True)
                y_test = new_y.reset_index(drop=True)
                CIDs_test.reset_index(drop=True, inplace=True)

            set_a_q(x_test, y_test, CIDs_test, a_score_test)

            # 4. 結果
            a_score_train = a_score_train.to_numpy()
            y_train = y_train.to_numpy()
            train_score = roc_auc_score(y_true_train, a_score_train)
            ROC_AOC_scores_train.append(train_score)
            train_scores.append(train_score)
            print(f"ROC/AUC train score: {train_score}")

            a_score_test = a_score_test.to_numpy()
            y_test = y_test.to_numpy()
            test_score = roc_auc_score(y_true_test, a_score_test)
            ROC_AOC_scores_test.append(test_score)
            test_scores.append(test_score)
            print(f"ROC/AUC test score: {test_score}")

        # 5foldCV終了
        print(f"ROC AUC train score: {statistics.median(ROC_AOC_scores_train)}, {ROC_AOC_scores_train}")
        print(f"ROC AUC test score: {statistics.median(ROC_AOC_scores_test)}, {ROC_AOC_scores_test}")

    # 10回のCV終了
    ed_time = time.time()
    print("======================================================")
    print(data_csv)
    print(f"train score : {statistics.median(train_scores)}")
    print(f"test score : {statistics.median(test_scores)}")
    print("計算時間 : {:.1f}".format(ed_time - st_time))
    print("======================================================")

    return


def main():
    ## 1. experiment all datasets
    INPUT_CSV, INPUT_TXT = read_data_list()
    for i in reversed(range(len(INPUT_CSV))):
        print("↓ " + str(INPUT_CSV[i]))
        test_main(INPUT_CSV[i], INPUT_TXT[i])

    ## 2. experiment one dataset
    # print(f"experiment {CSV_DATA}")
    # test_main(CSV_DATA, VALUE_DATA)

    return


if __name__ == "__main__":
    main()
