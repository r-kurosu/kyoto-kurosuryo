import statistics

import numpy as np
import pandas as pd
import pulp
import time, copy, itertools, math, warnings, os
import openpyxl as excel
import datetime
import pathlib
import read_datasets

from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import io, sys
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import dt_tools, read_datasets

CPLEX_PATH = "/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex"

# マクロ定義
RHO = 0.05
THETA = 0.1
N_LEAST = 10


def test_main(INPUT_CSV: str,
              INPUT_TXT: str,
              INPUT_TEST_CSV: str,
              INPUT_TEST_TXT: str,
              rho_arg,
              theta_arg
              ):
    # 1. read data set
    # train用
    x, y = dt_tools.read_dataset(INPUT_CSV, INPUT_TXT)
    x_df = x.reset_index(drop=True)
    y = pd.Series(y)
    CIDs = pd.Series(list(x.index))
    a_score = pd.Series([-1]*len(CIDs), index=list(CIDs))

    x_train = x.reset_index(drop=True)
    y_true_train = copy.deepcopy(y)
    y_train = y.reset_index(drop=True)

    # test用
    x_test_input, y_test_input = dt_tools.read_dataset(INPUT_TEST_CSV, INPUT_TEST_TXT)
    y_test = pd.Series(y_test_input)
    CIDs_test = pd.Series(list(x_test_input.index))
    a_score_test = pd.Series([-1] * len(CIDs_test), index=list(CIDs_test))

    x_test = x_test_input.reset_index(drop=True)
    y_true_test = copy.deepcopy(y_test)
    y_test = y_test.reset_index(drop=True)


    # 2. construct a decision tree using hyper plane
    # マクロ変数
    D = len(x_train)  # データの数
    K = len(x_train.columns)  # ベクトルサイズ（記述示の数）
    N_least = N_LEAST  # 許容
    p = 0

    w_p = []
    b_p = []
    c_p_A = []
    c_p_B = []

    while D > N_least:
        p += 1
        print(f"|D|={D}", f"p={p}")
        new_D, new_x_df, new_y = dt_tools.constructing_DT_based_HP(x_train, y_train, D, K, w_p, b_p, c_p_A, c_p_B, CIDs, a_score, rho_arg, theta_arg)
        D = new_D
        x_train = new_x_df.reset_index(drop=True)
        y_train = new_y.reset_index(drop=True)
        CIDs.reset_index(drop=True, inplace=True)
    q = p + 1
    a_score = dt_tools.set_a_q(x_train, y, CIDs, a_score)

    # 3. test ---------------------------------------------
    D = len(x_test)
    for p in range(len(b_p)):
        print(D)
        new_D, new_x_df, new_y = dt_tools.experiment_test(x_test, y_test, w_p[p], b_p[p], CIDs_test, a_score_test, rho_arg, theta_arg)
        D = new_D
        x_test = new_x_df.reset_index(drop=True)
        y_test = new_y.reset_index(drop=True)
        CIDs.reset_index(drop=True, inplace=True)

    a_score_test = dt_tools.set_a_q(x_test, y_test, CIDs_test, a_score_test)

    # 4. 結果
    a_score_train = a_score.to_numpy()
    y_true_train = y_true_train.to_numpy()
    ROCAUC_train_score = roc_auc_score(y_true_train, a_score_train)
    BACC_train_score = balanced_accuracy_score(y_true_train, a_score_train)

    a_score_test = a_score_test.to_numpy()
    ROCAUC_test_score = roc_auc_score(y_true_test, a_score_test)
    BACC_test_score = balanced_accuracy_score(y_true_test, a_score_test)

    # print("======================================================")
    # print(train_data)
    print(f"ROC/AUC train score : {ROCAUC_train_score}")
    print(f"ROC/AUC test score : {ROCAUC_test_score}")

    return ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score


def main(rho_arg, theta_arg, INPUT_CSV, INPUT_TXT, INPUT_TEST_CSV, INPUT_TEST_TXT):
    if not dt_tools.check_exist_dataset_for_test(INPUT_CSV, INPUT_TXT, INPUT_TEST_CSV, INPUT_TEST_TXT):
        return 0, 0, 0, 0

    # エクセルシートを用意
    wbname = dt_tools.prepare_output_file_for_test()
    wb = excel.Workbook()
    ws = wb.active
    dt_tools.wright_columns(ws)
    dt_tools.wright_parameter(ws, rho_arg, theta_arg, N_LEAST)

    train_score1, test_score1, bacc_train_score1, bacc_test_score1 = test_main(INPUT_CSV=INPUT_CSV,
                                                                               INPUT_TXT=INPUT_TXT,
                                                                               INPUT_TEST_CSV=INPUT_TEST_CSV,
                                                                               INPUT_TEST_TXT=INPUT_TEST_TXT,
                                                                               rho_arg=rho_arg,
                                                                               theta_arg=theta_arg)
    train_score2, test_score2, bacc_train_score2, bacc_test_score2 = test_main(INPUT_CSV=INPUT_TEST_CSV,
                                                                               INPUT_TXT=INPUT_TEST_TXT,
                                                                               INPUT_TEST_CSV=INPUT_CSV,
                                                                               INPUT_TEST_TXT=INPUT_TXT,
                                                                               rho_arg=rho_arg,
                                                                               theta_arg=theta_arg)

    ROCAUC_train_score = statistics.median([train_score1, train_score2])
    ROCAUC_test_score = statistics.median([test_score1, test_score2])
    BACC_train_score = statistics.median([bacc_train_score1, bacc_train_score2])
    BACC_test_score = statistics.median([bacc_test_score1, bacc_test_score2])

    dt_tools.output_xlx(ws, 1, INPUT_CSV, ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score)
    wb.save(wbname)

    return ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score


if __name__ == "__main__":
    INPUT_CSV, INPUT_TXT, INPUT_TEST_CSV, INPUT_TEST_TXT = read_datasets.read_data_list_for_test()
    for i in range(len(INPUT_CSV)):
        main(rho_arg=RHO,
             theta_arg=THETA,
             INPUT_CSV=INPUT_CSV[i],
             INPUT_TXT=INPUT_TXT[i],
             INPUT_TEST_CSV=INPUT_TEST_CSV[i],
             INPUT_TEST_TXT=INPUT_TEST_TXT[i]
             )
