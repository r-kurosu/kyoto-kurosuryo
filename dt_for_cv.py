import statistics

import datetime as datetime
import numpy as np
import pandas as pd
import pulp
import time, copy, itertools, math, warnings, os
import openpyxl as excel
import datetime
import pathlib
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
import matplotlib.pyplot as plt

import dt_tools, read_datasets
import io, sys

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# warnings.simplefilter('ignore')
CPLEX_PATH = "/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex"

# マクロ定義
RHO = 0.2
THETA = 0
N_LEAST = 10
LAMBDA = 1
C = 100


TIMES = 10 # CVの回数（評価実験は10で行う）
SEED = 1000 # 予備実験:0, 評価実験: 1000


def test_main(INPUT_CSV, INPUT_TXT, cv_times, rho_arg, theta_arg, lambda_arg, c_arg):
    # 1. read data set
    data_csv = INPUT_CSV
    value_text = INPUT_TXT

    x, y = dt_tools.read_dataset(data_csv, value_text)
    x_df = x.reset_index(drop=True)
    y = pd.Series(y)
    CIDs = pd.Series(list(x.index))
    a_score = pd.Series([-1]*len(CIDs), index=list(CIDs))
    f_score = pd.Series([-1]*len(CIDs), index=list(CIDs))

    # (TIMES)回 5-fold回す
    auc_test_scores = []
    auc_train_scores = []
    bacc_test_scores = []
    bacc_train_scores = []
    train_depths = []
    st_time = time.time()
    for times in range(cv_times):
        # print("-----------------------------------------------")
        # print(f"{times+1}回目の交差実験")
        # 5-foldCVによる分析
        kf = KFold(n_splits=5, shuffle=True, random_state=times+SEED)

        for train_id, test_id in kf.split(x_df):
            x_train, x_test = x_df.iloc[train_id], x_df.iloc[test_id]
            y_train, y_test = y.iloc[train_id], y.iloc[test_id]
            CIDs_train, CIDs_test = CIDs.iloc[train_id], CIDs.iloc[test_id]
            a_score_train, a_score_test = a_score.iloc[train_id], a_score.iloc[test_id]
            f_score_train, f_score_test = a_score.iloc[train_id], a_score.iloc[test_id]

            # 2. construct a decision tree using hyper plane
            x_train = x_train.reset_index(drop=True)
            y_true_train = copy.deepcopy(y_train)
            y_train = y_train.reset_index(drop=True)
            CIDs_train.reset_index(drop=True, inplace=True)

            # マクロ変数
            D = len(y_train)  # データの数
            K = len(x_df.columns)  # ベクトルサイズ（記述示の数）
            N_least = N_LEAST  # 許容される最後のノードサイズ
            LAMBDA_arg: int = math.floor(D / lambda_arg)
            p = 0

            w_p = []
            b_p = []
            c_p_A = []
            c_p_B = []
            depths = []

            while D > N_least:
                p += 1
                print(f"n={D}", f"p={p}")
                print(f"(class 0: {(y_train == 0).sum()}, class 1: {(y_train == 1).sum()})")
                new_D, new_x_df, new_y = dt_tools.constructing_DT_based_HP(x_train, y_train, D, K, w_p, b_p, c_p_A, c_p_B, CIDs_train, a_score_train, f_score_train, rho_arg, theta_arg, LAMBDA_arg, c_arg)
                D = new_D
                x_train = new_x_df.reset_index(drop=True)
                y_train = new_y.reset_index(drop=True)
                CIDs_train.reset_index(drop=True, inplace=True)

                # if dt_tools.check_mono(y_train):
                #     break

            a_q = dt_tools.decision_a_q(y_train)
            a_score_train = dt_tools.set_a_q(a_q, a_score_train)
            f_score_train = dt_tools.set_a_q_for_f(y_train, f_score_train)
            train_depths.append(p)

            # 3. test ---------------------------------------------
            D = len(x_test)
            x_test = x_test.reset_index(drop=True)
            y_true_test = copy.deepcopy(y_test)
            y_test = y_test.reset_index(drop=True)
            CIDs_test.reset_index(drop=True, inplace=True)
            
            LAMBDA_arg: int = math.floor(D / lambda_arg)
            
            for p in range(len(b_p)):
                new_D, new_x_df, new_y = dt_tools.experiment_test(x_test, y_test, w_p[p], b_p[p], c_p_A[p], c_p_B[p], CIDs_test, a_score_test, f_score_test, rho_arg, theta_arg, LAMBDA_arg)
                D = new_D
                x_test = new_x_df.reset_index(drop=True)
                y_test = new_y.reset_index(drop=True)
                CIDs_test.reset_index(drop=True, inplace=True)
                # remain_data = (a_score_test == -1).sum()

                # if remain_data <= N_LEAST:
                #     break
                # if dt_tools.check_mono(y_test):
                #     break

            a_score_test = dt_tools.set_a_q(a_q, a_score_test)
            f_score_test = dt_tools.set_a_q_for_f(y_test, f_score_test)

            # 4. 結果 -------------------------------
            a_score_train = a_score_train.to_numpy()
            f_score_train = f_score_train.to_numpy()
            auc_train_score = roc_auc_score(y_true_train.tolist(), f_score_train.tolist())
            bacc_train_score = balanced_accuracy_score(y_true_train.tolist(), a_score_train.tolist())
            auc_train_scores.append(auc_train_score)
            bacc_train_scores.append(bacc_train_score)
            # print(f"ROC/AUC train score: {auc_train_score}")
            # print(f"BACC train score: {bacc_train_score}")

            a_score_test = a_score_test.to_numpy()
            f_score_test = f_score_test.to_numpy()
            auc_test_score = roc_auc_score(y_true_test.tolist(), f_score_test.tolist())
            bacc_test_score = balanced_accuracy_score(y_true_test.tolist(), a_score_test.tolist())
            auc_test_scores.append(auc_test_score)
            bacc_test_scores.append(bacc_test_score)

            # print(f"ROC/AUC test score: {auc_test_score}")
            # print(f"BACC test score: {bacc_test_score}")
            # -----------------------------------------
        # 5foldCV終了

    # 10回のCV終了
    ed_time = time.time()
    # ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score = 0, 0, 0, 0
    ROCAUC_train_score = statistics.median(auc_train_scores)
    ROCAUC_test_score = statistics.median(auc_test_scores)
    BACC_train_score = statistics.median(bacc_train_scores)
    BACC_test_score = statistics.median(bacc_test_scores)
    max_depth = max(train_depths)

    print("======================================================")
    print(data_csv)
    print(f"max depth : {max_depth}")
    print(f"ROC/AUC train score (median): {ROCAUC_train_score}")
    print(f"ROC/AUC test score (median): {ROCAUC_test_score}")
    print(f"BACC train score (median): {BACC_train_score}")
    print(f"BACC test score (median): {BACC_test_score}")
    print("計算時間 : {:.1f}".format(ed_time - st_time))
    print("======================================================")

    return ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score, max_depth


def main(rho_arg, theta_arg, lambda_arg, c_arg, INPUT_CSV, INPUT_TXT, cv_times):
    if not dt_tools.check_exist_dataset_for_cv(INPUT_CSV, INPUT_TXT):
        return

    # エクセルシートを用意
    # wbname = dt_tools.prepare_output_file_for_ht_memo()
    # wb = excel.Workbook()
    # ws = wb.active
    # dt_tools.wright_columns(ws)
    # dt_tools.wright_parameter(ws, rho_arg, theta_arg, N_LEAST)

    ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score, max_depth = test_main(INPUT_CSV, INPUT_TXT, cv_times, rho_arg, theta_arg, lambda_arg, c_arg)
    # dt_tools.output_xlx(ws, 1, INPUT_CSV, ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score, max_depth)
    # wb.save(wbname)

    return ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score, max_depth


if __name__ == "__main__":
    INPUT_CSV, INPUT_TXT = read_datasets.read_data_list_for_cv()
    # エクセルシートを用意
    wbname_all = dt_tools.prepare_output_file_for_sum()
    wb_all = excel.Workbook()
    ws_all = wb_all.active
    dt_tools.wright_columns(ws_all)
    dt_tools.wright_parameter(ws_all, RHO, THETA, N_LEAST)

    for i in range(len(INPUT_CSV)):
        print(INPUT_CSV[i])
        ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score, max_depth\
            = main(rho_arg=RHO,
             theta_arg=THETA,
             lambda_arg=1,
             c_arg = 1,
             INPUT_CSV=INPUT_CSV[i],
             INPUT_TXT=INPUT_TXT[i],
             cv_times=TIMES
             )
        dt_tools.output_xlx(ws_all, i, INPUT_CSV[i], ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score, max_depth)
        ws_all["A1"] = SEED

    wb_all.save(wbname_all)
