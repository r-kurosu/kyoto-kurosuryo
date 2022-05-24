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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

import dt_tools, read_datasets
import io, sys
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# warnings.simplefilter('ignore')
CPLEX_PATH = "/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex"

# マクロ定義
RHO = 0.05
THETA = 0.1
N_LEAST = 10


TIMES = 2 # CVの回数（実験は10で行う）
SEED = 0


def test_main(INPUT_CSV, INPUT_TXT, cv_times, rho_arg, theta_arg):
    # 1. read data set
    data_csv = INPUT_CSV
    value_text = INPUT_TXT

    x, y = dt_tools.read_dataset(data_csv, value_text)
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
    test_scores_bacc = []
    train_scores_bacc = []
    st_time = time.time()
    for times in range(cv_times):
        # print("-----------------------------------------------")
        # print(f"{times+1}回目の交差実験")
        # 5-foldCVによる分析
        kf = KFold(n_splits=5, shuffle=True, random_state=times+SEED)
        # kf = KFold(n_splits=5, shuffle=True, random_state=times+1000)

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
            depths  = []

            while D > N_least:
                p += 1
                print(f"|D|={D}", f"p={p}")
                new_D, new_x_df, new_y = dt_tools.constructing_DT_based_HP(x_train, y_train, D, K, w_p, b_p, c_p_A, c_p_B, CIDs_train, a_score_train, rho_arg, theta_arg)
                D = new_D
                x_train = new_x_df.reset_index(drop=True)
                y_train = new_y.reset_index(drop=True)
                CIDs_train.reset_index(drop=True, inplace=True)
            q = p+1

            a_score_train = dt_tools.set_a_q(x_train, y_train, CIDs_train, a_score_train)
            depths.append(len(b_p))


            # 3. test ---------------------------------------------
            D = len(x_test)
            x_test = x_test.reset_index(drop=True)
            y_true_test = copy.deepcopy(y_test)
            y_test = y_test.reset_index(drop=True)
            CIDs_test.reset_index(drop=True, inplace=True)

            # print(len(b_p))
            for p in range(len(b_p)):
                new_D, new_x_df, new_y = dt_tools.experiment_test(x_test, y_test, w_p[p], b_p[p], CIDs_test, a_score_test, rho_arg, theta_arg)
                D = new_D
                x_test = new_x_df.reset_index(drop=True)
                y_test = new_y.reset_index(drop=True)
                CIDs_test.reset_index(drop=True, inplace=True)

            a_score_test = dt_tools.set_a_q(x_test, y_test, CIDs, a_score_test)

            # 4. 結果 -------------------------------
            a_score_train = a_score_train.to_numpy()
            train_score = roc_auc_score(y_true_train, a_score_train)
            bacc_train_score = balanced_accuracy_score(y_true_train, a_score_train)
            train_scores.append(train_score)
            train_scores_bacc.append(bacc_train_score)
            # print(f"ROC/AUC train score: {train_score}")
            # print(f"BACC train score: {bacc_train_score}")

            a_score_test = a_score_test.to_numpy()
            test_score = roc_auc_score(y_true_test, a_score_test)
            bacc_test_score = balanced_accuracy_score(y_true_test, a_score_test)
            test_scores.append(test_score)
            test_scores_bacc.append(bacc_test_score)
            # print(f"ROC/AUC test score: {test_score}")
            # print(f"BACC test score: {bacc_test_score}")
            # -----------------------------------------
        # 5foldCV終了

    # 10回のCV終了
    ed_time = time.time()
    ROCAUC_train_score = statistics.median(train_scores)
    ROCAUC_test_score = statistics.median(test_scores)
    BACC_train_score = statistics.median(train_scores_bacc)
    BACC_test_score = statistics.median(test_scores_bacc)

    print("======================================================")
    print(data_csv)
    print(f"max depth : {max(depths)}")
    print(f"ROC/AUC train score (median): {ROCAUC_train_score}")
    print(f"ROC/AUC test score (median): {ROCAUC_test_score}")
    print(f"ROC/AUC train score (median): {BACC_train_score}")
    print(f"ROC/AUC test score (median): {BACC_test_score}")
    print("計算時間 : {:.1f}".format(ed_time - st_time))
    print("======================================================")

    return ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score


def main(rho_arg, theta_arg, INPUT_CSV, INPUT_TXT, cv_times):
    if not dt_tools.check_exist_dataset_for_cv(INPUT_CSV, INPUT_TXT):
        return

    # エクセルシートを用意
    wbname = dt_tools.prepare_output_file_for_ht_memo()
    wb = excel.Workbook()
    ws = wb.active
    dt_tools.wright_columns(ws)
    dt_tools.wright_parameter(ws, rho_arg, theta_arg, N_LEAST)

    ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score = test_main(INPUT_CSV, INPUT_TXT, cv_times, rho_arg, theta_arg)
    dt_tools.output_xlx(ws, 1, INPUT_CSV, ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score)
    wb.save(wbname)

    return ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score


if __name__ == "__main__":
    INPUT_CSV, INPUT_TXT = read_datasets.read_data_list_for_cv()
    # エクセルシートを用意
    wbname_all = dt_tools.prepare_output_file_for_ht_memo()
    wb_all = excel.Workbook()
    ws_all = wb_all.active
    dt_tools.wright_columns(ws_all)
    dt_tools.wright_parameter(ws_all, RHO, THETA, N_LEAST)
    for i in range(len(INPUT_CSV)):
        ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score\
            = main(rho_arg=RHO,
             theta_arg=THETA,
             INPUT_CSV=INPUT_CSV[i],
             INPUT_TXT=INPUT_TXT[i],
             cv_times=TIMES
             )
        dt_tools.output_xlx(ws_all, 1, INPUT_CSV[i], ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score)

    wb_all.save(wbname_all)
