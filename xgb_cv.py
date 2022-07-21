import statistics

import numpy as np
import pandas as pd
import time, copy, itertools, math, warnings, os
import openpyxl as excel
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, roc_curve
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import dt_tools, read_datasets

# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# warnings.simplefilter('ignore')
CPLEX_PATH = "/Applications/CPLEX_Studio221/cplex/bin/x86-64_osx/cplex"

# マクロ定義
TIMES = 10 # CVの回数（評価実験は10で行う）
SEED = 1000 # 予備実験:0, 評価実験: 1000


def use_xgboost(INPUT_CSV, INPUT_TXT, cv_times):
    # 1. read data set
    data_csv = INPUT_CSV
    value_text = INPUT_TXT

    x, y = dt_tools.read_dataset(data_csv, value_text)
    x_df = x.reset_index(drop=True)
    y = pd.Series(y)

    # (TIMES)回 5-fold回す
    auc_test_scores = []
    auc_train_scores = []
    bacc_test_scores = []
    bacc_train_scores = []
    st_time = time.time()

    for times in range(cv_times):
        # print("-----------------------------------------------")
        # print(f"{times+1}回目の交差実験")
        kf = KFold(n_splits=5, shuffle=True, random_state=times+SEED)

        for train_id, test_id in kf.split(x_df):
            x_train, x_test = x_df.iloc[train_id], x_df.iloc[test_id]
            y_train, y_test = y.iloc[train_id], y.iloc[test_id]

            # 学習
            xgb_train = xgb.DMatrix(x_train, label=y_train)
            xgb_test = xgb.DMatrix(x_test, label=y_test)
            param = {
                # 1. 全体パラメータ
                # "silent": 0,
                
                # 2.ブースターパラメータ
                "max_depth": 6,  # デフォルト6
                "min_child_weight": 1,  # デフォルト1
                "eta": 0.1,  # 0.01~0.2が多いらしい
                "tree_method": "exact",
                "predictor": "cpu_predictor",
                "lambda": 1,  # 重みに関するL"正則 デフォルト1
                "alpha": 0,  # 重みに関するL1正則  # デフォルト0

                # 3. 学習タスクパラメータ
                'objective': 'binary:logistic',
                # "objective": "reg:linear",
                "eval_metric": "rmse",  # 損失関数 l(y, a)
                "seed": 0
            }
            model = xgb.train(
                param,
                xgb_train,
                num_boost_round=1
            )

            y_pred_proba_train = model.predict(xgb_train)
            y_pred_train = np.where(y_pred_proba_train > 0.5, 1, 0)
            auc_train_score = roc_auc_score(y_train, y_pred_proba_train)
            bacc_train_score = balanced_accuracy_score(y_train, y_pred_train)

            y_pred_proba = model.predict(xgb_test)
            y_pred = np.where(y_pred_proba > 0.5, 1, 0)
            auc_test_score = roc_auc_score(y_test, y_pred_proba)
            bacc_test_score = balanced_accuracy_score(y_test, y_pred)

            # 4. 結果 -------------------------------
            auc_train_scores.append(auc_train_score)
            bacc_train_scores.append(bacc_train_score)
            # print(f"ROC/AUC train score: {auc_train_score}")
            # print(f"BACC train score: {bacc_train_score}")

            auc_test_scores.append(auc_test_score)
            bacc_test_scores.append(bacc_test_score)
            # print(f"ROC/AUC test score: {auc_test_score}")
            # print(f"BACC test score: {bacc_test_score}")
            # -----------------------------------------
        # 5foldCV終了
    # 10回のCV終了
    ed_time = time.time()

    ROCAUC_train_score = statistics.median(auc_train_scores)
    ROCAUC_test_score = statistics.median(auc_test_scores)
    BACC_train_score = statistics.median(bacc_train_scores)
    BACC_test_score = statistics.median(bacc_test_scores)

    print("======================================================")
    print(data_csv)
    print(f"ROC/AUC train score (median): {ROCAUC_train_score}")
    print(f"ROC/AUC test score (median): {ROCAUC_test_score}")
    print(f"BACC train score (median): {BACC_train_score}")
    print(f"BACC test score (median): {BACC_test_score}")
    print("計算時間 : {:.1f}".format(ed_time - st_time))
    print("======================================================")

    return ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score


def main(INPUT_CSV, INPUT_TXT, cv_times):
    if not dt_tools.check_exist_dataset_for_cv(INPUT_CSV, INPUT_TXT):
        return

    ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score= use_xgboost(INPUT_CSV, INPUT_TXT, cv_times)

    return ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score


if __name__ == "__main__":
    INPUT_CSV, INPUT_TXT = read_datasets.read_data_list_for_cv()
    # エクセルシートを用意
    wbname_all = dt_tools.prepare_output_file_for_sum()
    wb_all = excel.Workbook()
    ws_all = wb_all.active
    dt_tools.wright_columns(ws_all)
    # dt_tools.wright_parameter(ws_all, RHO, THETA, N_LEAST)

    for i in range(len(INPUT_CSV)):
        print(INPUT_CSV[i])
        ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score\
            = main(
             INPUT_CSV=INPUT_CSV[i],
             INPUT_TXT=INPUT_TXT[i],
             cv_times=TIMES
             )
        dt_tools.output_xlx(ws_all, i, INPUT_CSV[i], ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score, depth=0)
        ws_all["A1"] = SEED

    wb_all.save(wbname_all)
