import time

import dt_algorithm, dt_temp
import pathlib
import openpyxl as excel
import datetime
import time
import os
import sys
import io

# sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

rho_list = [0.01, 0.03, 0.04, 0.05, 0.06, 0.08, 0.1]
theta_list = [0.01, 0.05, 0.08, 0.1, 0.2, 0.30, 0.5]


def wright_columns(ws, dataset):
    for i in range(len(rho_list)):
        ws.cell(row=1, column=i + 2).value = rho_list[i]
    for i in range(len(theta_list)):
        ws.cell(row=i + 2, column=1).value = theta_list[i]
    ws.cell(row=12, column=1).value = "theta"
    ws.cell(row=1, column=12).value = "rho"
    ws.cell(row=13, column=2).value = dataset
    return


def make_dir(now_time):
    y_m_d = now_time.strftime('%Y-%m-%d')
    p_file = pathlib.Path("outputfile/CV/" + y_m_d)
    p_file_ht = pathlib.Path("outputfile/CV/" + y_m_d + "/hyper_turning")
    p_file_ht_memo = pathlib.Path("outputfile/CV/" + y_m_d + "/ht_memo")

    if not p_file.exists():
        p_file.mkdir()
    if not p_file_ht.exists():
        p_file_ht.mkdir()
    if not p_file_ht_memo.exists():
        p_file_ht_memo.mkdir()
    return y_m_d


def prepare_output_file(data_name):
    # 出力用のファイルを準備
    now_time = datetime.datetime.now()
    y_m_d = make_dir(now_time)
    date_time = now_time.strftime('%Y%m%d-%H%M%S')

    file_name = f"outputfile/CV/{y_m_d}/hyper_turning/{date_time}_ht_cv_{data_name}.xlsx"

    return file_name


def main():
    INPUT_CSV, INPUT_TXT = dt_temp.read_data_list()
    wb_name = [0]*len(INPUT_CSV)

    for k, data in enumerate(INPUT_CSV):
        # エクセルシートを用意
        wb_name[k] = prepare_output_file(data_name=data)
        wb = excel.Workbook()
        ws = wb.active
        ws.title = "test score"
        wright_columns(ws, data)
        wb.create_sheet(title="train_score")
        ws_train_score = wb["train_score"]
        wright_columns(ws_train_score, data)
        score_data_test = [[0]*len(rho_list)]*len(theta_list)
        score_data_train = [[0]*len(rho_list)]*len(theta_list)
        # --
        for i, rho in enumerate(rho_list):
            for j, theta in enumerate(theta_list):
                st_time = time.time()
                print(f"rho:{rho},theta:{theta}")
                ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score = dt_temp.main(rho, theta, INPUT_CSV[k], INPUT_TXT[k])
                ws.cell(row=j + 2, column=i + 2).value = ROCAUC_test_score
                ws_train_score.cell(row=j + 2, column=i + 2).value = ROCAUC_train_score
                score_data_test[i][j] = ROCAUC_test_score
                score_data_train[i][j] = ROCAUC_train_score
                ed_time = time.time()
                print("time {:.1f}".format(ed_time - st_time))
        wb.save(wb_name[k])
        print(score_data_train)
        print(score_data_test)
    return


if __name__ == "__main__":
    main()
