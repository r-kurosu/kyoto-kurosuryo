import dt_algorithm_test
import pathlib
import openpyxl as excel
import datetime
import os

rho_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09]
theta_list = [0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.30, 0.5]


def wright_columns(ws):
    for i in range(len(rho_list)):
        ws.cell(row=1, column=i + 2).value = rho_list[i]
    for i in range(len(theta_list)):
        ws.cell(row=i + 2, column=1).value = theta_list[i]

    return


def make_dir(now_time):
    y_m_d = now_time.strftime('%Y-%m-%d')
    p_file = pathlib.Path("outputfile/TEST/" + y_m_d + "/hyper_turning")

    if not p_file.exists():
        p_file.mkdir()

    return y_m_d


def prepare_output_file():
    # 出力用のファイルを準備
    now_time = datetime.datetime.now()
    y_m_d = make_dir(now_time)
    date_time = now_time.strftime('%Y%m%d%H%M%S')

    file_name = f"outputfile/TEST/{y_m_d}/hyper_turning/{date_time}_ht_test.xlsx"

    return file_name


def main():
    # エクセルシートを用意
    wb_name = prepare_output_file()
    wb = excel.Workbook()
    ws = wb.active
    wright_columns(ws)
    # --
    for i, rho in enumerate(rho_list):
        for j, theta in enumerate(theta_list):
            ROCAUC_train_score, ROCAUC_test_score, BACC_train_score, BACC_test_score = dt_algorithm_test.main(rho, theta)
            ws.cell(row=i + 2, column=i + 2).value = ROCAUC_test_score

    wb.save(wb_name)

    return


if __name__ == "__main__":
    main()
