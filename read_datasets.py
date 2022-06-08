import numpy as np
import pandas as pd


def read_data_list_for_cv():
    # df = pd.read_csv("dataset.csv")
    # INPUT_CSV = []
    # INPUT_TXT = []
    # l_or_s = ["large", "small"]
    # h_list = [50, 100, 200]
    # for i in range(len(df)):
    #     for size in l_or_s:
    #         # # all linear desc
    #         # INPUT_CSV.append("dataset/classification_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_var0_desc_norm.csv")
    #         # INPUT_TXT.append("dataset/classification_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) +"_values.txt")
    #
    #         for h in h_list:
    #             INPUT_CSV.append("dataset/classification_var0_5000_42922/" + str(df.iloc[-i, 0]) + "_" + str(size) + "_var0_quadratic_h" + str(h) + "_desc_norm.csv")
    #             INPUT_TXT.append("dataset/classification_var0_5000_42922/" + str(df.iloc[-i, 0]) + "_" + str(size) + "_values.txt")

    # INPUT_CSV = [
    #     "dataset/classification_var0_5000_42922/AhR_large_var0_desc_norm.csv",
    #     "dataset/classification_var0_5000_42922/ATAD5_large_var0_desc_norm.csv",
    #     "dataset/classification_var0_5000_42922/PPAR_gamma_small_var0_desc_norm.csv",
    #     "dataset/classification_var0_5000_42922/PTC_MR_large_var0_desc_norm.csv",
    #     "dataset/classification_var0_5000_42922/PTC_MR_small_var0_quadratic_h50_desc_norm.csv"
    # ]
    # INPUT_TXT = [
    #     "dataset/classification_var0_5000_42922/AhR_large_values.txt",
    #     "dataset/classification_var0_5000_42922/ATAD5_large_values.txt",
    #     "dataset/classification_var0_5000_42922/PPAR_gamma_small_values.txt",
    #     "dataset/classification_var0_5000_42922/PTC_MR_large_values.txt",
    #     "dataset/classification_var0_5000_42922/PTC_MR_small_values.txt"
    # ]
    INPUT_CSV = [
        "dataset/FV/MUTAG_3elem_desc_norm.csv",
        "dataset/FV/MUTAG_5elem_desc_norm.csv",
        "dataset/FV/MUTAG_allelem_desc_norm.csv",
    ]
    INPUT_TXT =[
        "dataset/FV/MUTAG_values.txt",
        "dataset/FV/MUTAG_values.txt",
        "dataset/FV/MUTAG_values.txt",
    ]
    return INPUT_CSV, INPUT_TXT


def read_data_list_for_test():
    df = pd.read_csv("dataset.csv")
    TRAIN_CSV = []
    TRAIN_TXT = []
    TEST_CSV = []
    TEST_TXT = []
    l_or_s = ["large", "small"]
    h_list = [50, 100, 200]
    for i in range(len(df)):
        for size in l_or_s:
            # # all linear descs
            # TRAIN_CSV.append("dataset/classification_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_var0_desc_norm.csv")
            # TEST_CSV.append("dataset/classification_test_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_var0_desc_norm.csv")
            # TRAIN_TXT.append("dataset/classification_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_values.txt")
            # TEST_TXT.append("dataset/classification_test_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_values.txt")

            for h in h_list:
                TRAIN_CSV.append("dataset/classification_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_var0_quadratic_h" + str(h) + "_desc_norm.csv")
                TEST_CSV.append("dataset/classification_test_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_var0_quadratic_h" + str(h) + "_desc_norm.csv")
                TRAIN_TXT.append("dataset/classification_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_values.txt")
                TEST_TXT.append("dataset/classification_test_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_values.txt")

    # TRAIN_CSV = [
    #     "dataset/FV/MUTAG_3elem_desc.csv",
    #     "dataset/FV/MUTAG_3elem_desc_norm.csv",
    #     "dataset/FV/MUTAG_5elem_desc.csv",
    #     "dataset/FV/MUTAG_5elem_desc_norm.csv",
    #     "dataset/FV/MUTAG_allelem_desc.csv",
    #     "dataset/FV/MUTAG_allelem_desc_norm.csv",
    # ]
    # TRAIN_TXT = [
    #     "dataset/classification_var0_5000_42922/AhR_large_values.txt",
    #     "dataset/classification_var0_5000_42922/ATAD5_large_values.txt",
    #     "dataset/classification_var0_5000_42922/PPAR_gamma_small_values.txt",
    #     "dataset/classification_var0_5000_42922/PTC_MR_large_values.txt",
    #     "dataset/classification_var0_5000_42922/PTC_MR_small_values.txt"
    # ]
    # TEST_CSV = [
    #     "dataset/classification_test_var0_5000_42922/AhR_large_var0_desc_norm.csv",
    #     "dataset/classification_test_var0_5000_42922/ATAD5_large_var0_quadratic_h_100_desc_norm.csv",
    #     "dataset/classification_test_var0_5000_42922/PPAR_gamma_small_h200_var0_desc_norm.csv",
    #     "dataset/classification_test_var0_5000_42922/PTC_MR_large_var0_desc_norm.csv",
    #     "dataset/classification_test_var0_5000_42922/PTC_MR_small_var0_quadratic_h50_desc_norm.csv"
    # ]
    # TEST_TXT = [
    #     "dataset/classification_test_var0_5000_42922/AhR_large_values.txt",
    #     "dataset/classification_test_var0_5000_42922/ATAD5_large_values.txt",
    #     "dataset/classification_test_var0_5000_42922/PPAR_gamma_small_values.txt",
    #     "dataset/classification_test_var0_5000_42922/PTC_MR_large_values.txt",
    #     "dataset/classification_test_var0_5000_42922/PTC_MR_small_values.txt"
    # ]

    return TRAIN_CSV, TRAIN_TXT, TEST_CSV, TEST_TXT