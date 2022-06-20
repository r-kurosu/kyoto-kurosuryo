import numpy as np
import pandas as pd


def read_data_list_for_cv():
    df = pd.read_csv("dataset.csv")
    # df = pd.read_csv("dataset2.csv")

    INPUT_CSV = []
    INPUT_TXT = []
    # size_list = ["allelem", "5elem", "3elem"]
    # size_list = ["allelem", "3elem"]
    size_list = ["3elem"]
    # size_list = ["allelem"]
    for i in range(len(df)):
        for size in size_list:
            INPUT_CSV.append("dataset/FV/" + str(df.iloc[i, 0]) + "_" + str(size) + "_desc_norm.csv")
            INPUT_TXT.append("dataset/FV/" + str(df.iloc[i, 0]) + "_values.txt")


    # INPUT_CSV = [
    #     "dataset/FV/MUTAG_3elem_desc_norm.csv",
    #     "dataset/FV/MUTAG_5elem_desc_norm.csv",
    #     "dataset/FV/MUTAG_allelem_desc_norm.csv",
    # ]
    # INPUT_TXT =[
    #     "dataset/FV/MUTAG_values.txt",
    #     "dataset/FV/MUTAG_values.txt",
    #     "dataset/FV/MUTAG_values.txt",
    # ]

    return INPUT_CSV, INPUT_TXT


def read_data_list_for_test():
    df = pd.read_csv("dataset.csv")
    TRAIN_CSV = []
    TRAIN_TXT = []
    TEST_CSV = []
    TEST_TXT = []
    # size_list = ["allelem", "5elem", "3elem"]
    size_list = ["allelem", "3elem"]
    for i in range(len(df)):
        for size in size_list:
            # # all linear descs
            # TRAIN_CSV.append("dataset/classification_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_var0_desc_norm.csv")
            # TEST_CSV.append("dataset/classification_test_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_var0_desc_norm.csv")
            # TRAIN_TXT.append("dataset/classification_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_values.txt")
            # TEST_TXT.append("dataset/classification_test_var0_5000_42922/" + str(df.iloc[i, 0]) + "_" + str(size) + "_values.txt")

            TRAIN_CSV.append("dataset/FV/" + str(df.iloc[i, 0]) + "_test_" + str(size) + "_var0_quadratic_h" + str(h) + "_desc_norm.csv")
            TEST_CSV.append("dataset/FV/" + str(df.iloc[i, 0]) + "_test_" + str(size) + "_var0_quadratic_h" + str(h) + "_desc_norm.csv")
            TRAIN_TXT.append("dataset/FV/" + str(df.iloc[i, 0]) + "_test_" + str(size) + "_values.txt")
            TEST_TXT.append("dataset/FV/" + str(df.iloc[i, 0]) + "_test_" + str(size) + "_values.txt")

    # TRAIN_CSV = [
    #     "dataset/FV/MUTAG_3elem_desc.csv",
    #     "dataset/FV/MUTAG_3elem_desc_norm.csv",
    #     "dataset/FV/MUTAG_5elem_desc.csv",
    #     "dataset/FV/MUTAG_5elem_desc_norm.csv",
    #     "dataset/FV/MUTAG_allelem_desc.csv",
    #     "dataset/FV/MUTAG_allelem_desc_norm.csv",
    # ]

    return TRAIN_CSV, TRAIN_TXT, TEST_CSV, TEST_TXT