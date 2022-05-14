Made by Kurosu 

## Setting
- CPLEX PATH is determined 13b or 14l.
- install requirements (preparing now...)
---
##Output
CV: `outputfile/CV/yyyy-mm-dd/yyyymmddhhmmss.xlsx` に出力<br>
test: `outputfile/TEST/yyyy-mm-dd/yyyymmddhhmmss.xlsx`に出力<br>

評価項目は以下の4つ
- train score(ROC/AUC)
- test score(ROC/AUC)
- train score(BACC)
- test score(BACC)
---

## 1. test experiment
run dt_algorithm_test.py <br>
default dateset is all.

If you want to execute for 1 dataset, do follow steps. <br>
1. TIMES := 1 (l.25)
2. setting `CSV_DATA` and `VALUE_DATA` (l.20~23)
3. setting main func -> # -experiment one dataset (l.461)
---



## 2. CV experiment
run dt_algorithm.py
default dataset is all.

If you want to execute for 1 dataset, do follow steps. <br>
1. setting `INPUT_DATA, VALU_DATA, TEST_INPUT_DATA` and `TEST_VALUE_DATA` (l.19~22)
2. setting main func -> comment out  l. 437~447

