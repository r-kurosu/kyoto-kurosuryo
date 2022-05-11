Made by Kurosu 

## setting
- CPLEX PATH is determined 13b or 14l.
- install requirements (preparing now...)

---

## 1. test experiment
run dt_argorithm_test.py <br>
defolt dateset is all.

If you want to excute for 1 dataset, do follwing steps. <br>
1. TIMES := 1 (l.25)
2. setting `CSV_DATA` and `VALUE_DATA` (l.20~23)
3. setting main func -> # -experiment one dataset (l.461)
---



## 2. CV experiment
run dt_argolithm.py
defolt dataset is all.

If you want to excute for 1 dataset, do follwing steps. <br>
1. setting `INPUT_DATA, VALU_DATA, TEST_INPUT_DATA` and `TEST_VALUE_DATA` (l.19~22)
2. setting main func -> comment out  l. 437~447

