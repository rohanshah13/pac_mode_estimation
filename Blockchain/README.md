# Blockchain Application

First, use the following command to install dependencies (assuming Python 3):
```
$ pip3 install -r requirements.txt
```
1) To check for error plot, run 

```
$ python3 compare_error.py
```

2) To check for sample complexity, run 

```
$ python3 compare_samples.py
```
Various parameters which can be tuned within these scripts - 
```
beta - mistake probability

num_answer - total number of different answers which can be obtained (k)

m - size of the execution set

f_max - largest fraction of byzantine nodes possible (<0.5, needed for SPRT)

num_runs - total number os iterations to average errors and samples

num_nodes - total number of nodes

```
