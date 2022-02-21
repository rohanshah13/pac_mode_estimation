# PAC Mode estimation using PPR Martingale Confidence Sequences (Election Experiments)
Contains the implementation of various DCB and RR algorithm variants for estimating the winner of elections. In particular, we provide the implementation of DCB and RR variants of PPR-1VR, PPR-1V1, KLSN-1VR, KLSN-1V1, A1-1VR and A1-1V1.

## Prerequisites
Requires Python 3 for running the code. Install all the dependencies by running the following command:
```
$ pip3 install -r requirements.txt
```


## Reproducing the results
The main command for running the code
```
$ python3 runElectionBatch.py --dataset <dataset_name> --algorithm <algorithm_name>
```
The experiments are performed for 10 random seeds (from 0-9) and the mistake probability is set to 0.01. When a constituency is chosen votes are sampled in a batch of 200. The results containing the sample complexity, winning party, number of seats resolved completely and time elapsed are logged after executing the above command. These statistics for each of the runs, as well as the number of votes(unlabelled) sampled in each constituency, are also saved in `.npy` format at `results/<dataset_name>/<algorithm_name>`. 

As we sample without replacement we display two vote counts:
i)Unlabelled - Total number of samples, where a voter is counted as many times as his vote is sampled.
ii)Labelled - Total number of samples across constituencies, where a voter is counted only once even if his vote is sampled multiple times. This is the count reported in our paper.

Following flags for the algorithms can be used:

| `<algorithm_name>`| 
| :---              | 
| RRPPR1VR          | 
| RRPPR1V1          |
| RRA11VR           |
| RRA11V1           |
| RRKLSN1VR         |
| RRKLSN1V1         |
| DCBPPR1VR         | 
| DCBPPR1V1         |
| DCBA11VR          |
| DCBA11V1          |
| DCBKLSN1VR        |
| DCBKLSN1V1        |

## Datasets

We have provided synthetic datasets in the `data` folder. We have not attached preprocessed data from the *[India Elections website](https://www.indiavotes.com/)* due to legal concerns. Each of the files within a folder inside `data` directory (For example: `data/Easy-Easy`) represents a constituency and it contains information regarding the number of votes sampled for each of the parties within that constituency. 

The data from the below two links are to be parsed in the same format as that of the synthetic datasets in `data` for obtaining the results shown in the paper:
1. Indian National Elections 2014 (source - https://www.indiavotes.com/pc/info?eid=16&state=0)
2. Bihar State Elections 2015 (source - https://www.indiavotes.com/ac/info?stateac=58&eid=245)

Following flags for dataset can be used:

| `<dataset_name>` |
| :---             |
| Easy-Easy        |
| Easy-Hard        |
| Hard-Easy        |
| Hard-Hard        |