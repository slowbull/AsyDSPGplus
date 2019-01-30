# AsyDSPGplus

## Introduction
This is the code for paper [1]. Please cite it if this code helps you. :)

[1] Bin Gu, Zhouyuan Huo, Heng Huang, Asynchronous Doubly Stochastic Group Regularized Learning, AISTATS, 2018



## Installation
1. Clone this repository from github.   
2. run   ```sh install.sh```

  
   
## How to use.
   ```
   ./platform -logistic_l2_l1 -n_threads=4  -hogwild_trainer -learning_rate=1e-3  -l1_lambda=1e-6 -l2_lambda=1e-6 -n_epochs=20 -dense_svrg -print_loss_per_epoch --data_file="data/ijcnn1" --model_block="data/ijcnn1_block"

   ```
    


## Disclaimer

This repository uses code from [Cyclades](https://github.com/amplab/cyclades).


