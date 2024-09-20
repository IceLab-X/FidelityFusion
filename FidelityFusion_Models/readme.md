# FidelityFusion_Models Modules
In this section, we provide the code for the models used in our library.

## Supported models
### 1. AR (AutoRegression)
The method can be found in the following paper
[Predicting the Output from a Complex Computer Code When Fast Approximations Are Available](https://www.jstor.org/stable/2673557)
There are two types of implementation variations:
AR-Concat: The main proxy model is a concatenation of the low-fidelity model and the residual model
AR-OneKernel: only one large kernel matrix across different fidelity levels is created and inversed.


### 2. NAR (Nonlinear AutoRegression)
The method can be found in the following paper
[Nonlinear information fusion algorithms for data-efficient multi-fidelity modelling](https://royalsocietypublishing.org/doi/10.1098/rspa.2016.0751)
### 3. ResGP(Residual Gaussian Process)
The method can be found in the following paper
[Residual Gaussian process: A tractable nonparametric Bayesian emulator for multi-fidelity simulations](https://www.sciencedirect.com/science/article/abs/pii/S0307904X21001724)
### 4. GAR、CIGAR (Generalized Autoregression) 
The method can be found in the following paper
[GAR: Generalized Autoregression for Multi-Fidelity Fusion](https://proceedings.neurips.cc/paper_files/paper/2022/file/37e9e62294ff6607f6f7c170cc993f2c-Paper-Conference.pdf)

### 5. CAR (Continue Autoregression)
[ContinuAR: Continuous Autoregression For Infinite-Fidelity Fusion](https://openreview.net/pdf?id=wpfsnu5syT)
There are two types of implementation variations:
CAR-Concat: The main proxy model is a concatenation of the low-fidelity model and the residual model
CAR-OneKernel: only one large kernel matrix across different fidelity levels is created and inversed.


## two_fidelity_models
This module contains the code for the two fidelity models used in our library.
The models provided above (except for GAR) are all based on Gaussian processes, while GAR is based on HOGP implementation. Here is also an introduction to HOGP methods
### 1. HOGP (High Order Gaussian Process)
The method can be found in the following paper
[Scalable High-Order Gaussian Process Regression](https://proceedings.mlr.press/v89/zhe19a.html)

## CAR_paper-version
There are two forms of CAR paper proposed methods, please note that this version has updates on the main proxy models

## GAR_res_rho
Below this folder are ablation experiments on the connection structure of precision in GAR, replacing it with a single learning factor or a fixed constant

## Log
This folder mainly stores the running logs of the multi fidelity proxy model (if using log debugger to run the program, it will generate logs)
