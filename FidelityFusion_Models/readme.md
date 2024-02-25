# FidelityFusion_Models Modules
In this section, we provide the code for the models used in our library.

## Supported models
### 1. AR (AutoRegression)
The method can be found in the following paper
[Predicting the Output from a Complex Computer Code When Fast Approximations Are Available](https://www.jstor.org/stable/2673557)
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
## two_fidelity_models
This module contains the code for the two fidelity models used in our library.
The models provided above (except for GAR) are all based on Gaussian processes, while GAR is based on HOGP implementation. Here is also an introduction to HOGP methods
### 1. HOGP (High Order Gaussian Process)
The method can be found in the following paper
[Scalable High-Order Gaussian Process Regression](https://proceedings.mlr.press/v89/zhe19a.html)
