# MFBO for Discrete Fidelity Setting

## acq_demo.py

In this python file, I illustrate one Bayesian Optimization interation loop for simple objective function (Non linear sin) with two discrete fidelity setting, Resgp as MF model and UCB as the acquisition function.

## MF_discrete_acq_v2.py

This is the main file for Discrete Acquisition Function class where you can choose UCB_MF, EI_MF, PI_MF and KG_MF as your acquisition function. Correspondingly, there are several strategies for selection of next x and fidelity query. You may use the independent 'optimize_acq_mf' to find the argmax acq in all fidelities and then use 'acq_selection_fidelity' in 'Discrete Acquisition Function' class to choose targeted fidelity. 

Still, some other strategies of choosing next point (eg, cfkg and MFEI) are on their ways.

## MF_acq_illustration.ipynb 

This is a python notebook to help readers have a better understanding of the mathematical workflow behind all acquisition functions and selection strategies. (In progress)