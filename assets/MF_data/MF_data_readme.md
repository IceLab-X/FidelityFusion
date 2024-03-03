# MF_data 

In this section, we have provided 42 collected datasets, and the following are the specific formula explanations for these 42 datasets



## 1、forrester_my


## 2、Non linear Sin

high:
$$f_{high}(x) = (x - \sqrt{2}) f_{low}(x)^2$$

low:
$$f_{low}(x) = \sin(8 \pi x)$$

## 8~17:  tl2~tl10
\begin{aligned}
& y_h=\sin (2 \pi(x-0.1))+x^2 \\
& y_l=\sin (2 \pi(x-0.1)) \\
& y_h=x \sin (x) / 10 \\
& y_l=x \sin (x) / 10+x / 10 \\
& y_h=\cos (3.5 \pi x) \exp (-1.4 x) \\
& y_l=\cos (3.5 \pi x) \exp (-1.4 x)+0.75 x^2 \\
& y_h=4 x_1^2-2.1 x_1^4+\frac{1}{3} x_1^6+x_1 x_2-4 x_2^2+4 x_2^4 \\
& y_l=2 x_1^2-2.1 x_1^4+\frac{1}{3} x_1^6+0.5 x_1 x_2-4 x_2^2+2 x_2^4 \\
& y_h=\frac{1}{6}\left[\left(30+5 x_1 \sin \left(5 x_1\right)\right)\left(4+\exp \left(-5 x_2\right)\right)-100\right] \\
& y_l=\frac{1}{6}\left[\left(30+5 x_1 \sin \left(5 x_1\right)\right)\left(4+\frac{2}{5} \exp \left(-5 x_2\right)\right)-100\right] \\
& y_h=\sum_{i=1}^2 x_i^4-16 x_i^2+5 x_i \\
& y_l=\sum_{i=1}^2 x_i^4-16 x_i^2 \\
& y_h=\left[1-2 x_1+0.05 \sin \left(4 \pi x_2-x_1\right)\right]^2+\left[x_2-0.5 \sin \left(2 \pi x_1\right)\right]^2 \\
& y_l=\left[1-2 x_1+0.05 \sin \left(4 \pi x_2-x_1\right)\right]^2+4\left[x_2-0.5 \sin \left(2 \pi x_1\right)\right]^2 \\
& y_h=\left(x_1-1\right)^2+\left(x_1-x_2\right)^2+x_2 x_3+0.5 \\
& y_l=0.2 y_h-0.5 x_1-0.2 x_1 x_2-0.1 \\
& y_h=\sum_{i=1}^8 x_i^4-16 x_i^2+5 x_i \\
& y_l=\sum_{i=1}^8 0.3 x_i^4-16 x_i^2+5 x_i
\end{aligned}


## 18~25:  test3~test9



## 26~30:


from: R. Pellegrini et al. Assessing the Performance of an Adaptive Multi-Fidelity Gaussian Process with Noisy Training Data: A Statistical Analysis





## 28、P3?：

highest to lowest:
D can be 2\5\10

from: **Multi-Fidelity Sparse Polynomial Chaos and Kriging SurrogateModels Applied to Analytical Benchmark Problems**



## 31、?not sure：和non sin撞车了!



from: Multi-fifidelity regression using artifificial neural networks: efficient approximation of parameter-dependent output quantities



## 32、Colville function (和P2很像)

from: A radial basis function-based multi-fidelity surrogate model: exploring correlation between high-fidelity and low-fidelity modelse



## 33~44 、maolin1~maolin20：

maolin6有点奇怪, 没用到x2
maolin14和hartmann重复了?


from:  Maolin Shi1 & Liye Lv1 & Wei Sun1 & Xueguan Song1. A multi-fidelity surrogate model based on support vector regression



## 45、Toal：


from: Toal DJ (2015) Some considerations regarding the use of multi-fdelity Kriging in the construction of surrogate models. Struct Multidisc Optim 51(6):1223–1245



## 46~49：
和maolin13撞车:

和maolin19撞车:


All from:  Shuo Wang, et al. A multi‑fdelity surrogate model based on moving least squares: fusing diferent fdelity data for engineering design.
