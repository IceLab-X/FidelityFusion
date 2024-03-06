# MF_data 

In this section, we have provided 41 collected datasets, and the following are the specific formula explanations for these 41 datasets

## 1.colville：
$$\begin{aligned}
& y_h=100*(x_1^2-x_2)^2 + (x_1-1)^2 + (x_3-1)^2 + 90*(x_3^2-x_4) + 10.1*((x_2-1)^2+(x_4-1)^2) + 19.8*(x_2-1)*(x_4-1) \\
& y_l=y_{low}*(A^2(x_1,x_2,x_3,x_4)) - (A+0.5)(5*x_1^2 + 4*x_2^2 + 3*x_3^2 + x_4^2) \\
\end{aligned}$$
$xdim = 4,x\in[-1,1]^D,A<=0.68$

## 2.Non linear Sin：

$high:$
$$f_{high}(x) = (x - \sqrt{2}) f_{low}(x)^2$$

$low:$
$$f_{low}(x) = \sin(8 \pi x)$$
$xdim = 1,x\in[-5,10]^D$

## 3.Toal：
$\mathrm{HF}$ function:
$$
y_h=\sum_{i=1}^{10}\left(x_i-1\right)^2-\sum_{i=2}^{10} x_i x_{i-1}
$$

$\mathrm{LF}$ function:
$$
y_l=\sum_{i=1}^{10}\left(x_i-A\right)^2-(A-0.65) \sum_{i=2}^{10} i x_i x_{i-1}
$$
where $x_i \in[-100,100], i=1,2, \ldots, 10$. The parameter $A$ varies from 0 to 1 .

from: [Toal DJ (2015) Some considerations regarding the use of multi-fdelity Kriging in the construction of surrogate models. Struct Multidisc Optim 51(6):1223–1245](https://eprints.soton.ac.uk/373482/1/Some%2520Considerations%2520Regarding%2520the%2520Use%2520of%2520Co-Kriging%2520in%2520the%2520Construction%2520of%2520Surrogate%2520Models_Final.pdf)

## 4.forrester：
$$\begin{aligned}
& f_1(x) = (6x - 2)^2 \sin(12x - 4) \\
& f_2(x) = (5.5x - 2.5)^2 \sin(12x - 4) \\
& f_3(x) = 0.75 f_{1}(x) + 5 (x - 0.5) - 2 \\
& f_4(x) = 0.5 f_{1}(x) + 10 (x - 0.5) - 5 \\
\end{aligned}$$
$xdim = 1,x\in[0,1]^D$
from:[Forrester, A.I.; Sóbester, A.; Keane, A.J. Multi-fidelity optimization via surrogate modelling. Proc. R. Soc. A Math. Phys. Eng. Sci. 2007,463, 3251–3269.](https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2007.1900)

## 5~9. P1~P5：
$$\begin{equation}
\begin{array}{|c|c|c|c|c|c|}
\hline \text { Test } & & \text { Formulation } & \text { Ref. } & \text { Domain } & \text { D } \\
\hline P_1 & \begin{array}{l}
f_1(x)= \\
f_2(x)= \\
f_3(x)=
\end{array} & \begin{array}{l}
\sin \left(30(x-0.9)^4\right) \cos (2(x-0.9))+(x-0.9) / 2+\eta_1 \lambda_1(x) \\
\left(f_1(x)-1+x\right) /(1+0.25 x)+\eta_2 \lambda_1(x) \\
\sin \left(20(x-0.87)^4\right) \cos (2(x-0.87))+(x-0.87) / 2 \\
\left(2.5-(0.7 x-0.14)^2\right)+2 x+\eta_3 \lambda_1(x)
\end{array} & \begin{array}{l}
{[19]} \\
{[19]} \\
{[-]}\\
\end{array} & x \in[0,1] & 1 \\
\hline P_2 & \begin{array}{l}
f_1(x)= \\
f_2(x)= \\
f_3(x)=
\end{array} & \begin{array}{l}
\sin \left(30(x-0.9)^4\right) \cos (2(x-0.9))+(x-0.9) / 2+\eta_1 \lambda_2(x) \\
\left(f_1(x)-1+x\right) /(1+0.25 x)+\eta_2 \lambda_2(x) \\
\sin \left(20(x-0.87)^4\right) \cos (2(x-0.87))+(x-0.87)\left(2.5-(0.7 x-0.14)^2\right)+2 x+\eta_3 \lambda_2(x)
\end{array} & \begin{array}{l}
{[19]} \\
{[19]} \\
{[-]}\\
\end{array} & x \in[0,1] & 1 \\
\hline P_3 & \begin{array}{l}
f_1(\mathbf{x})= \\
f_2(\mathbf{x})= \\
f_3(\mathbf{x})=
\end{array} & \begin{array}{l}
\sum_{j=1}^{\mathbf{D}-1}\left[100\left(x_{j+1}-x_j^2\right)^2+\left(1-x_j\right)^2\right]+\eta_1 \lambda_1\left(x_1\right) \\
\sum_{j=1}^{\mathbf{D}-1}\left[50\left(x_{j+1}-x_j^2\right)^2+\left(-2-x_j\right)^2\right]-\sum_{j=1}^{\mathbf{D}} 0.5 x_j+\eta_2 \lambda_1\left(x_1\right) \\
\left(f_1(\mathbf{x})-4-\sum_{j=1}^{\mathbf{D}} 0.5 x_j\right) /\left(10+\sum_{j=1}^{\mathbf{D}} 0.25 x_j\right)+\eta_3 \lambda_1\left(x_j\right)
\end{array} & \begin{array}{l}
{[9]} \\
{[-]}\\
{[9]}
\end{array} & \mathbf{x} \in[-2,2] & 2 \\
\hline P_4 & \begin{array}{l}
f_1(\mathbf{x})= \\
f_2(\mathbf{x})= \\
f_3(\mathbf{x})=
\end{array} & \begin{array}{l}
\sum_{j=1}^{\mathbf{D}} \frac{x_j^2}{25}-\prod_{j=1}^{\mathbf{D}} \cos \left(\frac{x_j}{\sqrt{j}}\right)+1+\eta_1 \lambda_1\left(x_1\right) \\
\prod_{j=1}^{\mathbf{D}} \cos \left(\frac{x_j}{\sqrt{j}}\right)+1+\eta_2 \lambda_1\left(x_1\right) \\
\sum_{j=1}^{\mathbf{D}} \frac{x_j^2}{20}-\prod_{j=1}^{\mathbf{D}} \cos \left(\frac{x_j}{\sqrt{j+1}}\right)-1+\eta_3 \lambda_1\left(x_1\right)
\end{array} & \begin{array}{c}
{[21]} \\
{[-]}\\
{[-]}\\
\end{array} & \mathbf{x} \in[-6,5] & 2 \\
\hline P_5 & \begin{array}{r}
f_H(\mathbf{z})= \\
f_i(\mathbf{z})= \\
e_r\left(\mathbf{z}, \phi_i\right)= \\
\text { with } \\
\text { and } \\
\text { with }
\end{array} & \begin{array}{l}
\sum_{j=1}^{\mathbf{D}}\left(z_j^2+1-\cos \left(10 \pi z_j\right)\right) \\
f_H(\mathbf{z})+e_r\left(\mathbf{z}, \phi_i\right)+\eta_i \lambda_3\left(x_1\right), \quad i=1, \ldots, N \\
\sum_{j=1}^{\mathbf{D}} a\left(\phi_i\right) \cos ^2 \omega\left(\phi_i\right) z_j+b\left(\phi_i\right)+\pi, \quad i=1, \ldots, N \\
a\left(\phi_i\right)=\Theta\left(\phi_i\right), \omega\left(\phi_i\right)=10 \pi \Theta\left(\phi_i\right), b\left(\phi_i\right)=0.5 \pi \Theta\left(\phi_i\right) \\
\Theta\left(\phi_i\right)=1-0.0001 \phi_i \\
\phi=\{10000,5000,2500\}
\end{array} & \begin{array}{l}
{[20]} \\
{[20]} \\
{[20]} \\
{[20]} \\
{[20]} \\
{[-]}\\
\end{array} & \mathbf{x} \in[-0.1,0.2] & 2 \\
\hline
\end{array}
\end{equation}$$

from: [R. Pellegrini et al. Assessing the Performance of an Adaptive Multi-Fidelity Gaussian Process with Noisy Training Data: A Statistical Analysis](https://arxiv.org/pdf/2107.02455.pdf)

## 10~20. maolin1~maolin20:
$$\begin{array}{|c|c|c|c|c|c|}
\hline \text { No } & \mathrm{LF} / \mathrm{HF} & \text { Test functions } & \mathrm{D} & \mathrm{S} & r^2 \\
\hline 1 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=\frac{\sin (10 \pi x)}{2 x}+(x-1)^4 \\
y_L=\frac{\sin (10 \pi x)}{x}+2(x-1)^4
\end{array} & 1 & {[0,1]^{\mathrm{D}}} & 0.733 \\
\hline 5 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=\left(x_2-\frac{5.1 x_1^2}{4 \pi^2}+\frac{5.1 x_1}{\pi}-6\right)^2+10(1-0.125 \pi) \cos \left(x_1\right)+10 \\
y_L=(1-0.125 \pi) \cos \left(x_1\right)
\end{array} & 2 & {[0,5]^{\mathrm{D}}} & 0.771 \\
\hline 6 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=101 x_1^2+101\left(x_1^2+x_1^2\right)^2 \\
y_L=x_1^2+100\left(x_1^2+x_1^2\right)^4
\end{array} & 2 & {[-1,1]^{\mathrm{D}}} & 0.771 \\
\hline 7 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=\left[1-0.2 x_2+0.05 \sin \left(4 \pi x_2-x_1\right)\right]^2+\left[x_2-0.5 \sin \left(2 \pi x_1\right)\right]^2 \\
y_L=\left[1-0.2 x_2+0.05 \sin \left(4 \pi x_2-x_1\right)\right]^2+4\left[x_2-0.5 \sin \left(2 \pi x_1\right)\right]^2
\end{array} & 2 & {[-5,10]^{\mathrm{D}}} & 0.706 \\
\hline 8 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=\left(1.5-x_1+x_1 x_2\right)^2+\left(2.25-x_1+x_1 x_2^2\right)^2+\left(2.625-x_1+x_1 x_2^3\right)^2 \\
y_L=\left(1.5-x_1+x_1 x_2\right)^2+x_1+x_2
\end{array} & 2 & {[0,1]^{\mathrm{D}}} & 0.445 \\
\hline 10 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=\left[1-\exp \left(-\frac{1}{2 x_2}\right)\right] \frac{2300 x_1^3+1900 x_1^2+2092 x_2+60}{100 x_1^3+500 x_1^2+4 x_2+20} \\
y_L=-\frac{2}{5}\left[y_H\left(x_1+0.05, x_2+0.05\right)\right]+\frac{1}{4}\left[\begin{array}{c}
y_H\left(x_1+0.05, \max \left(0, x_2-0.05\right)\right) \\
+y_H\left(x_1-0.05, x_2+0.05\right)+y_H\left(x_1-0.05, \max \left(0, x_2-0.05\right)\right)
\end{array}\right]
\end{array} & 2 & {[0,0.5]^{\mathrm{D}}} & 0.752 \\
\hline 12 & \begin{array}{l}
\text { HF } \\
\text { LF }
\end{array} & \begin{array}{l}
y_H=x_1 \exp \left(-x_1^2-x_2^2\right) \\
y_L=x_1 \exp \left(-x_1^2-x_2^2\right)+\frac{x_1}{10}
\end{array} & 2 & {[-2,2]^{\mathrm{D}}} & 0.828 \\
\hline 13 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=\exp \left(x_1+x_2\right) \cos \left(x_1 x_2\right) \\
y_L=\exp \left(x_1+x_2\right) \cos \left(x_1 x_2\right)+\cos \left(x_1^2+x_2^2\right)
\end{array} & 2 & {[-1,1]^{\mathrm{D}}} & 0.927 \\
\hline 14 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=\sum_{i=1}^3 \alpha_i \exp \left(-\sum_{j=1}^3 A_{i j}\left(x_j-p_{i j}\right)^2\right) \\
y_L=\sum_{i=1}^3 \exp \left(-\sum_{j=1}^3 A_{i j}\left(x_j-p_{i j}\right)^2\right)
\end{array} & 2 & {[-1,1]^{\mathrm{D}}} & 0.927 \\
\hline 15 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=100\left(\exp \left(-\frac{2}{x_1^{175}}\right)+\exp \left(-\frac{2}{x_2^{1.75}}\right)+\exp \left(-\frac{2}{x_3^{175}}\right)\right) \\
y_L=100\left(\exp \left(-\frac{2}{x_1^{1.75}}\right)+\exp \left(-\frac{2}{x_7^{175}}\right)+0.2 * \exp \left(-\frac{2}{x_2^{175}}\right)\right)
\end{array} & 3 & {[0,1]^{\mathrm{D}}} & 0.865 \\
\hline 19 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=\sum_{i=1}^{d-1}\left[100\left(x_{i+1}-x_i^2\right)^2+\left(x_i-1\right)^2\right] \\
y_L=\sum_{i=1}^{d-1}\left[100\left(x_{i+1}-x_i\right)^2+4\left(x_i-1\right)^4\right]
\end{array} & 6 & {[-5,10]^{\mathrm{D}}} & 0.761 \\
\hline 20 & \begin{array}{l}
\mathrm{HF} \\
\mathrm{LF}
\end{array} & \begin{array}{l}
y_H=4\left(x_1-2+8 x_2-8 x_2^2\right)^2+\left(3-4 x_2\right)^2+16 \sqrt{x_3+1}\left(2 x_3-1\right)^2 \sum_{i=4}^8 i \ln \left(1+\sum_{j=3}^i x_j\right) \\
y_H=4\left(x_1-2+8 x_2-8 x_2^2\right)^2+\left(3-4 x_2\right)^2+16 \sqrt{x_3+1}\left(2 x_3-1\right)^2 \sum_{i=4}^8 \ln \left(1+\sum_{j=3}^i x_j\right)
\end{array} & 8 & {[0,1]^{\mathrm{D}}} & 0.731 \\
\hline
\end{array}$$
from:  [Maolin Shi1 & Liye Lv1 & Wei Sun1 & Xueguan Song1. A multi-fidelity surrogate model based on support vector regression](https://arxiv.org/ftp/arxiv/papers/1906/1906.09439.pdf)
## 21~30. tl1~tl10:
$tl1:$
$$\begin{aligned}
& y_h=(6x - 2)^2 \sin(12x - 4) \\
& y_l=0.56 * ((6x - 2)^2 \sin(12x - 4)) + 10(x - 0.5) - 5 \\
\end{aligned}$$
$xdim = 1,x\in[0,1]^D$

$tl2:$
$$\begin{aligned}
& y_h=\sin (2 \pi(x-0.1))+x^2 \\
& y_l=\sin (2 \pi(x-0.1)) \\
\end{aligned}$$
$xdim = 1,x\in[0,1]^D$

$tl3:$
$$\begin{aligned}
& y_h=x \sin (x) / 10 \\
& y_l=x \sin (x) / 10+x / 10 \\
\end{aligned}$$
$xdim = 1,x\in[0,10]^D$

$tl4:$
$$\begin{aligned}
& y_h=\cos (3.5 \pi x) \exp (-1.4 x) \\
& y_l=\cos (3.5 \pi x) \exp (-1.4 x)+0.75 x^2 \\
\end{aligned}$$
$xdim = 1,x\in[0,1]^D$

$tl5:$
$$\begin{aligned}
& y_h=4 x_1^2-2.1 x_1^4+\frac{1}{3} x_1^6+x_1 x_2-4 x_2^2+4 x_2^4 \\
& y_l=2 x_1^2-2.1 x_1^4+\frac{1}{3} x_1^6+0.5 x_1 x_2-4 x_2^2+2 x_2^4 \\
\end{aligned}$$
$xdim = 2,x∈[-2,2]^D$

$tl6:$
$$\begin{aligned}
& y_h=\frac{1}{6}\left[\left(30+5 x_1 \sin \left(5 x_1\right)\right)\left(4+\exp \left(-5 x_2\right)\right)-100\right] \\
& y_l=\frac{1}{6}\left[\left(30+5 x_1 \sin \left(5 x_1\right)\right)\left(4+\frac{2}{5} \exp \left(-5 x_2\right)\right)-100\right] \\
\end{aligned}$$
$xdim = 2,x\in[0,1]^D$

$tl7:$
$$\begin{aligned}
& y_h=\sum_{i=1}^2 x_i^4-16 x_i^2+5 x_i \\
& y_l=\sum_{i=1}^2 x_i^4-16 x_i^2 \\
\end{aligned}$$
$xdim = 2,x\in[-3,4]^D$

$tl8:$
$$\begin{aligned}
& y_h=\left[1-2 x_1+0.05 \sin \left(4 \pi x_2-x_1\right)\right]^2+\left[x_2-0.5 \sin \left(2 \pi x_1\right)\right]^2 \\
& y_l=\left[1-2 x_1+0.05 \sin \left(4 \pi x_2-x_1\right)\right]^2+4\left[x_2-0.5 \sin \left(2 \pi x_1\right)\right]^2 \\
\end{aligned}$$
$xdim = 2,x\in[0,1]^D$

$tl9:$
$$\begin{aligned}
& y_h=\left(x_1-1\right)^2+\left(x_1-x_2\right)^2+x_2 x_3+0.5 \\
& y_l=0.2 y_h-0.5 x_1-0.2 x_1 x_2-0.1 \\
\end{aligned}$$
$xdim = 3,x\in[0,1]^D$

$tl10:$
$$\begin{aligned}
& y_h=\sum_{i=1}^8 x_i^4-16 x_i^2+5 x_i \\
& y_l=\sum_{i=1}^8 0.3 x_i^4-16 x_i^2+5 x_i
\end{aligned}$$
$xdim = 8,x\in[-3,3]^D$

## 31~34.shuo:
$$\begin{array}{lllll}
\hline \text { No. } & \text { HF/LF } & \text { Test functions } & \text { D } & \text { S } \\
\hline 
6 & \text { HF } & y_h=\left[x_2-1.275\left(\frac{x_1}{\pi}\right)^2+5 \frac{x_1}{\pi}-6\right]^2+10\left(1-\frac{1}{8 \pi}\right) \cos \left(x_1\right) & 2 & \begin{array}{l}
x_1 \in[-5,10] \\
x_2 \in[0,15]
\end{array} \\
& \text { LF } & y_l=\frac{1}{2}\left[x_2-1.275\left(\frac{x_1}{\pi}\right)^2+5 \frac{x_1}{\pi}-6\right]^2+10\left(1-\frac{1}{8 \pi}\right) \cos \left(x_1\right) & \\
11 & \text { HF } & y_h=\sum_{i=1}^3 0.3 \sin \left(\frac{16}{15} x_i-1\right)+\left[\sin \left(\frac{16}{15} x_i-1\right)\right]^2 & 3 & {[-1,1]^{\mathrm{D}}} \\
& \text { LF } & y_l=\sum_{i=1}^3 0.3 \sin \left(\frac{16}{15} x_i-1\right)+0.2\left[\sin \left(\frac{16}{15} x_i-1\right)\right]^2 & \\
15 & \text { HF } & y_h=\sum_{i=1}^2\left[\left(x_{4 i-3}+10 x_{4 i-2}\right)^2+5\left(x_{4 i-1}-x_{4 i}\right)^2+\left(x_{4 i-2}-2 x_{4 i-1}\right)^4+10\left(x_{4 i-3}-x_{4 i}\right)^4\right] & 8 & {[0,1]^{\mathrm{D}}} \\
& \text { LF } & y_l=\sum_{i=1}^2\left[\left(x_{4 i-3}+10 x_{4 i-2}\right)^2+125\left(x_{4 i-1}-x_{4 i}\right)^2+\left(x_{4 i-2}-2 x_{4 i-1}\right)^4+10\left(x_{4 i-3}-x_{4 i}\right)^4\right] & \\
16 & \text { HF } & y_h=\sum_{i=1}^{10} \exp \left(x_i\right)\left[A(i)+x_i-\ln \left(\sum_{k=1}^{10} \exp \left(x_k\right)\right)\right] & 10 & {[-2,3]^{\mathrm{D}}} \\
& & A=[-6.089,-17.164,-34.054,-5.914,-24.721,-14.986,-24.100,-10.708,- & \\
& &26.662,-22.662,-22.179] & & \\
& \text { LF } & y_l=\sum_{i=1}^{10} \exp \left(x_i\right)\left[B(i)+x_i-\ln \left(\sum_{k=1}^{10} \exp \left(x_k\right)\right)\right] & \\
&& B=[-10,-10,-20,-10,-20,-20,-20,-10,-20,-20] & \\
\hline
\end{array}$$
All from:  [Shuo Wang, et al. A multi‑fdelity surrogate model based on moving least squares: fusing diferent fdelity data for engineering design.](https://link.springer.com/article/10.1007/s00158-021-03044-5)

## 35~41. test3~test9：
$$\begin{array}{|c|c|c|c|}
\hline \begin{array}{l}
\text { Test-3 } \\
\text { (cf. [32]) }
\end{array} & y^L(x)=\mathrm{e}^{1.4 x} \cos (3.5 \pi x) & y^H(x)=\mathrm{e}^x \cos (x)+\frac{1}{x^2} & x \in[0,1] \\
\hline \begin{array}{l}
\text { Test-4 } \\
(c f .[33])
\end{array} & y^L(x)=\sin \left(\frac{2 \pi x}{10}\right)+0.2 \sin \left(\frac{2 \pi x}{2.5}\right) & y^H(x)=\sin \left(\frac{2 \pi x}{2.5}\right)+\cos \left(\frac{2 \pi x}{2.5}\right) & x \in[0,10] \\
\hline \begin{array}{l}
\text { Test-5 } \\
\text { (cf. [34]) }
\end{array} & y^L\left(x_1, x_2\right)=y^H\left(0.7 x_1, 0.7 x_2\right)+x_1 x_2-65 & \begin{aligned}
y^H\left(x_1, x_2\right)=4 x_1^2- & 2.1 x_1^4+\frac{x_1^6}{3} \\
& -4 x_2^2+4 x_2^4+x_1 x_2
\end{aligned} & \left(x_1, x_2\right) \in[-2,2]^2 \\
\hline \begin{array}{l}
\text { Test-6 } \\
(c f .[35])
\end{array} & y^L(\mathbf{x})=100 \mathrm{e}^{\sin \left(x_1\right)}+5 x_2 x_3+x_4+\mathrm{e}^{x_5 x_6} & y^H(\mathbf{x})=\mathrm{e}^{\sin \left[\left(0.9\left(x_1+0.48\right)\right)^{10}\right]}+x_2 x_3+x_4 & \mathbf{x} \in[0,1]^6 \\
\hline \begin{array}{l}
\text { Test-7 } \\
(c f .[36])
\end{array} & \begin{aligned}
y^L(\mathbf{x})=\sum_{i=5}^8 x_i \cos & \left(\sum_{j=1}^4 x_j\right) \\
& +\sum_{i=5}^8 x_i \sin \left(\sum_{j=1}^4 x_j\right)
\end{aligned} & \begin{aligned}
y^H(\mathbf{x}) & =\left[\left(\sum_{i=5}^8 x_i \cdot \cos \left(\sum_{j=1}^4 x_j\right)\right)^2\right. \\
& \left.+\left(\sum_{i=5}^8 x_i \cdot \sin \left(\sum_{j=1}^4 x_j\right)\right)^2\right]^{\frac{1}{2}}
\end{aligned} & \begin{array}{l}
\left(x_1, \cdots, x_8\right) \\
\quad \in[0,2 \pi]^4 \times[0,1]^4
\end{array} \\
\hline \begin{array}{l}
\text { Test-8 } \\
(c f .[20])
\end{array} & y^H(\mathbf{x})=\left(x_1\right)^2+\sum_{i=2}^{20}\left(2 x_i^2-x_{i-1}\right)^2 & y^L(\mathbf{x})=0.8 y^H(\mathbf{x})-\sum_{i=1}^{19} 0.4 x_i x_{i+1}-50 & \mathbf{x} \in[-3,3]^{20} \\
\hline \begin{array}{l}
\text { Test-9 } \\
(c f .[37])
\end{array} & y^L(\mathbf{x})=\left(y^H(\mathbf{x})\right)^3+\left(y^H(\mathbf{x})\right)^2+y^H(\mathbf{x}) & \begin{aligned}
y^H(\mathbf{x})= & \left(x_1-1\right)^2+\left(x_{30}-1\right)^2 \\
& +30 \sum_{i=1}^{29}(30-i)\left(x_i^2-x_{i+1}\right)^2
\end{aligned} & \mathbf{x} \in[-3,2]^{30} \\
\hline
\end{array}$$

