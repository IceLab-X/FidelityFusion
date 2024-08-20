import sys
import os

from platypus import Problem, Real, Archive, NSGAII

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.append("C:\\Users\\icelab01\\Desktop\\ZhuohuaLiu_2024_BYA_jiebang\\FidelityFusion-zhenjie-s-branch\\FidelityFusion-zhenjie-s-branch\\FidelityFusion\\GaussianProcess")

# Define the objective function
import torch
from acq import UCB, EI, PI, KG, find_next_batch, optimize_acqf, PF
import numpy as np

# Define the objective function

import matplotlib.pyplot as plt
import GaussianProcess.kernel as kernel
from cigp import CIGP_withMean

global train_y


def objective_function(x):
    # Simple sum of sine functions for demonstration
    return torch.sin(x)+torch.sin(2*x)


# Initialize prior knowledge with 10 random points
input_dim = 1
num_initial_points = 5
train_x = torch.rand(num_initial_points, input_dim) * 10  # Random points in [0, 10] for each dimension
train_y = objective_function(train_x).reshape(-1,1)

# Define a simple exact GP model using GPyTorch
kernel1 = kernel.ARDKernel(1)
# kernel1 = kernel.MaternKernel(1)
# kernel1 = kernel.LinearKernel(1,-1.0,1.)
# kernel1 = kernel.SumKernel(kernel.LinearKernel(1), kernel.MaternKernel(1))

model = CIGP_withMean(1, 1, kernel=kernel1, noise_variance=2.)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

# Define the mean and variance functions for UCB
def mean_function(X):
    model.eval()
    with torch.no_grad():
        mean, _ = model.forward(train_x, train_y, X)
        return mean

def variance_function(X):
    model.eval()
    with torch.no_grad():
        _, var = model.forward(train_x, train_y, X)
        return var


thresholds = [-2]


ucb = UCB(mean_function, variance_function, kappa=5)
pi = PI(mean_function, variance_function)
ei = EI(mean_function, variance_function)
pf = PF(mean_function, variance_function, thresholds)

def obj(x):
    # 使用已知的目标函数值来计算new_y的最大值
    new_y = max(train_y) if len(train_y) > 0 else 0
    x = torch.tensor(x) if isinstance(x, list) else x
    x = x.reshape(1, -1)
    u = ucb.forward(x)
    p = pi.forward(x, new_y.max().item())
    e = ei.forward(x, new_y.max().item())
    f = pf.forward(x)
    f = torch.unsqueeze(torch.tensor(f), 0)
    max1 = max(0, mean_function(x).item())
    max2 = max(0, mean_function(x).item()/torch.sqrt(variance_function(x)).item())
    return [-u[0], -p[0], -e[0], -f[0], max1, max2]


best_y = []
# use it to remember the key iteration
key_iterations = [2,4,5,6,8,10]
predictions = []
iteration_label = True

# Bayesian optimization loop
bounds = torch.tensor([[0, 10]] * input_dim)

lb = np.zeros(input_dim)
ub = np.zeros(input_dim)
for i in range(input_dim):
    lb[i] = bounds[i][0]
    ub[i] = bounds[i][1]

for iteration in range(20):  # Run for 10 iterations
    for i in range(100):
        optimizer.zero_grad()
        loss = -model.log_likelihood(train_x, train_y)
        loss.backward()
        optimizer.step()
        print('iter', i, 'nll:{:.5f}'.format(loss.item()))

    problem = Problem(input_dim, 6)          # dim:变量的个数；6：对应着六个采集函数
    problem.function = obj  # 定义多目标函数

    for i in range(input_dim):
        problem.types[i] = Real(lb[i].reshape(lb.size), ub[i].reshape(ub.size))


    problem.directions[:] = Problem.MAXIMIZE                # 定义最大化
    arch = Archive()                            # 存储非支配解的集合
    algorithm = NSGAII(problem, population=100, archive=arch)  # 创建多目标优化算法

    def cb(a):
        print(a.nfe, len(a.archive), flush=True)


    mo_eval = 1000
    algorithm.run(mo_eval, callback=cb)  # 运行多目标优化算法，开始寻找合适的采集函数

    optimized = algorithm.population
    idxs = np.arange(len(optimized))
    idxs = np.random.permutation(idxs)
    idxs = idxs[0:1]
    for i in idxs:  # 更新训练集
        x = np.array(optimized[i].variables)
        x = np.reshape(x, (1, -1))
        x = torch.tensor(x)
        y = objective_function(x.squeeze()).reshape(-1, 1)
        # Update the model
        train_x = torch.cat([train_x, x])
        train_y = torch.cat([train_y, y])
        # Store the best objective value found so far
        best_y.append(y.max().item())


    # 在关键迭代时保存模型预测
    if (iteration + 1) in key_iterations:
        model.eval()
        fixed_dims = torch.full((1, input_dim - 1), 5.0)  # Example: set them to the midpoint (5.0)
        test_points = torch.linspace(0, 10, 100)
        test_X = torch.cat((test_points.unsqueeze(1), fixed_dims.expand(test_points.size(0), -1)), 1)
        true_y = objective_function(test_X)

        with torch.no_grad():
            pred_mean, pred_std = model.forward(train_x, train_y, test_X)
            predictions.append((pred_mean, pred_std))


# 绘制子图
plt.figure(figsize=(15, 12))
for i, (pred_mean, pred_std) in enumerate(predictions):
    plt.subplot(3, 2, i+1)
    plt.ylim(-5, 5)
    plt.plot(test_points.numpy(), true_y.numpy(), 'k-', label='True function')
    plt.plot(test_points.numpy(), pred_mean.numpy(), 'b--', label='GP mean')
    plt.fill_between(test_points.numpy().reshape(-1),
                     (pred_mean - 2 * pred_std).numpy().reshape(-1),
                     (pred_mean + 2 * pred_std).numpy().reshape(-1),
                     color='blue', alpha=0.2, label='GP uncertainty')

    observed_x = train_x[:, 0].numpy()  # Only the first dimension for all observed points
    observed_y = train_y.numpy()
    plt.scatter(observed_x[:num_initial_points+key_iterations[i]], observed_y[:num_initial_points+key_iterations[i]], c='r', zorder=3, label='Observed points')
    plt.title(f'Samples: {key_iterations[i]}')
    plt.legend()

plt.tight_layout()
plt.show()
