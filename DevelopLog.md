# DevelopLog
Create_time: 2024-3-11

## Development cycle(From 2023.12.21 to 2024.3.11)
## Library framework
### .vscode/
vscode编辑器调试文档
### assert/
- figures/
存放GAR的宣传图片
- MF_data/
收集的41个MF数据集合以及部分泊松方程的输入输出
  - MF_data_readme.md
  MF数据集合的说明文档
  - collected_data
  收集的41个MF数据函数
  支持自己输入的数据以及生成的数据(带有数据范围检测以及提醒)
### Bayesian_optimization/
贝叶斯优化模块
- exp.ipynb
使用教程
- con_mace_acq_demo
带约束的贝叶斯优化
### Data_simulation/
- Cost_function/
- Real_Application/
- Synthetic_MF_function/

### Experiment/
实验模块(支持指标计算以及实验结果的可视化)
- Readme.md
实验模块的说明文档
- CAR_Cost/
- CAR_Non_Subset/
CAR的非子集实验
- CAR_Subset/
CAR的子集实验
- GAR_Aligned/
GAR的对齐实验
- GAR_Non_Aligned/
GAR的非对齐实验
- GAR_Non_Subset/
GAR的非子集实验
- MFBO_continuous/
MFBO的连续实验
- MFBO_discrete/
MFBO的离散实验
- calculate_metrix
计算实验结果的指标
- Load_Mfdata
MF数据加载模块（支持生成子集非子集数据）
- log_debugger
实验日志调试模块(支持实验日志的读取以及分析)

### FidelityFusion_Models/
- log/
存放模型训练的日志
- two_fidelity_models/
存放只支持两个保真度的模型
- __init__.py
初始化文件
- AR、ResGP、NAR、CIGAR、GAR、CAR模型
- MF_data
MF数据管理模块，支持标准化以及非子集数据结构的求解
- readme.md
模型模块的说明文档

### Gaussian_Process/
- cigp_v10
一个比较稳定的gp版本，也是目前库模型的底层GP
- cigp_withMean
带有均值函数的GP
- cigp_withpack
使用gp_computation_pack的 GP
- gp_basic
实例化GP
- gp_computation_pack
GP的计算模块
- gp_transform
一些用于GP的变换函数以及标准化层
- GP_tutorial_basicGP.ipynb
GP的基础教程
- hogp_simple
High-Order-GP
- kernel
GP的核函数

### MF_Bayesian_Optimization/
多精度贝叶斯优化模块
- Continous
连续贝叶斯优化模块
- Discrete
离散贝叶斯优化模块

### MFGP_ver2023May
2023.5月的库重构版本

### nan_error_example
提供错误解决方法的示例

## 还存在的BUG
暂无，等待更多测试
- [ ] 部分kernel需要修改以支持高维

## 还需要完善的地方
- [ ] CAR方法预测不是很准
- [ ] MFBO还需要完善
