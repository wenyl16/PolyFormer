import os

import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
import matplotlib.pyplot as plt
from Simulator.Plotter import ErrorVisualizer
import os
import torch
from Simulator.cases.basic_cases import case_nonconvex
model_type = 'pretrainnet'
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
case = case_nonconvex(model_type = model_type)
ec = case['errorcalculator']

dim_theta = 3
fullnet = FullNet(dim_theta = dim_theta, A_init=case['A_hat'],b_init = case['b_hat'])
fullnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\fullnet_weights.pth', map_location=device))

# 定义范围和步长
x = np.linspace(-0.3, 0.3, num=31)
y = np.linspace(-0.3, 0.3, num=31)
# 生成网格点
X, Y = np.meshgrid(x, y)
points = np.column_stack((X.ravel(), Y.ravel()))  # 合并为Nx2的数组

def gen_error_data(points, filepath = None):
    ev = ErrorVisualizer()
    for dtheta in points:

        A_pred, b_pred = fullnet(torch.tensor(np.hstack((dtheta,[0.0])),dtype=torch.float32))
        A_pred = A_pred[0].detach().cpu().numpy()
        b_pred = b_pred[0].detach().cpu().numpy()
        theta = np.array([1+dtheta[0], 1+dtheta[1], 1.0])
        ec.update_parameters({'theta':theta})
        ec.update_polytope(A_hat=A_pred, b_hat=b_pred)
        ev.compute_errors(model=ec, num_sample=10)
        ec._iter+=1

    error_feas_max = [np.max(arr) for arr in ev.error_history['error_feas']]
    error_feas_mean = [np.mean(arr) for arr in ev.error_history['error_feas']]
    error_opt_max = [np.max(arr) for arr in ev.error_history['error_opt']]
    error_opt_mean = [np.mean(arr) for arr in ev.error_history['error_opt']]

    error_total = [(feas + opt)/2 for feas, opt in zip(ev.error_history['error_feas'], ev.error_history['error_opt'])]
    error_total_max = [np.max(arr) for arr in error_total]
    error_total_mean = [np.mean(arr) for arr in error_total]
    # 创建DataFrame
    df = pd.DataFrame({
        'iterations': ev.error_history['iterations'],
        'error_feas_max': error_feas_max,
        'error_feas_mean': error_feas_mean,
        'error_opt_max': error_opt_max,
        'error_opt_mean': error_opt_mean,
        'error_total_max': error_total_max,
        'error_total_mean': error_total_mean
    })

    # 保存为CSV文件
    if filepath:
        df.to_csv(filepath, index=False)
        print("CSV文件已保存为 'error_statistics.csv'")
    return df

filepath = f'{PROJECT_ROOT}\\results\\{case['casename']}\\error_statistics.csv'
# gen_error_data(points, filepath = filepath)

df = pd.read_csv(filepath)
error_opt_mean = np.array(df['error_opt_mean'])+np.array(df['error_feas_mean'])
# 3. 将 error_opt_mean 重塑为网格形状 (5x5)
Z = error_opt_mean.reshape(X.shape)  # 关键步骤！

# 4. 绘制曲面图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='k', alpha=0.8)

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, label='error_opt_mean')

# 设置标签和标题
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('error_opt_mean')
ax.set_title('3D Surface Plot of error_opt_mean')

# 调整视角
ax.view_init(elev=30, azim=45)  # 可修改角度

plt.tight_layout()
plt.show()