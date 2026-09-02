import os
import numpy as np
import scipy
from Simulator.Plotter import ShapeDrawer_2D
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.safe_region_case import safe_region_case
from Simulator import PROJECT_ROOT
import numpy as np
import matplotlib.pyplot as plt
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parallel= False
is_epigraph = False

# T = 12
# Delta_t = 5 / 60

# power profiles
# data_path = f'{PROJECT_ROOT}\\data\\profiles_data\\'
# data = np.load(data_path+'profiles_data.npz')
# full_p_bl, full_p_pv, full_theta_amb = 20*data['load_data'], 40*data['pv_data'], data['temp_data']

#
#
# # 时间轴：每个点间隔5分钟 => 1小时=60分钟
# n = len(full_p_bl)
# time_hours = np.arange(0, n * 5, 5) / 60  # 转为小时
#
# # 设置英文学术期刊风格
# plt.rcParams.update({
#     "font.family": "Times New Roman",
#     "font.size": 12,
#     "axes.labelsize": 18,
#     "axes.titlesize": 14,
#     "legend.fontsize": 12,
#     "xtick.labelsize": 14,
#     "ytick.labelsize": 14,
#     "figure.figsize": (7, 5),
#     "lines.linewidth": 1.5,
# })
#
# # 创建3行共享x轴的子图
# fig, axes = plt.subplots(3, 1, sharex=True)
#
# # -------- 上图：Boiler Pressure --------
# axes[0].plot(time_hours, full_p_bl, color='C0')
# # axes[0].set_ylabel('$p_{bl}$ [bar]', fontsize=13)
# axes[0].set_xlim(time_hours[0], time_hours[-1])
# axes[0].set_ylim(5,20)
# # -------- 中图：Valve Pressure --------
# axes[1].plot(time_hours, full_p_pv, color='C1')
# # axes[1].set_ylabel('$p_{pv}$ [bar]', fontsize=13)
# axes[1].set_ylim(0,40)
# # -------- 下图：Ambient Temperature --------
# axes[2].plot(time_hours, full_theta_amb, color='C2')
# # axes[2].set_ylabel('$\\theta_{amb}$ [°C]', fontsize=13)
# axes[2].set_xlabel('Time [h]', fontsize=13)
# axes[2].set_ylim(-15,0)
# # 去掉网格和上右边框
# for ax in axes:
#     ax.grid(False)
#     ax.spines['top'].set_visible(False)
#     ax.spines['right'].set_visible(False)
#
# # 调整布局
# plt.tight_layout()
# plt.show()

# initial profiles

import pickle
res_path = f'{PROJECT_ROOT}\\results\\safe_region\\mg_case\\min_state_res.pkl'

with open(res_path, 'rb') as f:  # 'rb' 表示二进制读取
    res_dict= pickle.load(f)
# 时间设置
deltaT = 5  # 分钟
T = 288  # 一天24小时，共288个时间点
time_points = np.arange(0, T * deltaT, deltaT) / 60  # 转为小时

# 将6个状态变量分开，分别提取
min_original_array = np.array(res_dict['min_original_list'])
min_approx_array = np.array(res_dict['min_approx_list'])

# 设置英文学术期刊风格
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 18,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "figure.figsize": (5, 8),
    "lines.linewidth": 1.5,
})

# 创建6行共享x轴的子图
fig, axes = plt.subplots(6, 1, sharex=True)
data_ranges = [[21,25]]*4+[[10,50],[10,40]]
y_lims = [[22.8, 23.4]]*4+[[9.9,10.5]]*2
# 绘制每个状态变量
for i in range(6):
    # data = (min_approx_array[:, i]-min_original_array[:, i])/(data_ranges[i][1]-data_ranges[i][0])
    # if i >=4:
    #     data = np.maximum(min_approx_array[:, i],min_original_array[:, i])
    data = min_approx_array[:, i]
    axes[i].plot(time_points, data, label=f"Original State {i+1}", color=f'C{i}')
    data = min_original_array[:, i]
    axes[i].plot(time_points, data, linestyle='--', label=f"Original State {i + 1}", color=f'C{i}')
    # axes[i].axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)

    y_min, y_max = y_lims[i][0], y_lims[i][1]
    axes[i].set_ylim(y_min, y_max)

    # 设置 y 轴刻度（3个点：最低、中间、最高）
    y_mid = (y_min + y_max) / 2
    axes[i].set_yticks([y_min, y_mid, y_max])
    # axes[i].legend(loc='upper right', fontsize=12)
    axes[i].grid(True)
    axes[i].spines['top'].set_visible(False)
    axes[i].spines['right'].set_visible(False)

# 设置x轴标签
axes[5].set_xlabel('Time [h]', fontsize=13)
axes[5].set_xlim(0, 24)  # 设置x轴范围为0到24小时
xticks = np.arange(0, 25, 4)  # 0, 4, 8, 12, 16, 20, 24
axes[5].set_xticks(xticks)
# 去掉网格和上右边框
for ax in axes:
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
# 调整布局
plt.tight_layout()
plt.show()


# T = 12
# Delta_t = 5 / 60
# params = {
#     'num_cont_tcl': 2,  # 连续型TCL数量改为2
#     'num_disc_tcl': 2,  # 离散型TCL数量改为2
#     'num_ess': 2,  # ESS数量改为2
#
#     # 电网参数
#     'C_L': 0.0,
#     'C_U': 30.0,
#
#     # 连续型TCL参数
#     'C_cont': [316.11, 140.56],  # 第一个保持原值，第二个使用原离散型的C值
#     'eta_cont': [4.0, 4.0],  # 第一个保持原值，第二个使用原离散型的eta值
#     'H_cont': [3.892, 0.92092],  # 第一个保持原值，第二个使用原离散型的H值
#     'p_cont_max_tcl': [17.4, 7.69],  # 第一个保持原值，第二个使用原离散型的最大功率值
#     'theta_min_cont': [21.0, 21.0],  # 最低温度相同
#     'theta_max_cont': [25.0, 25.0],  # 最高温度相同
#     'theta_set_cont': [23.0, 23.0],  # 设定温度相同
#
#     # 离散型TCL参数
#     'C_disc': [95.5, 170.42],  # 第一个保持原离散型值，第二个使用原连续型的C值
#     'eta_disc': [3.6, 3.6],  # 第一个保持原离散型值，第二个使用原连续型的eta值
#     'H_disc': [1.96, 2.10],  # 第一个保持原离散型值，第二个使用原连续型的H值
#     'p_disc_max_tcl': [15.73, 10.46],  # 第一个保持原离散型值，第二个使用原连续型的最大功率值
#     'theta_min_disc': [21.0, 21.0],  # 最低温度相同
#     'theta_max_disc': [25.0, 25.0],  # 最高温度相同
#     'theta_set_disc': [23.0, 23.0],  # 设定温度相同
#
#     # ESS参数
#     'eta_chg': [0.97, 0.98],  # 充电效率，第二个ESS设为0.95
#     'eta_dis': [0.98, 0.97],  # 放电效率，第二个ESS设为0.96
#     'pmax_chg_ess': [25.0, 10.0],  # 最大充电功率，第二个ESS设为40.0
#     'pmax_dis_ess': [25.0, 10.0],  # 最大放电功率，第二个ESS设为40.0
#     'e_min': [0.0, 0.0],  # 最小能量相同
#     'e_max': [50.0, 40.0],  # 最大能量，第二个ESS设为120.0
# }
# data_path = f'{PROJECT_ROOT}\\data\\profiles_data\\'
# data = np.load(data_path+'profiles_data.npz')
# full_p_bl, full_p_pv, full_theta_amb =20* data['load_data'], 40*data['pv_data'], data['temp_data']
# data = {}
# data['full_theta_amb'] = full_theta_amb
# data['full_p_bl'] = full_p_bl
# data['full_p_pv'] = full_p_pv
# start_time = 1
# sfc = safe_region_case()
# model_type = 'fullnet'
# case = sfc.build_mg_case(T=T, Delta_t=Delta_t, data=data, params=params, x_only=True, model_type=model_type,
#                          device=device, current_time=start_time)
# theta_amb_sample = case['metadata']['full_theta_amb'][start_time:start_time + T]
# p_bl_sample = case['metadata']['full_p_bl'][start_time:start_time + T]
# p_pv_sample = case['metadata']['full_p_pv'][start_time:start_time + T]
#
# # 归一化
# def _normalize(data_sample, data_range):
#     min_val, max_val = data_range
#     with np.errstate(divide='ignore', invalid='ignore'):
#         normalized = (data_sample - min_val) / (max_val - min_val)
#         normalized[max_val == min_val] = 0.0  # 处理除零情况
#     return normalized*2-1
# theta_amb_meta = _normalize(theta_amb_sample, (np.min(case['metadata']['full_theta_amb']), np.max(case['metadata']['full_theta_amb'])))
# p_bl_meta = _normalize(p_bl_sample,(np.min(case['metadata']['full_p_bl']), np.max(case['metadata']['full_p_bl'])))
# p_pv_meta = _normalize(p_pv_sample, (np.min(case['metadata']['full_p_pv']), np.max(case['metadata']['full_p_pv'])))
#
# input_data = np.hstack([p_bl_meta,p_pv_meta,theta_amb_meta],dtype=np.float32)
# # input_data = np.zeros(T*3,dtype=np.float32)
# fullnet = FullNet(dim_theta = case['params']['count'], A_init=case['A_hat'],b_init = case['b_hat'],n_hidden=128)
# fullnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\safe_region\\{case['casename']}\\fullnet_weights.pth', map_location=device))
# A_pred, b_pred = fullnet(torch.tensor(input_data))
# A_pred = A_pred[0].detach().cpu().numpy()
# b_pred = b_pred[0].detach().cpu().numpy()
#
# import matplotlib as mpl
# mpl.rcParams.update({
#     'font.family': 'serif',  # -serif 字体族（包含 Times New Roman）
#     'font.serif': ['Times New Roman'],  # 优先使用 Times New Roman
#     'font.size': 24,  # 全局基础字号设为 12
# })
# plt.figure(figsize=(3.346, 2.51))
#
# plotter = ShapeDrawer_2D()
# xlim = [-0.05,1.05]
# ylim = xlim
# facecolor = 'blue'
# edgecolor = 'blue'
#
# selected_dimensions = np.array([4,5])
# other_dimensions = np.setdiff1d(np.array(range(6)), selected_dimensions)
#
# plotter.plot_polygon(case['A_hat'][:,selected_dimensions], case['b_hat']-0.8*case['A_hat'][:,other_dimensions]@np.ones(4,dtype=np.float32),edgecolor=edgecolor,
#                      facecolor=facecolor, xlim=xlim, ylim=ylim,
#                      label=f'fullnet',
#                      # title=f'Training step = {0}',
#                      )
# facecolor = 'red'
# edgecolor = 'red'
# plotter.plot_polygon(A_pred[:,selected_dimensions] , b_pred-0.8*A_pred[:,other_dimensions]@np.ones(4,dtype=np.float32) ,edgecolor=edgecolor,
#                      facecolor=facecolor, xlim=xlim, ylim=ylim,
#                      label=f'fullnet',
#                      # title=f'Training step = {0}',
#                      )
# # plotter.show()
# figure_folder_name = f'{PROJECT_ROOT}\\results\\safe_region\\mg_case\\figures\\'
# plotter.save(figure_folder_name+f'fullnet.svg', show_legend=False)