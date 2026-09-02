import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
import matplotlib.pyplot as plt
from Simulator.Plotter import ShapeDrawer_2D
import torch
from Simulator.cases.safe_region_case import safe_region_case
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type = 'pretrainnet'
sfc = safe_region_case()


# case = sfc.build_case_1(x_only=True, model_type=model_type)
# dim_theta = 1
# pretrainnet =  PreTrainNet(case['A_hat'],case['b_hat'])
# pretrainnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\safe_region\\{case['casename']}\\pretrainnet_weights_fr.pth', map_location=device))
# A_pretrained,b_pretrained = pretrainnet()
# A_pretrained = A_pretrained[0].detach().cpu().numpy()
# b_pretrained = b_pretrained[0].detach().cpu().numpy()
#
# fullnet = FullNet(dim_theta = dim_theta, A_init=A_pretrained,b_init = b_pretrained,n_hidden=128)
# fullnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\safe_region\\{case['casename']}\\fullnet_weights_fr.pth', map_location=device))
# import matplotlib as mpl
# mpl.rcParams.update({
#     'font.family': 'serif',  # -serif 字体族（包含 Times New Roman）
#     'font.serif': ['Times New Roman'],  # 优先使用 Times New Roman
#     'font.size': 24,  # 全局基础字号设为 12
# })
# plt.figure(figsize=(3.346, 2.51))
#
#
# dtheta = [-1.0]
# sfc.model.theta.value += dtheta[0]
#
# figure_folder_name = f'{PROJECT_ROOT}\\results\\safe_region\\{case['casename']}\\figures\\comparison\\'
# os.makedirs(figure_folder_name, exist_ok=True)
# plotter = ShapeDrawer_2D()
# xlim = [-7, 7]
# ylim = [-3,3]
# facecolor = 'blue'
# edgecolor = 'blue'
#
#
# bp = sfc._cal_bound_points(n=100)
# plotter.plot_convex_hull(bp, alpha=0.3, facecolor='blue',
#                          edgecolor='blue',
#                          label='Original region')
# A_pred, b_pred = fullnet(torch.tensor(dtheta))
# A_pred = A_pred[0].detach().cpu().numpy()
# b_pred = b_pred[0].detach().cpu().numpy()
#
# facecolor = 'red'
# edgecolor = 'red'
# plotter.plot_polygon(A_pred , b_pred ,edgecolor=edgecolor,
#                      facecolor=facecolor, xlim=xlim, ylim=ylim,
#                      label=f'fullnet',
#                      # title=f'Training step = {0}',
#                      )
# plotter.save(figure_folder_name+f'fullnet dtheta = {dtheta[0]:.2e}.svg', show_legend=False)
#
#
