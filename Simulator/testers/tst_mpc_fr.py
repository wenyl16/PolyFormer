import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from Simulator.Plotter import ShapeDrawer_2D
import os
import torch
from Simulator.cases.MPC_case import MPCcase
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type = 'pretrainnet'
mpc_original = MPCcase()
mpc_original.build_simplecase(include_obj=True)

mpc = MPCcase()
case = mpc.build_simplecase_fr(model_type=model_type)


pretrainnet =  PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=False)
pretrainnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\MPC\\{case['casename']}\\pretrainnet_weights_fr.pth', map_location=device))
A_pretrained,b_pretrained = pretrainnet()
A_apx_fr = A_pretrained[0].detach().cpu().numpy()
b_apx_fr = b_pretrained[0].detach().cpu().numpy()

plotter = ShapeDrawer_2D()

dtheta = -0.3
xlim = [-0.5, 1.5]
ylim = [-0.5, 1.5]
b = np.array([1, 1, 0, 0, 1.5, -0.5, 0.7+dtheta , 0.7+dtheta ])

figure_folder_name = f'{PROJECT_ROOT}\\results\\MPC\\{case['casename']}\\figures\\comparison\\'
os.makedirs(figure_folder_name, exist_ok=True)


plotter.plot_polygon(A_apx_fr, b_apx_fr,
                     facecolor='green', xlim=xlim, ylim=ylim,
                     label=f'pretrain',
                     # title=f'Training step = {0}',
                     )