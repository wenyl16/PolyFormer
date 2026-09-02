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

case = mpc.build_simplecase_obj(model_type=model_type, A_fr=A_apx_fr, b_fr=b_apx_fr)
pretrainnet =  PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=True)
pretrainnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\MPC\\{case['casename']}\\pretrainnet_weights_obj.pth', map_location=device))
A_pretrained,b_pretrained = pretrainnet()
A_apx_obj = A_pretrained[0].detach().cpu().numpy()
b_apx_obj = b_pretrained[0].detach().cpu().numpy()

np.random.seed(32)
x0 = [1,-0.5]
x_history_original, u_history_original = mpc_original.receding_horizon_control(x0, steps=1)
# 绘制结果
# mpc.plot_results(x_history_original, u_history_original)


x_history_apx, u_history_apx = mpc.receding_horizon_control(x0, steps=1, is_apx=True, apx_params={'A_fr':A_apx_fr,
                                                                                                   'b_fr':b_apx_fr,
                                                                                                   'A_obj': A_apx_obj,
                                                                                                   'b_obj': b_apx_obj,
                                                                                                   # 'A_obj': case['A_hat'],
                                                                                                   # 'b_obj': case['b_hat'],
                                                                                                   })
# # 绘制结果
# mpc.plot_results(x_history_apx, u_history_apx)