import os


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
import matplotlib.pyplot as plt
from Simulator.Plotter import ShapeDrawer_2D
import os
import torch
from Simulator.cases.DRCC_case import DRCCModelBuilder
from Simulator import PROJECT_ROOT
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

np.random.seed(0)
N_var = 400
N_levels = 8
N_samples = 1280
dim_theta = 4
data_path = f'{PROJECT_ROOT}/data/DRCC/r_samples_x{N_var}g{N_levels}s{N_samples}'

r_samples = pd.read_csv(data_path+'.csv')

N_levels = len(set(r_samples['group']))
params = {'r_samples':r_samples,
          # 'R_min':[None]*N_levels,
          'R_limits':[(-0.1,0.1)]*N_levels,
          # 'eps':[None]*N_levels,
          # 'rho':[None]*N_levels,
          # 'max_group_total':[None]*N_levels
          }

model_type = 'fullnet'
#构建基础模型
portfolio =  DRCCModelBuilder(params)
apx_data = {}
params_meta_data = {}
dim = 50
A = np.ones([4 * dim + 2, dim])
b = np.zeros(4 * dim + 2)
pred_models = {}
for group_idx in range(portfolio.group_number):
    pred_models[group_idx] = FullNet(dim_theta=4, n_hidden=128, A_init=A, b_init=b)
    pred_models[group_idx].load_state_dict(
        torch.load(
            f'{PROJECT_ROOT}\\results\\DRCC\\x{N_var}g{N_levels}s{N_samples}\\g{group_idx}\\{model_type}_weights.pth',
            map_location=device))
    A_pred, b_pred = pred_models[group_idx](torch.tensor([0,0,0,0], dtype=torch.float32))
    A_pred = A_pred[0].detach().cpu().numpy()
    b_pred = b_pred[0].detach().cpu().numpy()
    apx_data[group_idx] = {'A': A_pred, 'b': b_pred}

portfolio.build_apx_model(apx_data=apx_data)
print(portfolio.solve(is_apx=True, tee = False))