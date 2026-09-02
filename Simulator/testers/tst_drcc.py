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
N_var = 300
N_levels = 5
N_samples = 900
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

#构建基础模型
portfolio =  DRCCModelBuilder(params)
apx_data = {}
params_meta_data = {}
for group_idx in range(portfolio.group_number):
    case = portfolio.build_drcc_train(portfolio.group_dataset[group_idx],device=device) #这个model_type不重要
    apx_data[group_idx] = {'A': case['A_hat'], 'b': case['b_hat']}

    R_mean = np.mean(portfolio.group_dataset[group_idx]['samples'])
    R_std = np.std(portfolio.group_dataset[group_idx]['samples'])
    R_min_range = np.array(
        (max(params['R_limits'][group_idx][0], R_mean - 1.6 * R_std), max(params['R_limits'][group_idx][0], R_mean - 1.4 * R_std)))
    rho_power_range = (-6, -3.5)
    eps_range = np.array((0.07, 0.13))
    max_total_range = np.array((min(1.0, 1.0 / portfolio.group_number + 0.2), min(1.0, 1.0 / portfolio.group_number + 0.4)))



    params_meta_data[group_idx] = {'R_min_range':R_min_range, 'rho_power_range':rho_power_range,
                                   'eps_range':eps_range,'max_total_range':max_total_range}

portfolio.build_original_model()
portfolio.build_apx_model(apx_data=apx_data)


#定义近似模型（pretrain，full）
# model_type = 'pretrainnet'
model_type = 'fullnet'
pred_models = {}

for group_idx in range(portfolio.group_number):
    if model_type == 'pretrainnet':
        pred_models[group_idx] =  PreTrainNet(apx_data[group_idx]['A'],apx_data[group_idx]['b'])
    elif model_type == 'fullnet':
        dim_theta = 4
        pred_models[group_idx] = FullNet(dim_theta=dim_theta, n_hidden=128, A_init=apx_data[group_idx]['A'], b_init=apx_data[group_idx]['b'])
    pred_models[group_idx].load_state_dict(
            torch.load(f'{PROJECT_ROOT}\\results\\DRCC\\{case['casename']}\\g{group_idx}\\{model_type}_weights.pth',
                       map_location=device))

#生成参数扰动样本
def sample_parameters(params_meta_data):
    sample = {}
    for group_idx, meta_data in params_meta_data.items():
        # np.random.seed(42)
        R_min = np.random.uniform(meta_data['R_min_range'][0], meta_data['R_min_range'][1])
        # eps = np.random.uniform(meta_data['eps_range'][0], meta_data['eps_range'][1])
        eps = np.mean(meta_data['eps_range'])
        # max_total = np.random.uniform(meta_data['max_total_range'][0], meta_data['max_total_range'][1])
        max_total = np.mean(meta_data['max_total_range'])
        rho_exp = np.random.uniform(meta_data['rho_power_range'][0], meta_data['rho_power_range'][1])
        rho = 10 ** rho_exp
        sample[group_idx]={
            "R_min": R_min,
            "rho": rho,
            "eps": eps,
            "max_group_total": max_total
        }
    return sample


#开始测试
r_test_samples = pd.read_csv(data_path + '_test.csv')
N_test = 1 if model_type=='pretrainnet' else 20
res_list = []
eps_points = np.linspace(0.03, 0.18, 31)
for eps in eps_points:
    print(eps)
    res_dict = {'eps':[],'obj_org': [], 'obj_apx': [], 'vr_org': [], 'vr_apx': [], 'vv_org': [], 'vv_apx': []}
    for i in range(N_test):
        print(i)
        new_apx_data = {}
        if model_type == 'pretrainnet':
            for group_idx in range(portfolio.group_number):
                A_pred, b_pred = pred_models[group_idx]()
                A_pred = A_pred[0].detach().cpu().numpy()
                b_pred = b_pred[0].detach().cpu().numpy()

                new_apx_data[group_idx] = {"A": A_pred, "b": b_pred}
        elif model_type == 'fullnet':
            sample = sample_parameters(params_meta_data)
            for group_idx in range(portfolio.group_number):
                sample[group_idx]['eps'] = eps
            delta_dict = portfolio.update_original_params(sample)

            # delta_dict[0][2] -= 0.02 #Rmin, rho, eps,  max_group_total
            # delta_dict[8][1] += 0.06
            # delta_dict[9][1] += 0.03
            # delta_dict[7][0] -= 0.2
            # delta_dict[8][0] -= 0.2
            # delta_dict[9][0] -= 0.2

            for group_idx in range(portfolio.group_number):

                # if group_idx>=7:
                #     delta_dict[group_idx][0] -= 0.3
                #     delta_dict[group_idx][1] += 0.22
                A_pred, b_pred = pred_models[group_idx](torch.tensor(delta_dict[group_idx], dtype=torch.float32))
                A_pred = A_pred[0].detach().cpu().numpy()
                b_pred = b_pred[0].detach().cpu().numpy()

                # delta_dict[group_idx][0] -= 0.2
                # A_pred_1, b_pred_1 = pred_models[group_idx](torch.tensor(delta_dict[group_idx], dtype=torch.float32))
                # A_pred_1 = A_pred_1[0].detach().cpu().numpy()
                # b_pred_1 = b_pred_1[0].detach().cpu().numpy()
                # A_pred = np.vstack([np.eye(20),-np.eye(20)])
                # b_pred = np.zeros(40)
                # print(max(np.abs(b_pred-b_pred_1)))

                new_apx_data[group_idx] = {"A": A_pred, "b": b_pred}

        portfolio.update_apx_params(new_apx_data)

        solution = portfolio.solve(is_apx=False, tee=False)
        solution_apx = portfolio.solve(is_apx=True, tee=False)

        res = portfolio.evaluate_solution(x_vals=solution['x'],r_test_samples = r_test_samples)
        res_apx = portfolio.evaluate_solution(x_vals=solution_apx['x'],r_test_samples = r_test_samples)
        res_dict['eps'].append(eps)
        res_dict['obj_org'].append(res['test_objective'])
        res_dict['obj_apx'].append(res_apx['test_objective'])
        res_dict['vr_org'].append(np.mean([res['violation_probabilities'][group_idx] for group_idx in range(portfolio.group_number)]))
        res_dict['vr_apx'].append(np.mean([res_apx['violation_probabilities'][group_idx] for group_idx in range(portfolio.group_number)]))
        res_dict['vv_org'].append(np.mean([res['violation_value'][group_idx] for group_idx in range(portfolio.group_number)]))
        res_dict['vv_apx'].append(np.mean([res_apx['violation_value'][group_idx] for group_idx in range(portfolio.group_number)]))
    res_list.append(res_dict)

res_path = f'{PROJECT_ROOT}\\results\\DRCC\\{case['casename']}\\test_result.pkl'
import pickle
with open(res_path, 'wb') as f:  # 'wb' 表示二进制写入
    pickle.dump(res_list, f)
# print(res_list)
