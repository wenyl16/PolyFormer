import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
import pyomo.environ as pyo
import matplotlib.pyplot as plt
from Simulator.Plotter import ShapeDrawer_2D
import os
import torch
from Simulator.cases.safe_region_case import safe_region_case
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type = 'fullnet'

T = 12
Delta_t = 5 / 60
params = {
    'num_cont_tcl': 2,  # 连续型TCL数量改为2
    'num_disc_tcl': 2,  # 离散型TCL数量改为2
    'num_ess': 2,  # ESS数量改为2

    # 电网参数
    'C_L': 0.0,
    'C_U': 30.0,

    # 连续型TCL参数
    'C_cont': [316.11, 140.56],  # 第一个保持原值，第二个使用原离散型的C值
    'eta_cont': [4.0, 4.0],  # 第一个保持原值，第二个使用原离散型的eta值
    'H_cont': [3.892, 0.92092],  # 第一个保持原值，第二个使用原离散型的H值
    'p_cont_max_tcl': [17.4, 7.69],  # 第一个保持原值，第二个使用原离散型的最大功率值
    'theta_min_cont': [21.0, 21.0],  # 最低温度相同
    'theta_max_cont': [25.0, 25.0],  # 最高温度相同
    'theta_set_cont': [23.0, 23.0],  # 设定温度相同

    # 离散型TCL参数
    'C_disc': [95.5, 170.42],  # 第一个保持原离散型值，第二个使用原连续型的C值
    'eta_disc': [3.6, 3.6],  # 第一个保持原离散型值，第二个使用原连续型的eta值
    'H_disc': [1.96, 2.10],  # 第一个保持原离散型值，第二个使用原连续型的H值
    'p_disc_max_tcl': [15.73, 10.46],  # 第一个保持原离散型值，第二个使用原连续型的最大功率值
    'theta_min_disc': [21.0, 21.0],  # 最低温度相同
    'theta_max_disc': [25.0, 25.0],  # 最高温度相同
    'theta_set_disc': [23.0, 23.0],  # 设定温度相同

    # ESS参数
    'eta_chg': [0.97, 0.98],  # 充电效率，第二个ESS设为0.95
    'eta_dis': [0.98, 0.97],  # 放电效率，第二个ESS设为0.96
    'pmax_chg_ess': [25.0, 10.0],  # 最大充电功率，第二个ESS设为40.0
    'pmax_dis_ess': [25.0, 10.0],  # 最大放电功率，第二个ESS设为40.0
    'e_min': [10.0, 10.0],  # 最小能量相同
    'e_max': [50.0, 40.0],  # 最大能量，第二个ESS设为120.0
}
data_path = f'{PROJECT_ROOT}\\data\\profiles_data\\'
data = np.load(data_path+'profiles_data.npz')
full_p_bl, full_p_pv, full_theta_amb =20* data['load_data'], 40*data['pv_data'], data['temp_data']

# total_periods = int(240 / Delta_t)
# full_theta_amb = np.random.uniform(-5, 5, total_periods)  # 环境温度
# full_p_bl = np.random.uniform(5, 15, total_periods)  # 基础负荷
# full_p_pv = np.random.uniform(0, 35, total_periods)  # 光伏发电
data = {}
data['full_theta_amb'] = full_theta_amb
data['full_p_bl'] = full_p_bl
data['full_p_pv'] = full_p_pv


sfc = safe_region_case()
is_epigraph = False
case = sfc.build_mg_case(T=T, Delta_t=Delta_t, data=data, params = params, x_only=True, model_type=model_type,device=device)

fullnet = FullNet(dim_theta = case['params']['count'], A_init=case['A_hat'],b_init = case['b_hat'],n_hidden=128)
fullnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\safe_region\\{case['casename']}\\fullnet_weights.pth', map_location=device))


def _normalize(data_sample, data_range):
    min_val, max_val = data_range
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = (data_sample - min_val) / (max_val - min_val)
        normalized[max_val == min_val] = 0.0  # 处理除零情况
    return normalized*2-1

# 测试误判率
# start_time = 1
# # 提取该时刻的决策窗口数据
# theta_amb_sample = case['metadata']['full_theta_amb'][start_time:start_time + T]
# p_bl_sample = case['metadata']['full_p_bl'][start_time:start_time + T]
# p_pv_sample = case['metadata']['full_p_pv'][start_time:start_time + T]
# case = sfc.build_mg_case(T=T, Delta_t=Delta_t, data=data, params = params, x_only=True, model_type=model_type,device=device,current_time=start_time)
#
# # 归一化
# theta_amb_meta = _normalize(theta_amb_sample, (np.min(case['metadata']['full_theta_amb']), np.max(case['metadata']['full_theta_amb'])))
# p_bl_meta = _normalize(p_bl_sample,(np.min(case['metadata']['full_p_bl']), np.max(case['metadata']['full_p_bl'])))
# p_pv_meta = _normalize(p_pv_sample, (np.min(case['metadata']['full_p_pv']), np.max(case['metadata']['full_p_pv'])))
#
# input_data = np.hstack([p_bl_meta,p_pv_meta,theta_amb_meta],dtype=np.float32)
# # input_data = np.zeros(T*3,dtype=np.float32)
#
# A_pred, b_pred = fullnet(torch.tensor(input_data))
# A_pred = A_pred[0].detach().cpu().numpy()
# b_pred = b_pred[0].detach().cpu().numpy()
# import time
# N_test = 1000
# apx_total_time = 0
# org_total_time = 0
# count_fault1 = 0
# count_fault2 = 0
# for i in range(N_test):
#     print(i)
#     x = np.random.rand(6)
#     # 判断是否满足Ax <= b
#     start_t = time.time()
#     condition = np.all(A_pred @ x <= b_pred)
#     # 记录结束时间
#     end_t = time.time()
#     # 计算总耗时（秒）
#     apx_total_time += end_t - start_t
#     original_model = case['errorcalculator']
#     start_t = time.time()
#     x_proj = original_model.project(target=x, to_approx = False)
#     end_t = time.time()
#     org_total_time += end_t - start_t
#     error = np.linalg.norm(x_proj-x)**2
#     if condition and (error>1e-5):
#         count_fault1+=1
#     elif (not condition) and (error<1e-5):
#         count_fault2+=1
#
# print(apx_total_time/N_test,org_total_time/N_test,count_fault1/N_test,count_fault2/N_test)

# 测试全天的最低初始状态
results = {'min_original_list':[],
'min_approx_list':[]}

for start_time in range(12*24*4,12*24*5):
    # start_time = 1
    # 提取该时刻的决策窗口数据
    theta_amb_sample = case['metadata']['full_theta_amb'][start_time:start_time + T]
    p_bl_sample = case['metadata']['full_p_bl'][start_time:start_time + T]
    p_pv_sample = case['metadata']['full_p_pv'][start_time:start_time + T]
    case = sfc.build_mg_case(T=T, Delta_t=Delta_t, data=data, params = params, x_only=True, model_type=model_type,device=device,current_time=start_time)

    # 归一化
    theta_amb_meta = _normalize(theta_amb_sample, (np.min(case['metadata']['full_theta_amb']), np.max(case['metadata']['full_theta_amb'])))
    p_bl_meta = _normalize(p_bl_sample,(np.min(case['metadata']['full_p_bl']), np.max(case['metadata']['full_p_bl'])))
    p_pv_meta = _normalize(p_pv_sample, (np.min(case['metadata']['full_p_pv']), np.max(case['metadata']['full_p_pv'])))

    input_data = np.hstack([p_bl_meta,p_pv_meta,theta_amb_meta],dtype=np.float32)
    # input_data = np.zeros(T*3,dtype=np.float32)
    A_pred, b_pred = fullnet(torch.tensor(input_data))
    A_pred = A_pred[0].detach().cpu().numpy()
    b_pred = b_pred[0].detach().cpu().numpy()
    # print(A_pred,b_pred)
    # 测试最坏初值
    dim_x = 6
    rated_minimum_original = -case['b_hat'][-6:]

    case['errorcalculator'].update_polytope(A_hat=A_pred,b_hat=b_pred)
    case['errorcalculator'].approx_model.var_proj_nonneg_constraints = pyo.Constraint(
        case['errorcalculator'].approx_model.var_proj.index_set(),
        rule=lambda m, *idx: m.var_proj[idx] >= 0
    )
    C = np.eye(dim_x)
    rated_minimum_approx = []
    for i in range(dim_x):
        rated_minimum_approx.append(case['errorcalculator'].optimize_direction(direction= C[i,:], in_approx = True)[i])
    mins = params['theta_min_cont']+params['theta_min_disc']+params['e_min']
    maxes = params['theta_max_cont']+params['theta_max_disc']+params['e_max']
    minimum_original = [
        d * (xmax - xmin) + xmin
        for d, xmax, xmin in zip(rated_minimum_original, maxes, mins)
    ]
    minimum_approx = [
        d * (xmax - xmin) + xmin
        for d, xmax, xmin in zip(rated_minimum_approx, maxes, mins)
    ]
    results['min_original_list'].append(minimum_original)
    results['min_approx_list'].append(minimum_approx)

path = f'{PROJECT_ROOT}\\results\\safe_region\\{case['casename']}\\min_state_res.pkl'
import pickle
with open(path, "wb") as f:
    pickle.dump(results, f)

#测试固定可行集的误差

# from Simulator.Plotter import ErrorVisualizer
# n_test = 50
# ev = ErrorVisualizer()
# case['errorcalculator'].update_polytope(A_hat=A_pred,b_hat=b_pred)
# ev.compute_errors(case['errorcalculator'],num_sample=50)
# error_param = ev.error_history
#
# ev = ErrorVisualizer()
# case['errorcalculator'].update_polytope(A_hat=case['A_hat'],b_hat=case['b_hat'])
# ev.compute_errors(case['errorcalculator'],num_sample=50)
# error_fix = ev.error_history
#
# print(np.mean(error_param['error_feas'][0] - error_fix['error_feas'][0])/np.mean(error_param['error_feas'][0]))
# print(np.mean(error_param['error_opt'][0] - error_fix['error_opt'][0])/np.mean(error_param['error_opt'][0]))


