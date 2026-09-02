import os
import numpy as np
import pandas as pd

from Simulator.Plotter import ErrorVisualizer
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from Simulator.cases.aggregation_case import Aggregator
from Simulator import  PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model_type = 'pretrainnet'


agg = Aggregator(seed=0, discrete_rate = 0.1)

agg.gen_EV(60)
agg.gen_TCL(40)
agg.gen_ESS(5)

visualizer = ErrorVisualizer()
np.random.seed(0)
num_sample = 40
case = agg.case_aggregator(model_type=model_type)
errorcalculator = case['errorcalculator']

# folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\discrete\\figures'
# # folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\figures'
# files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')],
#                key=lambda x: int(x.split('_')[1].split('.')[0]))
# iteration_list = []
# for i in range(0, len(files), 2):  # 假设每两个文件是一组(A和b)
#     if i + 1 < len(files):
#         print(i)
#         a_file = [f for f in files if f.startswith('A_')][i // 2]
#         b_file = [f for f in files if f.startswith('b_')][i // 2]
#         # 读取并打印
#         A = pd.read_csv(os.path.join(folder_path, a_file), header=None).values
#         b = pd.read_csv(os.path.join(folder_path, b_file), header=None).values.flatten()
#         i_iter = a_file.split('_')[1].split('.')[0]
#         iteration_list.append(i_iter)
#         errorcalculator.update_polytope(A_hat=A, b_hat=b)
#         visualizer.compute_errors(errorcalculator, num_sample=num_sample)
# result_path = f'{PROJECT_ROOT}\\results\\aggregation\\discrete\\error_history.pkl'
# # result_path = f'{PROJECT_ROOT}\\results\\aggregation\\error_history.pkl'
# import pickle
# with open(result_path, "wb") as f:
#     pickle.dump(visualizer.error_history, f)


# folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\data_cube\\'
# A = pd.read_csv(os.path.join(folder_path, 'A_cube.csv'), header=None).values
# b = pd.read_csv(os.path.join(folder_path, 'b_cube.csv'), header=None).values.flatten()
# case = agg.case_aggregator(A = A, b = b, model_type=model_type)
# errorcalculator = case['errorcalculator']
# visualizer.compute_errors(errorcalculator, num_sample=num_sample)
# # print(visualizer.error_history)
# result_path = f'{PROJECT_ROOT}\\results\\aggregation\\error_history_cube.pkl'
# import pickle
# with open(result_path, "wb") as f:
#     pickle.dump(visualizer.error_history, f)
#
# folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\data_cube\\'
# A = pd.read_csv(os.path.join(folder_path, 'A_cube.csv'), header=None).values
# b = pd.read_csv(os.path.join(folder_path, 'b_cube.csv'), header=None).values.flatten()
# case = agg.case_aggregator(A = A, b = b, model_type=model_type)
# agg.count_complexity()
#
# folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\discrete\\figures'
# # folder_path = f'{PROJECT_ROOT}\\results\\aggregation\\figures'
# files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')],
#                key=lambda x: int(x.split('_')[1].split('.')[0]))
# iteration_list = []
# i=len(files)-2
# a_file = [f for f in files if f.startswith('A_')][i // 2]
# b_file = [f for f in files if f.startswith('b_')][i // 2]
# # 读取并打印
# A = pd.read_csv(os.path.join(folder_path, a_file), header=None).values
# b = pd.read_csv(os.path.join(folder_path, b_file), header=None).values.flatten()
# errorcalculator.update_polytope(A_hat=A, b_hat=b)


# T = 24
# n_test = 10
# actions_rand = np.random.rand(n_test//2,T)
# actions_feas = np.array([
#     [0.        , 0.12227902, 0.13093164, 0.70359684, 0.9154062 , 0.93361783,
#      0.36337373, 0.06521021, 0.05479282, 0.63237062, 0.7912069 , 0.86521444,
#      0.05581981, 0.07096851, 0.87142061, 0.85252465, 0.88020974, 0.08594851,
#      0.08767512, 0.20478644, 0.09248264, 0.09145326, 0.97659704, 0.97859177],
#     [0.19100453, 0.12279355, 0.90858855, 0.109034  , 0.06544886, 0.86499908,
#      0.93772923, 0.0570253 , 0.86661712, 0.05768608, 0.88032965, 0.07912243,
#      0.0584331 , 0.86907194, 0.06687999, 0.20122116, 0.87751526, 0.82559947,
#      0.06194041, 0.90767717, 0.07999532, 0.08916374, 0.9709392 , 0.97510207],
#     [0.20748169, 0.92971945, 0.09937248, 0.11703826, 0.06858586, 0.94161029,
#      0.50972595, 0.91208238, 0.06187957, 0.29967297, 0.0834183 , 0.86928718,
#      0.43591326, 0.86098374, 0.06458626, 0.85196676, 0.8692467 , 0.08364255,
#      0.0695082 , 0.90358179, 0.0898512 , 0.97320738, 0.10193014, 0.7087338 ],
#     [0.        , 0.15014873, 0.11172917, 0.91623611, 0.27344514, 0.94273403,
#      0.03857266, 0.91022953, 0.87379364, 0.06065246, 0.86626798, 0.06748514,
#      0.07085175, 0.86053583, 0.47196061, 0.8525117 , 0.08531828, 0.07748523,
#      0.9028976 , 0.06624319, 0.97188003, 0.09374322, 0.36671871, 0.983121  ],
#     [0.        , 0.91357289, 0.07732961, 0.90694771, 0.05016652, 0.89841224,
#      0.04308278, 0.07009375, 0.88105151, 0.06470593, 0.32351372, 0.87187624,
#      0.86556066, 0.0795814 , 0.8879796 , 0.09712649, 0.0847747 , 0.8858776,
#      0.06859193, 0.89445903, 0.0874217 , 0.971029  , 0.10094444, 0.75249288]
# ])
# actions = np.vstack([actions_feas,actions_rand])
# import time
# import pyomo.environ as pyo
# errorcalculator.original_model.given_value_max = pyo.Constraint(range(T), rule=lambda model, t: model.var_proj[t] <= model.x_apx[t]+1e-5)
# errorcalculator.original_model.given_value_min = pyo.Constraint(range(T), rule=lambda model, t: model.var_proj[t] >= model.x_apx[t]-1e-5)
# errorcalculator.original_model.min_error.deactivate()
# errorcalculator.original_model.min_direction.deactivate()
# errorcalculator.original_model.zero = pyo.Objective(expr=0,sense=pyo.minimize)
# judge_start = time.perf_counter()
# for action in actions:
#     # print(np.all(A@action<=b+1e-5))#统计近似可行域的判断时间
#
#     # 统计原始可行域的判断时间
#     for t in range(T):
#         errorcalculator.original_model.x_apx[t].value = action[t]
#     print(errorcalculator.solver.solve(errorcalculator.original_model))
# judge_end = time.perf_counter()
# print('总耗时',(judge_end-judge_start)/n_test)

# result_path = f'{PROJECT_ROOT}\\results\\aggregation\\discrete\\error_history.pkl'
# result_path = f'{PROJECT_ROOT}\\results\\aggregation\\error_history.pkl'
# import pickle
# with open(result_path, "wb") as f:
#     pickle.dump(visualizer.error_history, f)