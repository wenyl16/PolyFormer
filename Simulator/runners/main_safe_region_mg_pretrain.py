import os

import numpy as np
import scipy

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.safe_region_case import safe_region_case
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parallel= False
is_epigraph = False

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
    'e_min': [0.0, 0.0],  # 最小能量相同
    'e_max': [50.0, 40.0],  # 最大能量，第二个ESS设为120.0
}
data_path = f'{PROJECT_ROOT}\\data\\profiles_data\\'
data = np.load(data_path+'profiles_data.npz')
full_p_bl, full_p_pv, full_theta_amb = 20*data['load_data'], 40*data['pv_data'], data['temp_data']

# total_periods = int(240 / Delta_t)
# full_theta_amb = np.random.uniform(-5, 5, total_periods)  # 环境温度
# full_p_bl = np.random.uniform(5, 15, total_periods)  # 基础负荷
# full_p_pv = np.random.uniform(0, 35, total_periods)  # 光伏发电
data = {}
data['full_theta_amb'] = full_theta_amb
data['full_p_bl'] = full_p_bl
data['full_p_pv'] = full_p_pv

model_type = 'pretrainnet'
# model_type = 'fullnet'

sfc = safe_region_case()
case = sfc.build_mg_case(T=T, Delta_t=Delta_t, data=data, params = params, x_only=True, model_type=model_type,device=device)
if model_type=='pretrainnet':
    n_train = 500
    model   = PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=is_epigraph, device = device)
elif model_type == 'fullnet':
    n_train = 20
    model= FullNet(dim_theta = case['params']['count'], A_init=case['A_hat'],b_init = case['b_hat'],is_epigraph=is_epigraph, n_hidden=128,device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)

trainer.configure(**case['trainer_configure'])
# trainer.configure(lr_A = 1e-5)
# trainer.configure(lr_b = 1e-2)
trainer.configure(lr = 1e-1)
trainer.initialize()
trainer.train(n_train = n_train , params_data=case['params'],parallel=parallel)


# trainer.configure(**case['trainer_configure'])
# # trainer.configure(lr_A = 1e-5)
# # trainer.configure(lr_b = 1e-2)
# trainer.configure(lr = 2e-3)
# trainer.initialize()
# trainer.train(n_train = n_train , params_data=case['params'],parallel=parallel)
#
# trainer.configure(rate_opt_feas = 1e-1, lr = 2e-4)
# trainer.initialize()
# trainer.train(n_train = 1 , params_data=case['params'],parallel=parallel)

torch.save(model.state_dict(), case['result_path'])
# path = f'{PROJECT_ROOT}\\results\\safe_region\\{case['casename']}\\training_history.pkl'
# import pickle
# with open(path, "wb") as f:
#     pickle.dump(case['errorcalculator'].training_history, f)