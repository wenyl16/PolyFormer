import numpy as np
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
data = np.load('profiles_data.npz')
full_p_bl, full_p_pv, full_theta_amb =20* data['load_data'], 40*data['pv_data'], data['temp_data']
