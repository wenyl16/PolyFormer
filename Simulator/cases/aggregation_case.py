import numpy as np
import pandas as pd
from pathlib import Path

from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator
import pyomo.environ as pyo
from Simulator.Plotter import ErrorVisualizer
import os

class Aggregator:
    def __init__(self, seed=None,T = 24, data = None, discrete_rate = 0.0):
        """
        初始化设备参数生成器

        参数:
            seed (int, optional): 随机种子，用于复现结果
        """

        self.rng = np.random.default_rng(seed)
        self.discrete_EV = 0
        self.discrete_TCL = 0

        if data is None:
            self.EV_list = []
            self.TCL_list = []
            self.ESS_list = []
        else:
            self.EV_list = data['EV_list']
            self.discrete_EV = sum(self.EV_list[i]['is_discrete'] for i in range(len(self.EV_list)))

            self.TCL_list = data['TCL_list']
            self.discrete_TCL = sum(self.TCL_list[i]['is_discrete'] for i in range(len(self.TCL_list)))

            self.ESS_list = data['ESS_list']
        self.T = T
        self.deltaT = 24/self.T

        self.data_path = PROJECT_ROOT / 'data'
        data = np.load(self.data_path / 'profiles_data' / 'profiles_data.npz')
        data_deltaT = 5/60
        self.discrete_rate = discrete_rate
        self.theta_amb = data['temp_data'][0:int(self.T*self.deltaT/data_deltaT):int(self.deltaT/data_deltaT)]
    def gen_ESS(self, N):
        for i in range(N):
            self.ESS_list.append({
                'Pchg': self.rng.choice([25.,50.],p=[0.5,0.5]),
                'init_SOC': self.rng.uniform(0.3,0.7),
                'eta_chg': self.rng.uniform(0.95,0.98),
                'eta_dis': self.rng.uniform(0.96, 0.98),
                'B':self.rng.choice([100.,200.],p=[0.5,0.5]),
            })
        print('ESS generated')

    def gen_TCL(self, N, n_discrete=None):
        hp_df = pd.read_csv(self.data_path / 'aggregator_data' / 'ZH_buildings.csv').head(N)
        if n_discrete is not None and not 0 <= n_discrete <= N:
            raise ValueError("n_discrete must be between 0 and N")
        discrete_flags = None
        if n_discrete is not None:
            discrete_flags = np.zeros(N, dtype=int)
            if n_discrete:
                discrete_flags[self.rng.choice(N, size=n_discrete, replace=False)] = 1
        COP = 3.85
        for index, row in hp_df.iterrows():
            # 提取必要的字段
            HBLD = row['HBLD']
            CBLD = row['CBLD']
            PRT = row['PRT']
            is_discrete = (
                int(discrete_flags[index])
                if discrete_flags is not None
                else self.rng.choice([0, 1], p=[1-self.discrete_rate, self.discrete_rate])
            )
            self.discrete_TCL += is_discrete
            # 创建字典并添加到列表
            self.TCL_list.append({
                'H': HBLD,  # 热容
                'C': CBLD,  # 热阻
                'temp_min':17.0,
                'temp_max':23.0,
                'temp_set': 20.0,
                'Qmax': PRT,  # 热功率
                'COP': COP,  # COP系数
                'is_discrete': is_discrete,
            })
        print('TCL generated')
    def gen_EV(self, N, n_discrete=None):
        """
        生成N个电动汽车(EV)的充电需求参数
        完全按照MATLAB版本逻辑实现

        参数:
            N (int): 要生成的电动汽车数量
        """
        P_chg = 7  # 充电功率 (kW)
        B = 50  # 电池容量 (kWh)
        eta_chg = 0.95  # 充电效率

        if n_discrete is not None and not 0 <= n_discrete <= N:
            raise ValueError("n_discrete must be between 0 and N")
        discrete_flags = None
        if n_discrete is not None:
            discrete_flags = np.zeros(N, dtype=int)
            if n_discrete:
                discrete_flags[self.rng.choice(N, size=n_discrete, replace=False)] = 1

        i = 0
        while i < N:
            is_discrete = (
                int(discrete_flags[i])
                if discrete_flags is not None
                else self.rng.choice([0, 1], p=[1-self.discrete_rate, self.discrete_rate])
            )
            self.discrete_EV += is_discrete
            # 白天充电 (不跨天)
            start_time = min(10, max(7, self.rng.normal(8.5, 1.5)))
            end_time = min(20, start_time + self.rng.uniform(8, 12))
            end_SOC = 1-P_chg/B
            start_SOC = min(1-2*P_chg/B, max(self.rng.normal(0.5, 0.2), 0.2))

            # 检查充电功率是否足够
            if (end_SOC - start_SOC) * B / (end_time - start_time) > P_chg:
                pass  # 原MATLAB代码中这里只是占位

            self.EV_list.append({
                'ta': start_time,
                'td': end_time,
                'SOCa': start_SOC,
                'SOCd': end_SOC,
                'SOCmax': 1,
                'P_chg': P_chg,
                'B': B,
                'eta_chg': eta_chg,
                'is_discrete': is_discrete,
            })
            i += 1


    def case_aggregator(
            self, A=None, b=None, model_type='pretrainnet', save_artifacts=True,
            result_root=None):
        # 构建Pyomo模型
        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=range(self.T))
        model.var_proj = pyo.Var(model.T)  # 聚合功率变量
        p_max = np.zeros(self.T)
        p_min = np.zeros(self.T)

        model.ESSs= pyo.Set(initialize=range(len(self.ESS_list)))
        model.e_ESS= pyo.Var(model.ESSs, model.T, within=pyo.Reals)
        model.p_chg_ESS = pyo.Var(model.ESSs, model.T, within=pyo.NonNegativeReals)
        model.p_dis_ESS = pyo.Var(model.ESSs, model.T, within=pyo.NonNegativeReals)
        model.u_ESS = pyo.Var(model.ESSs, model.T, within=pyo.Binary)


        model.TCLs= pyo.Set(initialize=range(len(self.TCL_list)))
        model.discete_TCLs = pyo.Set(initialize=range(self.discrete_TCL))
        model.temp_TCL= pyo.Var(model.TCLs, model.T, within=pyo.Reals)
        model.p_TCL = pyo.Var(model.TCLs, model.T, within=pyo.NonNegativeReals)
        model.u_TCL = pyo.Var(model.discete_TCLs, model.T, within=pyo.Binary)

        # 定义EV集合
        model.EVs = pyo.Set(initialize=range(len(self.EV_list)))
        model.discete_EVs = pyo.Set(initialize=range(self.discrete_EV))
        # 定义变量
        model.e_EV = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.p_EV = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.u_EV = pyo.Var(model.discete_EVs, model.T, within=pyo.Binary)  # u_EV为0-1变量

        # 初始化基线
        baseline = np.zeros(self.T)

        # ESS能量动态方程
        for i in model.ESSs:
            ess_data = self.ESS_list[i]
            p_max += ess_data['Pchg']
            p_min -= ess_data['Pchg']
            def energy_dynamics_rule(m, t):
                if t > 0:
                    return m.e_ESS[i, t] == m.e_ESS[i, t - 1] + (self.deltaT * (
                                m.p_chg_ESS[i, t] * ess_data['eta_chg'] - m.p_dis_ESS[i, t] / ess_data['eta_dis']))
                return pyo.Constraint.Skip

            model.add_component(f"ess_energy_dynamics_{i}", pyo.Constraint(model.T, rule=energy_dynamics_rule))

            def energy_limit_rule(m, t):
                return pyo.inequality(0, m.e_ESS[i, t], ess_data['B'])

            model.add_component(f"ess_energy_limit_{i}", pyo.Constraint(model.T, rule=energy_limit_rule))

            def charging_power_limit_rule(m, t):
                return m.p_chg_ESS[i, t]<=m.u_ESS[i,t]*ess_data['Pchg']

            model.add_component(f"ess_charging_power_limit_{i}", pyo.Constraint(model.T, rule=charging_power_limit_rule))

            def discharging_power_limit_rule(m, t):
                return m.p_dis_ESS[i, t]<=(1-m.u_ESS[i,t])*ess_data['Pchg']

            model.add_component(f"ess_discharging_power_limit_{i}",
                                pyo.Constraint(model.T, rule=discharging_power_limit_rule))

            def initial_energy_condition_rule(m):
                return m.e_ESS[i, 0] == ess_data['init_SOC'] * ess_data['B']

            model.add_component(f"ess_initial_energy_condition_{i}", pyo.Constraint(rule=initial_energy_condition_rule))

            def final_energy_condition_rule(m):
                return m.e_ESS[i, self.T - 1] >= m.e_ESS[i, 0]

            model.add_component(f"ess_final_energy_condition_{i}", pyo.Constraint(rule=final_energy_condition_rule))

        i_discrete = 0
        for i in model.TCLs:
            tcl_data = self.TCL_list[i]  # 默认使用第一个EV的数据
            p_max+=tcl_data['Qmax'] / tcl_data['COP']
            if tcl_data['is_discrete']:
                def power_limit_rule(m, t):
                    return m.p_TCL[i, t] == m.u_TCL[i_discrete, t] * tcl_data['Qmax']/tcl_data['COP']
                model.add_component(f"tcl_power_limit_discrete{i}", pyo.Constraint(model.T, rule=power_limit_rule))
                i_discrete+=1
            else:
                def power_limit_rule(m, t):
                    return pyo.inequality(0, m.p_TCL[i, t], tcl_data['Qmax']/tcl_data['COP'])
                model.add_component(f"tcl_power_limit{i}", pyo.Constraint(model.T, rule=power_limit_rule))

            alpha = np.exp(-self.deltaT*tcl_data['H']/tcl_data['C'])
            # 动态温度更新公式
            def temperature_dynamics_rule(m, t):
                if t > 0:
                    return m.temp_TCL[i, t] == (alpha * m.temp_TCL[i, t - 1] +(1 - alpha) * (self.theta_amb[t] +tcl_data['COP']/tcl_data['H']*model.p_TCL[i, t]))
                return pyo.Constraint.Skip

            model.add_component(f"temperature_dynamics{i}", pyo.Constraint(model.T, rule=temperature_dynamics_rule))

            # 温度限制约束：确保温度在上下限之间
            def temperature_limit_rule(m, t):
                return pyo.inequality(tcl_data['temp_min'], m.temp_TCL[i, t], tcl_data['temp_max'])

            model.add_component(f"temperature_limit{i}", pyo.Constraint(model.T, rule=temperature_limit_rule))

            # # 温度终止条件：在最后时刻温度不小于设定的目标温度
            # def final_temperature_condition_rule(m):
            #     return m.temp_TCL[i, self.T - 1] >= tcl_data['temp_set']
            #
            # model.add_component(f"final_temperature_condition{i}",
            #                     pyo.Constraint(rule=final_temperature_condition_rule))
            def initial_temperature_condition_rule(m):
                return m.temp_TCL[i, 0] == tcl_data['temp_set']
            model.add_component(f"initial_temperature_condition{i}", pyo.Constraint(rule=initial_temperature_condition_rule))



        i_discrete = 0
        # 1. 确保每个EV的功率和能量满足边界条件
        for i in model.EVs:
            ev_data = self.EV_list[i]  # 默认使用第一个EV的数据
            # 计算离散化的到达和离开时间索引
            ta_idx = int(np.floor(ev_data['ta'] / self.deltaT))
            td_idx = int(np.ceil(ev_data['td'] / self.deltaT))
            p_max[ta_idx:td_idx]+=ev_data['P_chg']
            if ev_data['is_discrete']:
                def power_limit_rule(m, t):
                    if ta_idx <= t < td_idx:
                        return m.p_EV[i, t] == m.u_EV[i_discrete, t] * ev_data['P_chg']  # 功率为0或ev_data['P_chg']
                    else:
                        return m.p_EV[i, t] == 0  # 离开时功率为 0
                model.add_component(f"ev_power_limit_discrete{i}", pyo.Constraint(model.T, rule=power_limit_rule))
                i_discrete+=1
            else:
                # 约束1: 功率上下限约束
                def power_limit_rule(m, t):
                    if ta_idx <= t < td_idx:
                        return (0, m.p_EV[i,t], ev_data['P_chg'])
                    else:
                        return m.p_EV[i,t] == 0
                model.add_component(f"ev_power_limit{i}", pyo.Constraint(model.T, rule=power_limit_rule))

            # 约束2: 初始能量状态
            def initial_energy_rule(m):
                return m.e_EV[i,ta_idx] == ev_data['SOCa'] * ev_data['B']
            model.add_component(f"initial_energy{i}", pyo.Constraint(rule=initial_energy_rule))

            # 约束3: 能量动态更新
            def energy_dynamics_rule(m, t):
                if ta_idx <= t < td_idx:
                    return m.e_EV[i,t + 1] == m.e_EV[i,t] + m.p_EV[i,t] * ev_data['eta_chg'] * self.deltaT
                else:
                    return pyo.Constraint.Skip
            model.add_component(f"energy_dynamics{i}", pyo.Constraint(model.T, rule=energy_dynamics_rule))

            # 约束4: 能量状态上下限
            def energy_limit_rule(m, t):
                if ta_idx <= t <= td_idx:
                    return (ev_data['SOCa'] * ev_data['B'], m.e_EV[i,t], ev_data['SOCmax'] * ev_data['B'])
                else:
                    return pyo.Constraint.Skip
            model.add_component(f"energy_limit{i}", pyo.Constraint(model.T, rule=energy_limit_rule))

            # 约束5: 离开时最小能量要求
            def departure_energy_rule(m):
                return m.e_EV[i, td_idx] >= ev_data['SOCd'] * ev_data['B']
            model.add_component(f"departure_energy{i}", pyo.Constraint(rule=departure_energy_rule))

        def agg_power_rule(m, t):
            return m.var_proj[t]*(p_max[t]-p_min[t])+p_min[t] == sum(m.p_EV[i, t] for i in m.EVs)+sum(m.p_TCL[i, t] for i in m.TCLs)+sum(m.p_chg_ESS[i, t] for i in m.ESSs)-sum(m.p_dis_ESS[i, t] for i in m.ESSs)
        model.agg_power_constraint = pyo.Constraint(model.T, rule=agg_power_rule)
        self.model = model
        # 近似器矩阵（功率+能量约束）
        lower_tri = np.tril(np.ones((self.T, self.T)))
        lower_tri_normalized = lower_tri / np.sqrt(lower_tri.sum(axis=1, keepdims=True))
        if A is None:
            A_hat = np.vstack([
                np.eye(self.T),  # 功率上限
                -np.eye(self.T),  # 功率下限
                lower_tri_normalized,  # 累积能量下限
                -lower_tri_normalized  # 累积能量上限
            ])
            # 误差计算器
            errorcalculator = ErrorCalculator(
                original_model={'model': model},
                A_hat=A_hat,
                solver='gurobi',
            )
        else:
            errorcalculator = ErrorCalculator(
                original_model={'model': model},
                A_hat=A,
                b_hat=b,
                solver='gurobi',
            )

        # 训练回调函数
        case_name = 'aggregation'
        scenario_name = (
            'mixed'
            if self.discrete_EV or self.discrete_TCL or self.ESS_list
            else 'continuous'
        )
        result_root = PROJECT_ROOT / 'results' if result_root is None else Path(result_root)
        result_folder = result_root / case_name
        if scenario_name == 'mixed':
            result_folder = result_folder / 'discrete'
        figure_folder = result_folder / 'figures'
        if save_artifacts:
            figure_folder.mkdir(parents=True, exist_ok=True)
        visualizer = ErrorVisualizer()
        num_sample = 20
        if save_artifacts:
            np.savetxt(figure_folder / f'b_{errorcalculator._iter}.csv', errorcalculator.b_hat, delimiter=',')
            np.savetxt(figure_folder / f'A_{errorcalculator._iter}.csv', errorcalculator.A_hat, delimiter=',')
        n_train = 500
        def training_callback(error_calculator, epoch=None):
            len_his = len(error_calculator.training_history['feas'])
            print(
                f"Iter {error_calculator._iter}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
                f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
            if save_artifacts and error_calculator._iter % 50 == 0:
                np.savetxt(figure_folder / f'b_{error_calculator._iter}.csv', error_calculator.b_hat, delimiter=',')
                np.savetxt(figure_folder / f'A_{error_calculator._iter}.csv', error_calculator.A_hat, delimiter=',')


            # visualizer.compute_errors(error_calculator, num_sample=num_sample)
            # print((np.mean(visualizer.error_history['error_feas'][-1]),np.mean(visualizer.error_history['error_opt'][-1])))
            # if error_calculator._iter % n_train == 0:
            #     visualizer.plot_dual_violin(save_path=f'{figure_folder}/{model_type}_errors_violin.png')
        # 训练参数配置
        trainer_configure = {
            "call_interval": 1,
            "training_callback": training_callback,
            "optimizer": 'SGD',
            # "lr_A": 5e-9,
            # "lr_b": 1e-1,
            "lr": 1e-3,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 200, "gamma": 0.98},
            "n_cal": 3,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1.0
        }
        params = {  # 名字，初值，误差数据集
            'params_dict': {},
            'dataloader': [None],
            'count': 0,
        }
        return {
            'casename': case_name,
            'A_hat': errorcalculator.A_hat,
            'b_hat': errorcalculator.b_hat,
            'errorcalculator': errorcalculator,
            'trainer_configure': trainer_configure,
            'params': params,
            'result_path': (
                result_root / case_name / f'{model_type}_weights.pthdisc'
                if scenario_name == 'mixed'
                else result_folder / f'{model_type}_weights.pth'
            ),
            'metadata': {
                'T': self.T,
                'scenario': scenario_name,
                'num_evs': len(self.EV_list),
                'ev_data_list': self.EV_list
            },
            'n_train': n_train
        }
    def build_cube_approximation(self):
        model = pyo.ConcreteModel()
        model.T = pyo.Set(initialize=range(self.T))
        model.P_max = pyo.Var(model.T, within=pyo.Reals)
        model.P_min = pyo.Var(model.T, within=pyo.Reals)
        pmax = np.zeros(self.T)
        pmin = np.zeros(self.T)

        model.TCLs = pyo.Set(initialize=range(len(self.TCL_list)))
        model.temp_TCL_u = pyo.Var(model.TCLs, model.T, within=pyo.Reals)
        model.p_TCL_u = pyo.Var(model.TCLs, model.T, within=pyo.NonNegativeReals)
        model.temp_TCL_l = pyo.Var(model.TCLs, model.T, within=pyo.Reals)
        model.p_TCL_l = pyo.Var(model.TCLs, model.T, within=pyo.NonNegativeReals)

        # 定义EV集合
        model.EVs = pyo.Set(initialize=range(len(self.EV_list)))
        model.e_EV_u = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.p_EV_u = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.e_EV_l = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量
        model.p_EV_l = pyo.Var(model.EVs, model.T, within=pyo.NonNegativeReals)  # 每个EV的功率变量

        for i in model.TCLs:
            tcl_data = self.TCL_list[i]
            pmax += tcl_data['Qmax'] / tcl_data['COP']
            def power_upper_limit_rule(m, t):
                return pyo.inequality(0, m.p_TCL_u[i, t], tcl_data['Qmax'] / tcl_data['COP'])

            model.add_component(f"tcl_power_upper_limit{i}", pyo.Constraint(model.T, rule=power_upper_limit_rule))
            def power_lower_limit_rule(m, t):
                return pyo.inequality(0, m.p_TCL_l[i, t], tcl_data['Qmax'] / tcl_data['COP'])

            model.add_component(f"tcl_power_lower_limit{i}", pyo.Constraint(model.T, rule=power_lower_limit_rule))

            def tcl_lower_upper_rule(m, t):
                return m.p_TCL_l[i, t]<=m.p_TCL_u[i, t]
            model.add_component(f"tcl_lower_upper{i}", pyo.Constraint(model.T, rule=tcl_lower_upper_rule))

            alpha = np.exp(-self.deltaT * tcl_data['H'] / tcl_data['C'])

            # 动态温度更新公式
            def temperature_u_dynamics_rule(m, t):
                if t > 0:
                    return m.temp_TCL_u[i, t] == (alpha * m.temp_TCL_u[i, t - 1] + (1 - alpha) * (
                            self.theta_amb[t] + tcl_data['COP'] / tcl_data['H'] * model.p_TCL_u[i, t]))
                return pyo.Constraint.Skip

            model.add_component(f"temperature_u_dynamics{i}", pyo.Constraint(model.T, rule=temperature_u_dynamics_rule))
            # 动态温度更新公式
            def temperature_l_dynamics_rule(m, t):
                if t > 0:
                    return m.temp_TCL_l[i, t] == (alpha * m.temp_TCL_l[i, t - 1] + (1 - alpha) * (
                            self.theta_amb[t] + tcl_data['COP'] / tcl_data['H'] * model.p_TCL_l[i, t]))
                return pyo.Constraint.Skip

            model.add_component(f"temperature_l_dynamics{i}", pyo.Constraint(model.T, rule=temperature_l_dynamics_rule))

            # 温度限制约束：确保温度在上下限之间
            def temperature_u_limit_rule(m, t):
                return pyo.inequality(tcl_data['temp_min'], m.temp_TCL_l[i, t], tcl_data['temp_max'])

            model.add_component(f"temperature_u_limit{i}", pyo.Constraint(model.T, rule=temperature_u_limit_rule))

            # 温度限制约束：确保温度在上下限之间
            def temperature_l_limit_rule(m, t):
                return pyo.inequality(tcl_data['temp_min'], m.temp_TCL_l[i, t], tcl_data['temp_max'])

            model.add_component(f"temperature_l_limit{i}", pyo.Constraint(model.T, rule=temperature_l_limit_rule))

            def initial_temperature_u_condition_rule(m):
                return m.temp_TCL_u[i, 0] == tcl_data['temp_set']

            model.add_component(f"initial_temperature_u_condition{i}",
                                pyo.Constraint(rule=initial_temperature_u_condition_rule))
            def initial_temperature_l_condition_rule(m):
                return m.temp_TCL_l[i, 0] == tcl_data['temp_set']

            model.add_component(f"initial_temperature_l_condition{i}",
                                pyo.Constraint(rule=initial_temperature_l_condition_rule))
        for i in model.EVs:
            ev_data = self.EV_list[i]  # 默认使用第一个EV的数据
            # 计算离散化的到达和离开时间索引
            ta_idx = int(np.floor(ev_data['ta'] / self.deltaT))
            td_idx = int(np.ceil(ev_data['td'] / self.deltaT))
            pmax[ta_idx:td_idx] += ev_data['P_chg']

            # 约束1: 功率上下限约束
            def power_u_limit_rule(m, t):
                if ta_idx <= t < td_idx:
                    return (0, m.p_EV_u[i, t], ev_data['P_chg'])
                else:
                    return m.p_EV_u[i, t] == 0
            model.add_component(f"ev_power_u_limit{i}", pyo.Constraint(model.T, rule=power_u_limit_rule))
            # 约束1: 功率上下限约束
            def power_l_limit_rule(m, t):
                if ta_idx <= t < td_idx:
                    return (0, m.p_EV_l[i, t], ev_data['P_chg'])
                else:
                    return m.p_EV_l[i, t] == 0
            model.add_component(f"ev_power_l_limit{i}", pyo.Constraint(model.T, rule=power_l_limit_rule))

            def ev_lower_upper_rule(m, t):
                return m.p_EV_l[i, t]<=m.p_EV_u[i, t]
            model.add_component(f"ev_lower_upper{i}", pyo.Constraint(model.T, rule=ev_lower_upper_rule))

            # 约束2: 初始能量状态
            def initial_energy_u_rule(m):
                return m.e_EV_u[i, ta_idx] == ev_data['SOCa'] * ev_data['B']

            model.add_component(f"initial_energy_u{i}", pyo.Constraint(rule=initial_energy_u_rule))
            # 约束2: 初始能量状态
            def initial_energy_l_rule(m):
                return m.e_EV_l[i, ta_idx] == ev_data['SOCa'] * ev_data['B']

            model.add_component(f"initial_energy_l{i}", pyo.Constraint(rule=initial_energy_l_rule))

            # 约束3: 能量动态更新
            def energy_u_dynamics_rule(m, t):
                if ta_idx <= t < td_idx:
                    return m.e_EV_u[i, t + 1] == m.e_EV_u[i, t] + m.p_EV_u[i, t] * ev_data['eta_chg'] * self.deltaT
                else:
                    return pyo.Constraint.Skip
            model.add_component(f"energy_u_dynamics{i}", pyo.Constraint(model.T, rule=energy_u_dynamics_rule))

            # 约束3: 能量动态更新
            def energy_l_dynamics_rule(m, t):
                if ta_idx <= t < td_idx:
                    return m.e_EV_l[i, t + 1] == m.e_EV_l[i, t] + m.p_EV_l[i, t] * ev_data['eta_chg'] * self.deltaT
                else:
                    return pyo.Constraint.Skip

            model.add_component(f"energy_l_dynamics{i}", pyo.Constraint(model.T, rule=energy_l_dynamics_rule))

            # 约束4: 能量状态上下限
            def energy_u_limit_rule(m, t):
                if ta_idx <= t <= td_idx:
                    return (ev_data['SOCa'] * ev_data['B'], m.e_EV_u[i, t], ev_data['SOCmax'] * ev_data['B'])
                else:
                    return pyo.Constraint.Skip

            model.add_component(f"energy_u_limit{i}", pyo.Constraint(model.T, rule=energy_u_limit_rule))
            # 约束4: 能量状态上下限
            def energy_l_limit_rule(m, t):
                if ta_idx <= t <= td_idx:
                    return (ev_data['SOCa'] * ev_data['B'], m.e_EV_l[i, t], ev_data['SOCmax'] * ev_data['B'])
                else:
                    return pyo.Constraint.Skip

            model.add_component(f"energy_l_limit{i}", pyo.Constraint(model.T, rule=energy_l_limit_rule))

            # 约束5: 离开时最小能量要求
            def departure_energy_u_rule(m):
                return m.e_EV_u[i, td_idx] >= ev_data['SOCd'] * ev_data['B']

            model.add_component(f"departure_energy_u{i}", pyo.Constraint(rule=departure_energy_u_rule))
            # 约束5: 离开时最小能量要求
            def departure_energy_l_rule(m):
                return m.e_EV_l[i, td_idx] >= ev_data['SOCd'] * ev_data['B']

            model.add_component(f"departure_energy_l{i}", pyo.Constraint(rule=departure_energy_l_rule))

        def agg_power_u_rule(m, t):
            return m.P_max[t] * (pmax[t] - pmin[t]) + pmin[t] == sum(m.p_EV_u[i, t] for i in m.EVs) + sum(
                m.p_TCL_u[i, t] for i in m.TCLs)

        model.agg_power_u_constraint = pyo.Constraint(model.T, rule=agg_power_u_rule)
        def agg_power_l_rule(m, t):
            return m.P_min[t] * (pmax[t] - pmin[t]) + pmin[t] == sum(m.p_EV_l[i, t] for i in m.EVs) + sum(
                m.p_TCL_l[i, t] for i in m.TCLs)

        model.agg_power_l_constraint = pyo.Constraint(model.T, rule=agg_power_l_rule)

        model.obj = pyo.Objective(expr=-sum(model.P_max[t]-model.P_min[t] for t in model.T))
        solver = pyo.SolverFactory('gurobi')
        import time
        start_t = time.time()
        solver.solve(model)
        end_t = time.time()
        comp_time = end_t-start_t
        P_max = np.array([pyo.value(model.P_max[i]) for i in model.P_max])
        P_min = np.array([pyo.value(model.P_min[i]) for i in model.P_min])

        A_hat = np.vstack([np.eye(self.T), -np.eye(self.T)])
        b_hat = np.hstack([P_max, -P_min])
        return (A_hat, b_hat), comp_time
    def count_complexity(self):
        # 1. 查看连续变量个数
        continuous_vars = sum(1 for v in self.model.component_data_objects(pyo.Var, active=True)
                              if not v.is_binary() and not v.is_integer())
        print(f"连续变量个数: {continuous_vars}")

        # 2. 查看0-1变量（二进制变量）个数
        binary_vars = sum(1 for v in self.model.component_data_objects(pyo.Var, active=True)
                          if v.is_binary())
        print(f"0-1变量个数: {binary_vars}")

        # 3. 查看约束数量
        constraints = sum(1 for c in self.model.component_data_objects(pyo.Constraint, active=True))
        print(f"约束数量: {constraints}")
if __name__ == "__main__":
    print(
        "Use `python -m Simulator.runners.main_agg --help` for the paper "
        "aggregation workflows."
    )

