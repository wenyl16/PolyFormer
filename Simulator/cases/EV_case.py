import numpy as np
from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator
from pyomo.environ import *
from Simulator.Plotter import ErrorVisualizer
import os

class EVGenerator:
    def __init__(self, seed=None,T = 24):
        """
        初始化设备参数生成器

        参数:
            seed (int, optional): 随机种子，用于复现结果
        """
        self.rng = np.random.default_rng(seed)
        self.EV_list = []
        self.T = 24
        self.deltaT = 24/self.T

    def gen_EV(self, N):
        """
        生成N个电动汽车(EV)的充电需求参数
        完全按照MATLAB版本逻辑实现

        参数:
            N (int): 要生成的电动汽车数量
        """
        P_chg = 7  # 充电功率 (kW)
        B = 50  # 电池容量 (kWh)
        eta_chg = 0.95  # 充电效率

        i = 0
        k = 0
        while i < N:
            p1 = 0.6
            p = self.rng.uniform(0, 1)

            if p <= p1:
                # 情况1: 晚上充电 (跨天)
                start_time = min(self.rng.normal(18, 2), 22)
                end_time = max(24 + 7, self.rng.normal(24 + 8, 1.5))
                end_SOC = 0.9 + 0.09 * self.rng.binomial(1, 0.5)
                start_SOC = min(0.9, max(self.rng.normal(0.5, 0.2), 0.3))

                # 检查充电功率是否足够
                if (end_SOC - start_SOC) * B / (end_time - start_time) > P_chg:
                    pass  # 原MATLAB代码中这里只是占位

                # 第一段充电 (当天晚上)
                self.EV_list.append({
                    'ta': start_time,
                    'td': 24-self.deltaT,
                    'SOCa': start_SOC,
                    'SOCd': (end_SOC - start_SOC) / (end_time - start_time) * (24-self.deltaT - start_time) + start_SOC,
                    'SOCmax': (end_SOC - start_SOC) / (end_time - start_time) * (24-self.deltaT - start_time) + start_SOC,
                    'P_chg': P_chg,
                    'B': B,
                    'eta_chg': eta_chg
                })
                k += 1

                # 第二段充电 (次日凌晨)
                self.EV_list.append({
                    'ta': 0,
                    'td': end_time - 24,
                    'SOCa': (end_SOC - start_SOC) / (end_time - start_time) * (24-self.deltaT - start_time) + start_SOC,
                    'SOCd': end_SOC,
                    'SOCmax': 1,
                    'P_chg': P_chg,
                    'B': B,
                    'eta_chg': eta_chg
                })
                k += 1

                i += 1
            else:
                # 情况2: 白天充电 (不跨天)
                start_time = min(10, max(7, self.rng.normal(8.5, 1.5)))
                end_time = min(20, start_time + self.rng.uniform(8, 12))
                end_SOC = 0.9 + 0.09 * self.rng.binomial(1, 0.5)
                start_SOC = min(0.9, max(self.rng.normal(0.6, 0.2), 0.2))

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
                    'eta_chg': eta_chg
                })
                k += 1
                i += 1
    def case_EV(self, model_type='pretrainnet'):
        """
        EV聚合调度案例实现

        参数:
            model_type (str): 模型类型，默认为'pretrainnet'

        返回:
            dict: 包含案例配置的字典
        """
        if not self.EV_list:
            raise ValueError("No EV data available. Please run gen_EV() first.")

        # 构建Pyomo模型
        model = ConcreteModel()
        model.T = Set(initialize=range(self.T))

        # 定义EV集合
        model.EVs = Set(initialize=range(len(self.EV_list)))

        # 定义变量
        model.var_proj = Var(model.T)  # 聚合功率变量
        model.e = Var(model.EVs, model.T, within=NonNegativeReals)  # 每个EV的功率变量
        model.p = Var(model.EVs, model.T, within=NonNegativeReals)  # 每个EV的功率变量

        # 初始化基线
        baseline = np.zeros(self.T)

        # 1. 确保每个EV的功率和能量满足边界条件
        for i in model.EVs:
            ev_data = self.EV_list[i]  # 默认使用第一个EV的数据

            # 计算离散化的到达和离开时间索引
            ta_idx = int(np.floor(ev_data['ta'] / self.deltaT))
            td_idx = int(np.ceil(ev_data['td'] / self.deltaT))

            # 约束1: 功率上下限约束
            def power_limit_rule(m, t):
                if ta_idx <= t <= td_idx:
                    return (0, m.p[i,t], ev_data['P_chg'])
                else:
                    return m.p[i,t] == 0
            model.add_component(f"power_limit{i}", Constraint(model.T, rule=power_limit_rule))

            # 约束2: 初始能量状态
            def initial_energy_rule(m):
                return m.e[i,ta_idx] == ev_data['SOCa'] * ev_data['B']
            model.add_component(f"initial_energy{i}", Constraint(rule=initial_energy_rule))

            # 约束3: 能量动态更新
            def energy_dynamics_rule(m, t):
                if ta_idx <= t < td_idx:
                    return m.e[i,t + 1] == m.e[i,t] + m.p[i,t] * ev_data['eta_chg'] * self.deltaT
                else:
                    return Constraint.Skip
            model.add_component(f"energy_dynamics{i}", Constraint(model.T, rule=energy_dynamics_rule))

            # 约束4: 能量状态上下限
            def energy_limit_rule(m, t):
                if ta_idx <= t <= td_idx:
                    return (ev_data['SOCa'] * ev_data['B'], m.e[i,t], ev_data['SOCmax'] * ev_data['B'])
                else:
                    return Constraint.Skip
            model.add_component(f"energy_limit{i}", Constraint(model.T, rule=energy_limit_rule))

            # 约束5: 离开时最小能量要求
            def departure_energy_rule(m):
                return m.e[i, td_idx] >= ev_data['SOCd'] * ev_data['B']
            model.add_component(f"departure_energy{i}", Constraint(rule=departure_energy_rule))

        def agg_power_rule(m, t):
            return m.var_proj[t] == sum(m.p[i, t] for i in m.EVs)
        model.agg_power_constraint = Constraint(model.T, rule=agg_power_rule)

        # 近似器矩阵（功率+能量约束）
        A_hat = np.vstack([
            np.eye(self.T),  # 功率上限
            -np.eye(self.T),  # 功率下限
            np.tril(np.ones((self.T, self.T))),  # 累积能量下限
            -np.tril(np.ones((self.T, self.T)))  # 累积能量上限
        ])

        # 误差计算器
        errorcalculator = ErrorCalculator(
            original_model={'model': model},
            A_hat=A_hat,
            solver='gurobi',
        )

        # 训练回调函数
        case_name = 'ev_agg'
        figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'
        os.makedirs(figure_folder, exist_ok=True)
        visualizer = ErrorVisualizer()
        num_sample = 20

        n_train = 1001
        def training_callback(error_calculator, epoch=None):
            len_his = len(error_calculator.training_history['feas'])
            print(
                f"Iter {error_calculator._iter}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
                f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
            # np.savetxt(f'{figure_folder}/b_{error_calculator._iter}.csv', error_calculator.b_hat, delimiter=',')
            # np.savetxt(f'{figure_folder}/A_{error_calculator._iter}.csv', error_calculator.A_hat, delimiter=',')

            # visualizer.compute_errors(error_calculator, num_sample=num_sample)
            # print((np.mean(visualizer.error_history['error_feas'][-1]),np.mean(visualizer.error_history['error_opt'][-1])))
            # if error_calculator._iter % n_train == 0:
            #     visualizer.plot_dual_violin(save_path=f'{figure_folder}/{model_type}_errors_violin.png')
        # 训练参数配置
        trainer_configure = {
            "call_interval": 50,
            "training_callback": training_callback,
            "optimizer": 'SGD',
            "lr_A": 5e-9,
            "lr_b": 1e-1,
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
            'A_hat': A_hat,
            'b_hat': errorcalculator.b_hat,
            'errorcalculator': errorcalculator,
            'trainer_configure': trainer_configure,
            'params': params,
            'result_path': f'{PROJECT_ROOT}\\results\\{case_name}\\{model_type}_weights.pth',
            'metadata': {
                'T': self.T,
                'num_evs': len(self.EV_list),
                'ev_data_list': self.EV_list
            },
            'n_train': n_train
        }
# 使用示例
if __name__ == "__main__":
    # 可设置随机种子以便复现结果
    ev_gen = EVGenerator(seed=42)

    # 生成5个电动汽车
    ev_gen.gen_EV(5)
    aa = ev_gen.case_EV()
    print(1)
