import pyomo.environ as pyo
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator, pyomo_params_to_numpy
from Simulator.Plotter import ShapeDrawer_2D, ErrorVisualizer
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
np.random.seed(42)

class safe_region_case:

    def __init__(self):

        self.model = None
        self.solver = 'gurobi'
    def build_case_1(self,x_only = True,model_type='pretrainnet',total_samples = 100,batch_size = 5,device = 'cpu'):

        model = pyo.ConcreteModel()
        K = 10

        model.K = pyo.RangeSet(0, K)
        model.K_controls = pyo.RangeSet(0, K - 1)

        dim_x = 2
        dim_u = 1
        model.x = pyo.Var(model.K, range(dim_x), within=pyo.Reals, initialize=0.0)
        model.u = pyo.Var(model.K_controls, range(dim_u), within=pyo.Reals, initialize=0.0)
        model.theta = pyo.Param(initialize=0, mutable=True)
        self.A = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        self.B = np.array([[1.0],
                           [0.5]])

        # 系统动态约束
        def dynamics_constraint(model, k, i):
            return model.x[k + 1, i] == sum(self.A[i, j] * model.x[k, j] for j in range(dim_x)) + \
                                         sum(self.B[i, j] * model.u[k, j] for j in range(dim_u))
        model.dynamics = pyo.Constraint(model.K_controls, range(dim_x), rule=dynamics_constraint)

        # 状态约束
        def x_constraint_rule(m, k):
            return sum((m.x[k, i])**2 for i in range(dim_x)) <= 25+10*m.theta+m.theta**2
        model.x_constraint = pyo.Constraint(model.K_controls, rule=x_constraint_rule)

        # 控制输入约束
        def u_upper_constraint_rule(model, k, i):
            return model.u[k, i] <= 0.25
        model.u_upper_constraint = pyo.Constraint(model.K_controls, range(dim_u), rule=u_upper_constraint_rule)

        def u_lower_constraint_rule(model, k, i):
            return model.u[k, i] >= -0.25
        model.u_lower_constraint = pyo.Constraint(model.K_controls, range(dim_u), rule=u_lower_constraint_rule)

        # 保存模型和维度
        self.model = model
        self.dim_x = dim_x
        self.dim_u = dim_u
        class CaseData(Dataset):
            def __init__(self, size=total_samples):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    'theta': 2*torch.rand(1, device=device)-1,  # 归一化值 ∈ [-1,1]
                }
        params_dict, param_count = pyomo_params_to_numpy(model)
        params = {  # 名字，初值，误差数据集
            'params_dict': params_dict,
            'dataloader': DataLoader(
                CaseData(),
                batch_size=batch_size,
                shuffle=True
            ),
            'count': param_count,
        }
        if x_only:
            n =16
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)  # 从0到2π的n个等分角
            x = np.cos(angles)
            y = np.sin(angles)
            self.A_hat = np.column_stack((x, y))
            errorcalculator = self.build_x_fr()
        else:
            A_x = np.vstack([np.eye(self.dim_x),
                             -np.eye(self.dim_x),
                             [1, 1],
                             [1, -1],
                             [-1, -1],
                             [-1, 1]
                             ], dtype=float)
            A_u = np.array([[1],
                            [-1]], dtype=float)
            ncons_xu = 30
            np.random.seed(0)
            A_xu = np.random.randn(ncons_xu, self.dim_x + self.dim_u)

            self.A_hat = np.vstack([
                np.hstack([A_x, np.zeros([A_x.shape[0], self.dim_u])]),
                np.hstack([np.zeros([A_u.shape[0], self.dim_x]), A_u]),
                A_xu,
            ], dtype=float)  # 所有的A矩阵写在这里
            errorcalculator = self.build_xu_fr()

        case_name = 'simple_case'
        if x_only:
            figure_folder = f'{PROJECT_ROOT}\\results\\safe_region\\{case_name}\\figures'
            plotter = ShapeDrawer_2D()
            os.makedirs(figure_folder + f'/pretrain_process', exist_ok=True)
            xlim = [-5.5, 5.5]
            ylim = xlim
            plt.figure(figsize=(8, 6))

            bp = self._cal_bound_points(n=100)
            plotter.plot_convex_hull(bp, alpha=0.3, facecolor='blue',
                         edgecolor='blue',
                         label='Original region')


        n_train = 501

        def training_callback(error_calculator, epoch=None):
            len_his = len(error_calculator.training_history['feas'])
            print(
                f"Iter {error_calculator._iter}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
                f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
            # print( f"Iter {error_calculator._iter}: FeasErr={error_calculator.training_history['feas'][-1]:.2e}, "
            #     f"OptErr={error_calculator.training_history['opt'][-1]:.2e}")
            # print((errorcalculator.A_hat[-1],errorcalculator.b_hat[-1]))
            if model_type.lower() == 'pretrainnet' and x_only:
                if len(plotter.shapes)>1:
                    plotter.remove_shape(plotter.shapes[-1]['id'])
                plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,
                                     facecolor='green', xlim=xlim, ylim=ylim,
                                     label=f'Approximation',
                                     title=f'Training step = {epoch}'
                                     )
                plotter.save(figure_folder + f'/pretrain_process/step{epoch}.png')

        # 训练参数配置
        if model_type.lower() == 'pretrainnet':
            trainer_configure = {
                "call_interval": 5,
                "training_callback": training_callback,
                "optimizer": 'SGD',
                "lr": 2e-2,
                "batch_size": 1,
                "scheduler": {"type": "StepLR", "step_size": 200, "gamma": 0.98},
                "n_cal": 1,
                "cal_feas": True,
                "cal_opt": True,
                "rate_opt_feas": 1.
            }
        else:
            trainer_configure = {
                "call_interval": 1,
                "training_callback": training_callback,
                "optimizer": "adam",
                # "optimizer": "sgd",
                "lr": 2e-3,
                "batch_size": batch_size,
                "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.98},
                "n_cal":2,
                "cal_feas": True,
                "cal_opt": True,
                "rate_opt_feas": 1.,
            }

        return {
            'casename': case_name,
            'A_hat': errorcalculator.A_hat,
            'b_hat': errorcalculator.b_hat,
            'errorcalculator': errorcalculator,
            'trainer_configure': trainer_configure,
            'params': params,
            'result_path': f'{PROJECT_ROOT}\\results\\safe_region\\{case_name}\\{model_type}_weights_fr.pth',
            'n_train': n_train
        }

    def _cal_bound_points(self, n = 100):
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)  # 从0到2π的n个等分角
        x = np.cos(angles)
        y = np.sin(angles)
        c_lists = np.column_stack((x, y))
        model = self.model
        model.c = pyo.Param(range(2),mutable=True, initialize=1., within= pyo.Reals)
        model.max_direction = pyo.Objective(expr=sum(model.c[i]*model.var_proj[i] for i in range(2)), sense=pyo.maximize)
        # 存储结果的数组
        boundary_points = np.zeros((n, 2))
        solver = pyo.SolverFactory(self.solver)
        # 对每个方向求解最大化问题
        for i, c_vec in enumerate(c_lists):
            # 更新目标函数系数
            for j in range(2):
                model.c[j] = c_vec[j]

            # 求解模型
            results = solver.solve(model, tee=False)  # tee=False不显示输出

            # 检查求解状态
            if results.solver.termination_condition == pyo.TerminationCondition.optimal:
                # 获取最优解坐标
                boundary_points[i] = [pyo.value(model.var_proj[0]), pyo.value(model.var_proj[1])]
            else:
                # 处理求解失败情况
                print(f"警告：方向 {i}({c_vec}) 求解失败")
                print(f"终止条件: {results.solver.termination_condition}")
                boundary_points[i] = [np.nan, np.nan]  # 标记为NaN
        model.max_direction.deactivate()
        return boundary_points
    def build_case_2(self,x_only = True,model_type='pretrainnet'):

        model = pyo.ConcreteModel()
        K = 10

        model.K = pyo.RangeSet(0, K)
        model.K_controls = pyo.RangeSet(0, K - 1)

        dim_x = 4
        dim_u = 2
        model.x = pyo.Var(model.K, range(dim_x), within=pyo.Reals, initialize=0.0)
        model.u = pyo.Var(model.K_controls, range(dim_u), within=pyo.Reals, initialize=0.0)

        self.A = np.array([[0.7, -0.1, 0.0, 0.0],
                           [0.2, -0.5, 0.1, 0.0],
                           [0., 0.1, 0.1, 0.],
                           [0.5, 0.0, 0.5, 0.5]])
        self.B = np.array([[0, 0.1],
                           [0.1, 1],
                           [0.1, 0],
                           [0,0]])

        # 系统动态约束
        def dynamics_constraint(model, k, i):
            return model.x[k + 1, i] == sum(self.A[i, j] * model.x[k, j] for j in range(dim_x)) + \
                                         sum(self.B[i, j] * model.u[k, j] for j in range(dim_u))
        model.dynamics = pyo.Constraint(model.K_controls, range(dim_x), rule=dynamics_constraint)

        # 状态约束
        def x_upper_constraint_rule(m, k, i):
            return m.x[k, i] <= 5.
        model.x_upper_constraint = pyo.Constraint(model.K_controls, range(dim_x), rule=x_upper_constraint_rule)

        def x_lower_constraint_rule(m, k, i):
            return m.x[k, i] >= -5.
        model.x_lower_constraint = pyo.Constraint(model.K_controls, range(dim_x), rule=x_lower_constraint_rule)

        # 控制输入约束
        def u_upper_constraint_rule(m, k, i):
            return m.u[k, i] <= 5.
        model.u_upper_constraint = pyo.Constraint(model.K_controls, range(dim_u), rule=u_upper_constraint_rule)

        def u_lower_constraint_rule(m, k, i):
            return m.u[k, i] >= -5.
        model.u_lower_constraint = pyo.Constraint(model.K_controls, range(dim_u), rule=u_lower_constraint_rule)

        # 保存模型和维度
        self.model = model
        self.dim_x = dim_x
        self.dim_u = dim_u

        if x_only:
            A_hat = np.vstack([np.eye(self.dim_x),
                             -np.eye(self.dim_x),])
            np.random.seed(0)
            ncons_x = 30
            self.A_hat = np.vstack([A_hat,np.random.randn(ncons_x, self.dim_x)])
            errorcalculator = self.build_x_fr()
        else:
            ncons_xu = 30
            np.random.seed(0)
            A_xu = np.random.randn(ncons_xu, self.dim_x + self.dim_u)

            self.A_hat = np.vstack([
                A_xu,
            ], dtype=float)  # 所有的A矩阵写在这里
            errorcalculator = self.build_xu_fr()

        case_name = 'simple_case_2'
        n_train = 500
        figure_folder = f'{PROJECT_ROOT}\\results\\safe_region\\{case_name}\\figures'
        os.makedirs(figure_folder, exist_ok=True)
        visualizer = ErrorVisualizer()
        num_sample = 50

        def training_callback(error_calculator, epoch=None):
            len_his = len(error_calculator.training_history['feas'])
            print(
                f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
                f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
            visualizer.compute_errors(error_calculator, num_sample=num_sample)
            if epoch % n_train <=5 :
                visualizer.plot_dual_violin(save_path=f'{figure_folder}/{model_type}_errors_violin.png')

        # 训练参数配置
        trainer_configure = {
            "call_interval": 50,
            "training_callback": training_callback,
            "optimizer": 'SGD',
            "lr": 2e-2,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 200, "gamma": 0.98},
            "n_cal": 1,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1.
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
            'result_path': f'{PROJECT_ROOT}\\results\\safe_region\\{case_name}\\{model_type}_weights.pth',
            'n_train': n_train
        }
    # 向量标幺化函数
    @staticmethod
    def _normalize_vector(x_array, xrange):
        xmin, xmax = xrange
        if xmax == xmin:
            return {k: 0.0 for k in range(len(x_array))}
        return {k: 2*(x_array[k] - xmin) / (xmax - xmin)-1.0 for k in range(len(x_array))}

    # 向量反标幺化函数
    @staticmethod
    def _denormalize(x_normalized, xrange):
        xmin, xmax = xrange
        return (x_normalized+1.0)/2 * (xmax - xmin) + xmin
    @staticmethod
    def _calculate_average_curve(data, T):
        num_curves = len(data) - T + 1
        all_curves = np.zeros((num_curves, T))
        for i in range(num_curves):
            all_curves[i] = data[i:i + T]  # 第i条曲线：从索引i开始的16个元素
        average_curve = np.mean(all_curves, axis=0)
        return average_curve
    def build_mg_case(self,T = 16,
                      Delta_t = 15 / 60,
                      params = None,
                      data = None,
                      x_only=True,
                      model_type='pretrainnet',
                      current_time=None,
                      total_samples=100,
                      batch_size=5,
                      device='cpu'
                      ):
        """
        微电网案例构建函数

        状态变量 x 包含:
        - 连续TCL温度: theta_cont[i,t]
        - 离散TCL温度: theta_disc[i,t]
        - ESS能量: e[i,t]

        控制变量 u 包含:
        - 连续TCL功率: p_cont_tcl[i,t]
        - 离散TCL功率: p_disc_tcl[i,t] (通过二进制变量u_disc_tcl控制)
        - ESS充电功率: p_chg_ess[i,t]
        - ESS放电功率: p_dis_ess[i,t]
        - 光伏功率: p_pv[t]
        """
        if not data:
            total_periods = int(240 / Delta_t)
            full_theta_amb = np.random.uniform(-5, 5, total_periods)  # 环境温度
            full_p_bl = np.random.uniform(5, 15, total_periods)  # 基础负荷
            full_p_pv = np.random.uniform(0, 35, total_periods)  # 光伏发电
        else:
            full_theta_amb = data['full_theta_amb']
            full_p_bl = data['full_p_bl']
            full_p_pv = data['full_p_pv']
            total_periods = len(full_theta_amb)
        # 计算范围
        self.theta_amb_range = (np.min(full_theta_amb), np.max(full_theta_amb))
        self.p_bl_range = (np.min(full_p_bl), np.max(full_p_bl))
        self.p_pv_range = (np.min(full_p_pv), np.max(full_p_pv))
        self.theta_amb_init = np.mean(self.theta_amb_range)*np.ones(T,dtype=np.float32)
        self.p_bl_init = np.mean(self.p_bl_range) * np.ones(T, dtype=np.float32)
        self.p_pv_init = np.mean(self.p_pv_range) * np.ones(T, dtype=np.float32)

        if current_time is not None:
            theta_amb = full_theta_amb[current_time:current_time + T]
            p_bl = full_p_bl[current_time:current_time + T]
            p_pv = full_p_pv[current_time:current_time + T]
        else:
            theta_amb = self.theta_amb_init
            p_bl = self.p_bl_init
            p_pv = self.p_pv_init
        # theta_amb_init = self._calculate_average_curve(data['full_theta_amb'], T)
        # p_bl_init = self._calculate_average_curve(data['full_p_bl'], T)
        # p_pv_init = self._calculate_average_curve(data['full_p_pv'], T)

        if not params:
            params = {
                'num_cont_tcl': 1,  # 连续型TCL数量改为1
                'num_disc_tcl': 1,  # 离散型TCL数量保持不变
                'num_ess': 1,  # ESS数量保持不变

                # 电网参数
                'C_L': 0.0,
                'C_U': 20.0,

                # 连续型TCL参数（只保留第一个设备的参数）
                'C_cont': [316.11],  # 只保留第一个C值
                'eta_cont': [4.0],  # 只保留第一个eta值
                'H_cont': [3.892],  # 只保留第一个R值
                'p_cont_max_tcl': [17.5],  # 只保留第一个最大功率值
                'theta_min_cont': [21.0],  # 只保留第一个最低温度
                'theta_max_cont': [25.0],  # 只保留第一个最高温度
                'theta_set_cont': [23.0],

                # 离散型TCL参数（保持不变）
                'C_disc': [140.56],  # 每个离散型TCL的C
                'eta_disc': [3.6],  # 每个离散型TCL的eta
                'H_disc': [1.1537],  # 每个离散型TCL的R
                'p_disc_max_tcl': [10.4],  # 每个离散型TCL的最大功率
                'theta_min_disc': [21.0],  # 每个离散型TCL的最低温度
                'theta_max_disc': [25.0],  # 每个离散型TCL的最高温度
                'theta_set_disc': [23.0],

                # ESS参数（保持不变）
                'eta_chg': [0.97],  # 每个ESS的充电效率
                'eta_dis': [0.98],  # 每个ESS的放电效率
                'pmax_chg_ess': [50.0],  # 每个ESS的最大充电功率
                'pmax_dis_ess': [50.0],  # 每个ESS的最大放电功率
                'e_min': [0.0],  # 每个ESS的最小能量
                'e_max': [150.0],  # 每个ESS的最大能量
            }

        model = pyo.ConcreteModel()

        # 时间索引
        model.T = pyo.RangeSet(0, T - 1)

        # 设备索引
        model.CONT_TCL = pyo.RangeSet(0, params['num_cont_tcl'] - 1) if params['num_cont_tcl'] > 0 else []
        model.DISC_TCL = pyo.RangeSet(0, params['num_disc_tcl'] - 1) if params['num_disc_tcl'] > 0 else []
        model.ESS = pyo.RangeSet(0, params['num_ess'] - 1) if params['num_ess'] > 0 else []

        # 计算状态和控制维度
        dim_x = params['num_cont_tcl'] + params['num_disc_tcl'] + params['num_ess']
        dim_u = params['num_cont_tcl'] + params['num_disc_tcl'] + 2 * params['num_ess'] + 1
        self.dim_x = dim_x
        self.dim_u = dim_u
        # 创建统一的状态变量 x 和控制变量 u
        model.x = pyo.Var(model.T, range(dim_x), within=pyo.Reals, initialize=0.0)
        model.u = pyo.Var(model.T, range(dim_u), within=pyo.Reals, initialize=0.0)

        # 公共参数
        model.C_L = params['C_L']
        model.C_U = params['C_U']



        # 定义标幺化后的meta参数（这些是模型真正的params）
        model.p_bl_meta = pyo.Param(
            model.T,
            # initialize=0,
            initialize=self._normalize_vector(p_bl, self.p_bl_range),
            mutable=True,
            domain=pyo.Reals
        )

        model.p_pv_meta = pyo.Param(
            model.T,
            # initialize=0,
            initialize=self._normalize_vector(p_pv, self.p_pv_range),
            mutable=True,
            domain=pyo.Reals
        )

        model.theta_amb_meta = pyo.Param(
            model.T,
            # initialize=0,
            initialize=self._normalize_vector(theta_amb, self.theta_amb_range),
            mutable=True,
            domain=pyo.Reals
        )

        # 定义真实参数作为Expression（由标幺化数据恢复得到）
        model.p_bl = pyo.Expression(
            model.T,
            rule=lambda model, t: self._denormalize(model.p_bl_meta[t], self.p_bl_range)
        )

        model.p_pv = pyo.Expression(
            model.T,
            rule=lambda model, t: self._denormalize(model.p_pv_meta[t], self.p_pv_range)
        )

        model.theta_amb = pyo.Expression(
            model.T,
            rule=lambda model, t: self._denormalize(model.theta_amb_meta[t], self.theta_amb_range)
        )
        # ==================== 连续型TCL ====================
        if model.CONT_TCL:
            # 参数
            model.alpha_cont = pyo.Param(model.CONT_TCL, initialize=lambda model, i: np.exp(
                -Delta_t / (params['C_cont'][i] / params['H_cont'][i])), mutable=False)
            model.eta_cont = pyo.Param(model.CONT_TCL, initialize=lambda model, i: params['eta_cont'][i], mutable=False)
            model.R_cont = pyo.Param(model.CONT_TCL, initialize=lambda model, i: 1 / params['H_cont'][i], mutable=False)
            model.p_cont_max_tcl = pyo.Param(model.CONT_TCL,
                                             initialize=lambda model, i: params['p_cont_max_tcl'][i], mutable=False)
            model.theta_min_cont = pyo.Param(model.CONT_TCL,
                                             initialize=lambda model, i: params['theta_min_cont'][i], mutable=False)
            model.theta_max_cont = pyo.Param(model.CONT_TCL,
                                             initialize=lambda model, i: params['theta_max_cont'][i], mutable=False)
            model.theta_set_cont = pyo.Param(model.CONT_TCL,
                                             initialize=lambda model, i: params['theta_set_cont'][i], mutable=False)

            # 变量
            model.p_cont_tcl = pyo.Var(model.CONT_TCL, model.T, within=pyo.NonNegativeReals)
            model.theta_cont = pyo.Var(model.CONT_TCL, model.T)

            # 关联约束: 将原始变量与统一的x, u关联
            def cont_tcl_state_link_rule(model, i, t):
                # theta_cont[i,t] 对应 x[t, i]
                # return model.theta_cont[i, t] == model.x[t, i]
                return (model.theta_cont[i, t]-model.theta_min_cont[i])/(model.theta_max_cont[i]-model.theta_min_cont[i]) == model.x[t, i]

            model.cont_tcl_state_link = pyo.Constraint(model.CONT_TCL, model.T, rule=cont_tcl_state_link_rule)

            def cont_tcl_control_link_rule(model, i, t):
                # p_cont_tcl[i,t] 对应 u[t, i]
                return model.p_cont_tcl[i, t] == model.u[t, i]

            model.cont_tcl_control_link = pyo.Constraint(model.CONT_TCL, model.T, rule=cont_tcl_control_link_rule)

            # 约束
            def cont_tcl_power_rule(model, i, t):
                return pyo.inequality(0, model.p_cont_tcl[i, t], model.p_cont_max_tcl[i])

            model.cont_tcl_power_constr = pyo.Constraint(model.CONT_TCL, model.T, rule=cont_tcl_power_rule)

            def theta_cont_dynamics_rule(model, i, t):
                if t > 0:
                    return model.theta_cont[i, t] == (model.alpha_cont[i] * model.theta_cont[i, t - 1] +
                                                      (1 - model.alpha_cont[i]) * (model.theta_amb[t] +
                                                                                   model.eta_cont[i] * model.R_cont[
                                                                                       i] *
                                                                                   model.p_cont_tcl[i, t]))
                return pyo.Constraint.Skip

            model.theta_cont_dynamics = pyo.Constraint(model.CONT_TCL, model.T, rule=theta_cont_dynamics_rule)

            # 温度约束
            def theta_cont_limit_rule(model, i, t):
                return pyo.inequality(model.theta_min_cont[i], model.theta_cont[i, t], model.theta_max_cont[i])

            model.theta_cont_limit_constr = pyo.Constraint(model.CONT_TCL, model.T, rule=theta_cont_limit_rule)

            # 温度终止条件：最终温度不小于设定值
            def tcl_cont_final_condition_rule(model, i):
                return model.theta_cont[i, T - 1] >= model.theta_set_cont[i]

            model.tcl_cont_final_condition = pyo.Constraint(model.DISC_TCL, rule=tcl_cont_final_condition_rule)

        # ==================== 离散型TCL ====================
        if model.DISC_TCL:
            # 参数
            model.alpha_disc = pyo.Param(model.DISC_TCL, initialize=lambda model, i: np.exp(
                -Delta_t / (params['C_disc'][i] / params['H_disc'][i])), mutable=False)
            model.eta_disc = pyo.Param(model.DISC_TCL, initialize=lambda model, i: params['eta_disc'][i], mutable=False)
            model.R_disc = pyo.Param(model.DISC_TCL, initialize=lambda model, i: 1 / params['H_disc'][i], mutable=False)
            model.p_disc_max_tcl = pyo.Param(model.DISC_TCL,
                                             initialize=lambda model, i: params['p_disc_max_tcl'][i], mutable=False)
            model.theta_min_disc = pyo.Param(model.DISC_TCL,
                                             initialize=lambda model, i: params['theta_min_disc'][i], mutable=False)
            model.theta_max_disc = pyo.Param(model.DISC_TCL,
                                             initialize=lambda model, i: params['theta_max_disc'][i], mutable=False)
            model.theta_set_disc = pyo.Param(model.DISC_TCL,
                                             initialize=lambda model, i: params['theta_set_disc'][i], mutable=False)

            # 变量
            model.p_disc_tcl = pyo.Var(model.DISC_TCL, model.T, within=pyo.NonNegativeReals)
            model.u_disc_tcl = pyo.Var(model.DISC_TCL, model.T, within=pyo.Binary)
            model.theta_disc = pyo.Var(model.DISC_TCL, model.T)

            # 关联约束
            offset_disc_state = params['num_cont_tcl']
            offset_disc_control = params['num_cont_tcl']

            def disc_tcl_state_link_rule(model, i, t):
                # theta_disc[i,t] 对应 x[t, offset_disc_state + i]
                # return model.theta_disc[i, t] == model.x[t, offset_disc_state + i]
                return (model.theta_disc[i, t] - model.theta_min_disc[i]) / (model.theta_max_disc[i] - model.theta_min_disc[i]) == \
                model.x[t, offset_disc_state + i]
            model.disc_tcl_state_link = pyo.Constraint(model.DISC_TCL, model.T, rule=disc_tcl_state_link_rule)

            def disc_tcl_control_link_rule(model, i, t):
                # p_disc_tcl[i,t] 对应 u[t, offset_disc_control + i]
                return model.p_disc_tcl[i, t] == model.u[t, offset_disc_control + i]

            model.disc_tcl_control_link = pyo.Constraint(model.DISC_TCL, model.T, rule=disc_tcl_control_link_rule)

            # 约束
            def disc_tcl_power_rule(model, i, t):
                return model.p_disc_tcl[i, t] == model.u_disc_tcl[i, t] * model.p_disc_max_tcl[i]

            model.disc_tcl_power_constr = pyo.Constraint(model.DISC_TCL, model.T, rule=disc_tcl_power_rule)

            def theta_disc_dynamics_rule(model, i, t):
                if t > 0:
                    return model.theta_disc[i, t] == (model.alpha_disc[i] * model.theta_disc[i, t - 1] +
                                                      (1 - model.alpha_disc[i]) * (model.theta_amb[t] +
                                                                                   model.eta_disc[i] * model.R_disc[
                                                                                       i] *
                                                                                   model.p_disc_tcl[i, t]))
                return pyo.Constraint.Skip

            model.theta_disc_dynamics = pyo.Constraint(model.DISC_TCL, model.T, rule=theta_disc_dynamics_rule)

            # 温度约束
            def theta_disc_limit_rule(model, i, t):
                return pyo.inequality(model.theta_min_disc[i], model.theta_disc[i, t], model.theta_max_disc[i])

            model.theta_disc_limit_constr = pyo.Constraint(model.DISC_TCL, model.T, rule=theta_disc_limit_rule)

            # 温度终止条件：最终温度不小于设定值
            def tcl_disc_final_condition_rule(model, i):
                return model.theta_disc[i, T - 1] >= model.theta_set_disc[i]

            model.tcl_disc_final_condition = pyo.Constraint(model.DISC_TCL, rule=tcl_disc_final_condition_rule)

        # ==================== ESS ====================
        if model.ESS:
            # 参数
            model.eta_chg = pyo.Param(model.ESS, initialize=lambda model, i: params['eta_chg'][i], mutable=False)
            model.eta_dis = pyo.Param(model.ESS, initialize=lambda model, i: params['eta_dis'][i], mutable=False)
            model.pmax_chg_ess = pyo.Param(model.ESS, initialize=lambda model, i: params['pmax_chg_ess'][i], mutable=False)
            model.pmax_dis_ess = pyo.Param(model.ESS, initialize=lambda model, i: params['pmax_dis_ess'][i], mutable=False)
            model.e_min = pyo.Param(model.ESS, initialize=lambda model, i: params['e_min'][i], mutable=False)
            model.e_max = pyo.Param(model.ESS, initialize=lambda model, i: params['e_max'][i], mutable=False)

            # 变量
            model.p_chg_ess = pyo.Var(model.ESS, model.T, within=pyo.NonNegativeReals)
            model.p_dis_ess = pyo.Var(model.ESS, model.T, within=pyo.NonNegativeReals)
            model.u_ess = pyo.Var(model.ESS, model.T, within=pyo.Binary)
            model.e = pyo.Var(model.ESS, model.T)

            # 关联约束
            offset_ess_state = params['num_cont_tcl'] + params['num_disc_tcl']
            offset_ess_control = params['num_cont_tcl'] + params['num_disc_tcl']

            def ess_state_link_rule(model, i, t):
                # e[i,t] 对应 x[t, offset_ess_state + i]
                # return model.e[i, t] == model.x[t, offset_ess_state + i]
                return (model.e[i, t] - model.e_min[i]) / (model.e_max[i] - model.e_min[i]) == \
                model.x[t, offset_ess_state + i]
            model.ess_state_link = pyo.Constraint(model.ESS, model.T, rule=ess_state_link_rule)

            def ess_chg_control_link_rule(model, i, t):
                # p_chg_ess[i,t] 对应 u[t, offset_ess_control + 2*i]
                return model.p_chg_ess[i, t] == model.u[t, offset_ess_control + 2 * i]

            model.ess_chg_control_link = pyo.Constraint(model.ESS, model.T, rule=ess_chg_control_link_rule)

            def ess_dis_control_link_rule(model, i, t):
                # p_dis_ess[i,t] 对应 u[t, offset_ess_control + 2*i + 1]
                return model.p_dis_ess[i, t] == model.u[t, offset_ess_control + 2 * i + 1]

            model.ess_dis_control_link = pyo.Constraint(model.ESS, model.T, rule=ess_dis_control_link_rule)

            # 约束
            def ess_chg_rule(model, i, t):
                return model.p_chg_ess[i, t] <= model.u_ess[i, t] * model.pmax_chg_ess[i]

            model.ess_chg_constr = pyo.Constraint(model.ESS, model.T, rule=ess_chg_rule)

            def ess_dis_rule(model, i, t):
                return model.p_dis_ess[i, t] <= (1 - model.u_ess[i, t]) * model.pmax_dis_ess[i]

            model.ess_dis_constr = pyo.Constraint(model.ESS, model.T, rule=ess_dis_rule)

            def e_dynamics_rule(model, i, t):
                if t > 0:
                    return model.e[i, t] == (model.e[i, t - 1] + Delta_t *
                                             (model.p_chg_ess[i, t] * model.eta_chg[i] - model.p_dis_ess[i, t] /
                                              model.eta_dis[i]))
                return pyo.Constraint.Skip

            model.e_dynamics = pyo.Constraint(model.ESS, model.T, rule=e_dynamics_rule)

            # ESS能量范围约束
            def e_limit_rule(model, i, t):
                return pyo.inequality(model.e_min[i], model.e[i, t], model.e_max[i])

            model.e_limit_constr = pyo.Constraint(model.ESS, model.T, rule=e_limit_rule)

            # ESS终止条件：最终能量不小于初始值
            def ess_final_condition_rule(model, i):
                return model.e[i, T - 1] >= 0.9*model.e[i, 0]

            model.ess_final_condition = pyo.Constraint(model.ESS, rule=ess_final_condition_rule)


        model.p_pv_var = pyo.Var(model.T, within=pyo.NonNegativeReals, bounds=(0, None))

        # 光伏功率与控制变量关联
        offset_pv_control = params['num_cont_tcl'] + params['num_disc_tcl'] + 2 * params['num_ess']

        def pv_control_link_rule(model, t):
            # p_pv[t] 对应 u[t, offset_pv_control]
            return model.p_pv_var[t] == model.u[t, offset_pv_control]

        model.pv_control_link = pyo.Constraint(model.T, rule=pv_control_link_rule)

        def pv_power_limit_rule(model, t):
            return model.p_pv_var[t] <= model.p_pv[t]

        model.pv_power_limit = pyo.Constraint(model.T, rule=pv_power_limit_rule)
        # 电网功率变量
        model.p_grid = pyo.Var(model.T, within=pyo.NonNegativeReals)

        # 功率平衡约束
        def power_balance_rule(model, t):
            cont_power = sum(model.p_cont_tcl[i, t] for i in model.CONT_TCL) if model.CONT_TCL else 0
            disc_power = sum(model.p_disc_tcl[i, t] for i in model.DISC_TCL) if model.DISC_TCL else 0
            chg_power = sum(model.p_chg_ess[i, t] for i in model.ESS) if model.ESS else 0
            dis_power = sum(model.p_dis_ess[i, t] for i in model.ESS) if model.ESS else 0

            return (cont_power + disc_power + chg_power + model.p_bl[t] ==
                    dis_power + model.p_pv_var[t] + model.p_grid[t])

        model.power_balance_constr = pyo.Constraint(model.T, rule=power_balance_rule)

        # 电网功率约束
        def grid_power_rule(model, t):
            return pyo.inequality(model.C_L, model.p_grid[t], model.C_U)

        model.grid_power_constr = pyo.Constraint(model.T, rule=grid_power_rule)

        # 保存模型
        self.model = model

        # ==================== 构建误差计算器 ====================
        if x_only:
            # 仅状态约束
            A_hat = np.vstack([np.eye(self.dim_x),
                               -np.eye(self.dim_x)])
            ncons_x = 0
            self.A_hat = np.vstack([A_hat, np.random.randn(ncons_x, self.dim_x)])
            errorcalculator = self.build_x_fr()
        else:
            # 状态和控制约束
            ncons_xu = 30
            A_xu = np.random.randn(ncons_xu, self.dim_x + self.dim_u)
            self.A_hat = A_xu
            errorcalculator = self.build_xu_fr()

        case_name = 'mg_case'
        n_train = 500
        figure_folder = f'{PROJECT_ROOT}\\results\\safe_region\\{case_name}\\figures'
        os.makedirs(figure_folder, exist_ok=True)
        visualizer = ErrorVisualizer()
        num_sample = 20

        def training_callback(error_calculator, epoch=None):
            len_his = len(error_calculator.training_history['feas'])
            print(
                f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
                f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
            visualizer.compute_errors(error_calculator, num_sample=num_sample)
            if epoch % n_train <= 5:
                visualizer.plot_dual_violin(save_path=f'{figure_folder}/{model_type}_errors_violin.png')

        # 训练参数配置

        if model_type.lower() == 'pretrainnet':
            trainer_configure = {
                "call_interval": 50,
                "training_callback": training_callback,
                "optimizer": 'SGD',
                # "lr_A": 2e-5,
                # "lr_b": 2e-4,
                "lr":2e-5,
                "batch_size": 1,
                "scheduler": {"type": "StepLR", "step_size": 200, "gamma": 0.98},
                "n_cal": 2,
                "cal_feas": True,
                "cal_opt": True,
                'feas_tol': 1e-10,
                'opt_tol': 1e-10,
                "rate_opt_feas": 1.
            }
        else:
            trainer_configure = {
                "call_interval": 1,
                "training_callback": training_callback,
                "optimizer": "adam",
                # "optimizer": "sgd",
                "lr": 4e-6,
                "batch_size": batch_size,
                "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.98},
                "n_cal": 2,
                "cal_feas": True,
                "cal_opt": True,
                'feas_tol': 1e-10,
                'opt_tol': 1e-10,
                "rate_opt_feas": 1.0,
            }
        class CaseData(Dataset):
            def __init__(self, size=total_samples):
                self.size = size
                self.theta_amb_range = (np.min(full_theta_amb), np.max(full_theta_amb))
                self.p_bl_range = (np.min(full_p_bl), np.max(full_p_bl))
                self.p_pv_range = (np.min(full_p_pv), np.max(full_p_pv))
            @staticmethod
            def _normalize(data_sample, data_range):
                min_val, max_val = data_range
                with np.errstate(divide='ignore', invalid='ignore'):
                    normalized = (data_sample - min_val) / (max_val - min_val)
                    normalized[max_val == min_val] = 0.0  # 处理除零情况
                return normalized*2-1.0
            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                # 随机选择一个起始时刻（确保不超出范围）
                start_time = np.random.randint(0, total_periods - T)

                # 提取该时刻的决策窗口数据
                theta_amb_sample = full_theta_amb[start_time:start_time + T]
                p_bl_sample = full_p_bl[start_time:start_time + T]
                p_pv_sample = full_p_pv[start_time:start_time + T]


                # 归一化
                theta_amb_meta = self._normalize(theta_amb_sample, self.theta_amb_range)
                p_bl_meta = self._normalize(p_bl_sample, self.p_bl_range)
                p_pv_meta = self._normalize(p_pv_sample, self.p_pv_range)

                return {
                    'p_bl_meta': torch.tensor(p_bl_meta, dtype=torch.float32, device=device),
                    'p_pv_meta': torch.tensor(p_pv_meta, dtype=torch.float32, device=device),
                    'theta_amb_meta': torch.tensor(theta_amb_meta, dtype=torch.float32, device=device),
                }
        params_dict, param_count = pyomo_params_to_numpy(model)
        params = {  # 名字，初值，误差数据集   `````````````````
            'params_dict': params_dict,
            'dataloader': DataLoader(
                CaseData(),
                batch_size=batch_size,
                shuffle=True
            ),
            'count': param_count,
        }

        return {
            'casename': case_name,
            'A_hat': errorcalculator.A_hat,
            'b_hat': errorcalculator.b_hat,
            'errorcalculator': errorcalculator,
            'trainer_configure': trainer_configure,
            'params': params,
            'result_path': f'{PROJECT_ROOT}\\results\\safe_region\\{case_name}\\{model_type}_weights.pth',
            'n_train': n_train,
            'metadata':{
                'Delta_t':Delta_t,
                'total_periods':total_periods,
                'T':T,
                'full_theta_amb': full_theta_amb,
                'full_p_bl': full_p_bl,
                'full_p_pv': full_p_pv,
            }
        }

    def update_mg_params(self, new_params):
        """
        更新微电网模型参数

        参数:
            model: pyo.ConcreteModel 模型实例
            new_params: dict 包含新参数的字典，格式为:
                {
                    'theta_amb': numpy.ndarray,
                    'p_bl': numpy.ndarray,
                    'p_pv': numpy.ndarray
                }
        """

        # 更新每个参数
        for param_name in new_params.keys():
            # 获取新数据
            new_data = new_params[param_name]
            param_range = getattr(self, f"{param_name}_range")

            # 标幺化新数据
            normalized_data = self._normalize_vector(new_data, param_range)

            # 获取模型中的meta参数
            meta_param = getattr(self.model, f"{param_name}_meta")

            # 更新meta参数值
            for t in self.model.T:
                meta_param[t] = normalized_data[t]
    def build_xu_fr(self):
        model = self.model
        model.var_proj = pyo.Var(range(self.dim_x+self.dim_u),domain=pyo.Reals,initialize=0.0)

        def x_link_rule(m,i):
            return m.x[0,i]==m.var_proj[i]
        model.x_link = pyo.Constraint(range(self.dim_x),rule=x_link_rule)

        def u_link_rule(m,i):
            return m.u[0,i]==m.var_proj[i+self.dim_x]
        model.u_link = pyo.Constraint(range(self.dim_u),rule=u_link_rule)


        original_model = {'model': model}

        errorcalculator = ErrorCalculator(
            original_model=original_model,
            A_hat=self.A_hat,
            solver=self.solver
        )
        return errorcalculator
    def build_x_fr(self):
        model = self.model
        model.var_proj = pyo.Var(range(self.dim_x),domain=pyo.Reals,initialize=0.0)

        def x_link_rule(m,i):
            return m.x[0,i]==m.var_proj[i]
        model.x_link = pyo.Constraint(range(self.dim_x),rule=x_link_rule)

        original_model = {'model': model}

        errorcalculator = ErrorCalculator(
            original_model=original_model,
            A_hat=self.A_hat,
            solver=self.solver
        )
        return errorcalculator
    def test(self, A_hat, b_hat):

        return

    #
    # def solve_mpc(self, x0):
    #
    #     # fix 初始状态变量
    #     for i in range(self.dim_x):
    #         self.model.x[0, i].fix(x0[i])
    #
    #     # 求解
    #     solver = pyo.SolverFactory('gurobi')
    #     result = solver.solve(self.model, tee=False)
    #
    #     if result.solver.status != pyo.SolverStatus.ok or result.solver.termination_condition != pyo.TerminationCondition.optimal:
    #         raise RuntimeError("MPC求解失败")
    #     # print(max(list(pyo.value(sum(self.model.x[k, i] ** 2 for i in range(self.dim_x))) for k in self.model.K)))
    #
    #     # 结果提取
    #     u_opt = np.array([[pyo.value(self.model.u[k, i]) for i in range(self.dim_u)]
    #                       for k in self.model.K_controls])
    #
    #
    #
    #     # 解锁 x0，以便下一步重新 fix
    #     for i in range(self.dim_x):
    #         self.model.x[0, i].unfix()
    #
    #     return u_opt[0], self.model.objective()
    #
    # def solve_mpc_apx(self, x0, A_fr, b_fr, A_obj, b_obj):
    #     apx_model = pyo.ConcreteModel()
    #
    #     dim_x = self.dim_x
    #     dim_u = self.dim_u
    #     n_constraints = A_fr.shape[0]
    #     n_obj = A_obj.shape[0]
    #
    #     # 集合定义
    #     apx_model.I_x = pyo.RangeSet(0, dim_x - 1)
    #     apx_model.I_u = pyo.RangeSet(0, dim_u - 1)
    #     apx_model.I_c = pyo.RangeSet(0, n_constraints - 1)
    #     apx_model.I_obj = pyo.RangeSet(0, n_obj - 2)
    #
    #     # 变量定义
    #     apx_model.x = pyo.Var(apx_model.I_x)
    #     apx_model.u = pyo.Var(apx_model.I_u)
    #     apx_model.f = pyo.Var()
    #
    #
    #     # 约束1：线性不等式 A[x; u] <= b
    #     def ineq_constraint_rule(model, i):
    #         expr = sum(A_fr[i, j] * model.x[j] for j in model.I_x) + \
    #                sum(A_fr[i, dim_x + j] * model.u[j] for j in model.I_u)
    #         return expr <= b_fr[i]
    #
    #     apx_model.ineq_constraints = pyo.Constraint(apx_model.I_c, rule=ineq_constraint_rule)
    #
    #     # 约束2：目标函数 A[x; u; f] <= b
    #     def objective_rule(model, i):
    #         expr = sum(A_obj[i, j] * model.x[j] for j in model.I_x) + \
    #                sum(A_obj[i, dim_x + j] * model.u[j] for j in model.I_u) + \
    #                A_obj[i, dim_x + dim_u] * model.f
    #         return expr <= b_obj[i]
    #     apx_model.objective_pwl = pyo.Constraint(apx_model.I_obj, rule=objective_rule)
    #
    #     # fix 初始状态变量
    #     for i in range(self.dim_x):
    #         apx_model.x[i].fix(x0[i])
    #
    #     apx_model.x_next = pyo.Var(apx_model.I_x)
    #     # 系统动态约束
    #     def dynamics_constraint_rule(model, i):
    #         return model.x_next[i] == sum(self.A[i, j] * model.x[j] for j in range(dim_x)) + \
    #                                      sum(self.B[i, j] * model.u[j] for j in range(dim_u))
    #     apx_model.dynamics = pyo.Constraint(range(dim_x), rule=dynamics_constraint_rule)
    #
    #     # 约束3：下一步的可行性约束 A[x_next; u] <= b
    #     def ineq_constraint_next_rule(model, i):
    #         expr = sum(A_fr[i, j] * model.x_next[j] for j in model.I_x) + \
    #                sum(A_fr[i, dim_x + j] * model.u[j] for j in model.I_u)
    #         return expr <= b_fr[i]
    #
    #     apx_model.ineq_constraints_next = pyo.Constraint(apx_model.I_c, rule=ineq_constraint_next_rule)
    #
    #     # 目标函数：min f
    #     apx_model.obj = pyo.Objective(expr=apx_model.f+0*sum(apx_model.x_next[i]**2 for i in apx_model.I_x), sense=pyo.minimize)
    #
    #     # 求解
    #     solver = pyo.SolverFactory('gurobi')
    #     result = solver.solve(apx_model, tee=False)
    #
    #     # 结果提取
    #     if result.solver.status != pyo.SolverStatus.ok or result.solver.termination_condition != pyo.TerminationCondition.optimal:
    #         raise RuntimeError("MPC_apx求解失败")
    #     u_opt = np.array([pyo.value(apx_model.u[i]) for i in apx_model.I_u])
    #     f_opt = pyo.value(apx_model.f)
    #     print((apx_model.x[0](),apx_model.x[1](), u_opt))
    #
    #
    #     return u_opt, f_opt
    #
    # def receding_horizon_control(self, x0, steps=20, is_apx = False, apx_params = None):
    #     """滚动时域控制，返回系统轨迹和控制输入序列"""
    #     # 存储结果
    #     x_history = [np.array(x0).flatten()]
    #     u_history = []
    #
    #     # 滚动优化
    #     x_current = np.array(x0).flatten()
    #     for s in range(steps):
    #
    #         # 求解当前MPC问题
    #         if is_apx:
    #             u_opt, _ = self.solve_mpc_apx(x_current, A_fr = apx_params['A_fr'], b_fr = apx_params['b_fr'], A_obj = apx_params['A_obj'], b_obj = apx_params['b_obj'], )
    #         else:
    #             u_opt, _ = self.solve_mpc(x_current)
    #
    #         # 应用第一个控制输入
    #         u_current = u_opt.flatten()
    #         u_history.append(u_current)
    #
    #         # 系统演化
    #         x_next = self.A @ x_current + self.B @ u_current
    #         x_history.append(x_next.flatten())
    #
    #         # 更新当前状态
    #         x_current = x_next.flatten()
    #         # print(u_current)
    #
    #     return np.array(x_history), np.array(u_history)
    #
    # def plot_results(self, x_history, u_history):
    #     """绘制控制结果"""
    #     plt.figure(figsize=(12, 6))
    #
    #     plt.subplot(3, 1, 1)
    #     plt.plot(x_history[:, 0], 'b-', linewidth=2)
    #     plt.grid(True)
    #     plt.subplot(3, 1, 2)
    #     plt.plot(x_history[:, 1], 'r-', linewidth=2)
    #     plt.grid(True)
    #
    #     plt.subplot(3, 1, 3)
    #     plt.step(range(len(u_history)), u_history[:, 0], 'g-', linewidth=2, where='post')
    #     plt.grid(True)
    #
    #     plt.tight_layout()
    #     plt.show()

# 主程序
if __name__ == "__main__":
    # 创建MPC控制器实例
    sfc = safe_region_case()
    # case = sfc.build_simplecase(x_only=True)
