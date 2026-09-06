import pyomo.environ as pyo
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator, pyomo_params_to_numpy
from Simulator.Plotter import ShapeDrawer_2D
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class MPCcase:
    """基于文章Example 1的MPC控制器实现"""

    def __init__(self):

        self.model = None

    def build_simplecase(self,include_obj = False):
        """只构建一次MPC模型，初始状态在solve时fix"""
        model = pyo.ConcreteModel()
        K = 10

        model.K = pyo.RangeSet(0, K)
        model.K_controls = pyo.RangeSet(0, K - 1)

        dim_x = 2
        dim_u = 1
        model.x = pyo.Var(model.K, range(dim_x), within=pyo.Reals, initialize=0.0)
        model.u = pyo.Var(model.K_controls, range(dim_u), within=pyo.Reals, initialize=0.0)

        self.A = np.array([[1.0, 1.0],
                           [0.0, 1.0]])
        self.B = np.array([[1.0],
                           [0.5]])
        self.x_range = [-5, 5]
        self.u_range = [-0.25, 0.25]
        # 系统动态约束
        def dynamics_constraint(model, k, i):
            return model.x[k + 1, i] == sum(self.A[i, j] * model.x[k, j] for j in range(dim_x)) + \
                                         sum(self.B[i, j] * model.u[k, j] for j in range(dim_u))
        model.dynamics = pyo.Constraint(model.K_controls, range(dim_x), rule=dynamics_constraint)

        # 状态约束
        def x_constraint_rule(m, k):
            return sum((m.x[k, i])**2 for i in range(dim_x)) <= self.x_range[1]**2
        model.x_constraint = pyo.Constraint(model.K_controls, rule=x_constraint_rule)

        # 控制输入约束
        def u_upper_constraint_rule(model, k, i):
            return model.u[k, i] <= self.u_range[1]
        model.u_upper_constraint = pyo.Constraint(model.K_controls, range(dim_u), rule=u_upper_constraint_rule)

        def u_lower_constraint_rule(model, k, i):
            return model.u[k, i] >= self.u_range[0]
        model.u_lower_constraint = pyo.Constraint(model.K_controls, range(dim_u), rule=u_lower_constraint_rule)

        # 保存模型和维度
        self.model = model
        self.dim_x = dim_x
        self.dim_u = dim_u


        if include_obj:
            # 目标函数
            def objective_rule(model):
                cost = 0.0
                for k in model.K:
                    for i in range(self.dim_x):
                        cost += model.x[k, i] ** 2
                        # cost += model.x[k, i]
                for k in model.K_controls:
                    for i in range(self.dim_u):
                        cost += 0.01 * model.u[k, i] ** 2
                        # cost += 0.01 * model.u[k, i]
                return cost
            self.model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)

    def build_simplecase_fr(self,model_type='pretrainnet'):
        self.build_simplecase()
        model = self.model
        model.var_proj = pyo.Var(range(self.dim_x+self.dim_u),domain=pyo.Reals,initialize=0.0)

        def x_link_rule(m,i):
            return m.x[0,i]==m.var_proj[i]*(self.x_range[1]-self.x_range[0])+self.x_range[0]
        model.x_link = pyo.Constraint(range(self.dim_x),rule=x_link_rule)

        def u_link_rule(m,i):
            return m.u[0,i]==m.var_proj[i+self.dim_x]*(self.u_range[1]-self.u_range[0])+self.u_range[0]
        model.u_link = pyo.Constraint(range(self.dim_u),rule=u_link_rule)


        original_model = {'model': model}
        A_x = np.vstack([np.eye(self.dim_x),
                         -np.eye(self.dim_x),
                         [1,1],
                         [1,-1],
                         [-1,-1],
                         [-1,1]
                         ],dtype=float)
        A_u = np.array([[1],
                        [-1]],dtype=float)
        ncons_xu = 30
        np.random.seed(0)
        A_xu = np.random.randn(ncons_xu,self.dim_x+self.dim_u)
        A_hat = np.vstack([
            np.hstack([A_x,np.zeros([A_x.shape[0],self.dim_u])]),
            np.hstack([np.zeros([A_u.shape[0], self.dim_x]),A_u]),
            A_xu,
        ], dtype=float)  # 所有的A矩阵写在这里

        row_norms = np.linalg.norm(A_hat, axis=1, keepdims=True)

        # 避免除以零（如果某行全零，则保持原样）
        row_norms[row_norms == 0] = 1
        # 归一化
        A_hat = A_hat / row_norms

        errorcalculator = ErrorCalculator(
            original_model=original_model,
            A_hat=A_hat,
            solver='gurobi'
        )

        case_name = 'simple_case'
        figure_folder = f'{PROJECT_ROOT}\\results\\MPC\\{case_name}\\figures'
        os.makedirs(figure_folder + f'/pretrain_process', exist_ok=True)

        plt.figure(figsize=(8, 6))
        xlim = [-0.1,1.1]
        ylim = xlim
        plotter = ShapeDrawer_2D()

        plotter.plot_polygon(errorcalculator.A_hat[:A_x.shape[0],:2], errorcalculator.b_hat[:A_x.shape[0]],
                             facecolor='green', xlim=xlim, ylim=ylim,
                             label=f'Approximation',
                             title=f'Training step = {0}'
                             )
        plotter.save(figure_folder + f'/pretrain_process/step0{0}')

        n_train = 501
        def training_callback(error_calculator, epoch=None):
            len_his = len(error_calculator.training_history['feas'])
            print(f"Iter {error_calculator._iter}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
                f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
            # print( f"Iter {error_calculator._iter}: FeasErr={error_calculator.training_history['feas'][-1]:.2e}, "
            #     f"OptErr={error_calculator.training_history['opt'][-1]:.2e}")
            # print((errorcalculator.A_hat[-1],errorcalculator.b_hat[-1]))
            if model_type.lower() == 'pretrainnet':
                plotter.remove_shape(plotter.shapes[-1]['id'])
                plotter.plot_polygon(errorcalculator.A_hat[:A_x.shape[0], :2], errorcalculator.b_hat[:A_x.shape[0]],
                                     facecolor='green', xlim=xlim, ylim=ylim,
                                     label=f'Approximation',
                                     title=f'Training step = {epoch}'
                                     )
                plotter.save(figure_folder + f'/pretrain_process/step{epoch}')
        # 误差计算器

        # 训练参数配置
        trainer_configure = {
            "call_interval": 5,
            "training_callback": training_callback,
            "optimizer": 'SGD',
            # "lr_A": 2e-2,
            # "lr_b":2e-1,
            "lr":5e-1,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 200, "gamma": 0.98},
            "n_cal": 1,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 0.6
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
            'result_path': f'{PROJECT_ROOT}\\results\\MPC\\{case_name}\\{model_type}_weights_fr.pth',
            'n_train': n_train
        }
    def build_simplecase_obj(self,model_type='pretrainnet', A_fr = None, b_fr = None):
        self.build_simplecase()
        model = self.model
        model.var_proj = pyo.Var(range(self.dim_x+self.dim_u+1),domain=pyo.Reals,initialize=0.0)

        def x_link_rule(m,i):
            return m.x[0,i]==m.var_proj[i]*(self.x_range[1]-self.x_range[0])+self.x_range[0]
        model.x_link = pyo.Constraint(range(self.dim_x),rule=x_link_rule)

        def u_link_rule(m,i):
            return m.u[0,i]==m.var_proj[i+self.dim_x]*(self.u_range[1]-self.u_range[0])+self.u_range[0]
        model.u_link = pyo.Constraint(range(self.dim_u),rule=u_link_rule)


        # 约束1：线性不等式 A[x; u] <= b
        def apx_fr_rule(m, i):
            expr = sum(A_fr[i, j] * m.x[0,j] for j in range(self.dim_x)) + \
                   sum(A_fr[i, self.dim_x + j] * m.u[0,j] for j in range(self.dim_u))
            return expr <= b_fr[i]
        model.apx_fr = pyo.Constraint(range(b_fr.shape[0]), rule=apx_fr_rule)


        model.f = model.var_proj[self.dim_x+self.dim_u]

        model.obj_real = pyo.Expression(
            expr=sum(model.x[k, i]**2 for k in model.K for i in range(self.dim_x)) +
                 sum(0.01 * model.u[k, i]**2 for k in model.K_controls for i in range(self.dim_u))
        )

        solver = pyo.SolverFactory('gurobi')

        model.init_obj_max = pyo.Objective(expr=model.obj_real,sense=pyo.maximize)
        solver.solve(model)
        fmax = pyo.value(model.init_obj_max)
        model.init_obj_max.deactivate()
        model.init_obj_min = pyo.Objective(expr=model.obj_real, sense=pyo.minimize)
        solver.solve(model)
        fmin = pyo.value(model.init_obj_min)
        model.init_obj_min.deactivate()
        self.f_range = [fmin,fmax]
        model.obj = pyo.Expression(expr=(model.obj_real-fmin)/(fmax-fmin))
        model.epigraph = pyo.Constraint(expr=(model.f>=model.obj))
        # model.f_max_cons = pyo.Constraint(expr=(model.f <=277.))


        original_model = {'model': model,
                          'fmax_solver':'gurobi',
                          'FR_params':{'A_fr':A_fr, 'b_fr':b_fr},
                          }

        ncons_f = 50

        # A_f = np.random.randn(ncons_f,self.dim_x+self.dim_u+1)
        # A_f[:,-1] = -np.abs(A_f[:,-1])
        # A_hat = A_f  # 所有的A矩阵写在这里

        A_f = np.random.randn(ncons_f,self.dim_x+self.dim_u+1)
        row_norms = np.linalg.norm(A_f, axis=1, keepdims=True)

        # 避免除以零（如果某行全零，则保持原样）
        row_norms[row_norms == 0] = 1

        # 归一化
        A_f = A_f / row_norms
        A_f[:,-1] = -np.abs(A_f[:,-1])

        A_hat = np.vstack([A_f,
                           np.hstack([np.zeros([2,self.dim_x+self.dim_u]),[[-1.],[1.]]]),
        ], dtype=float) # 这一行不能变，这表示目标函数的上界
        errorcalculator = ErrorCalculator(
            original_model=original_model,
            A_hat=A_hat,
            is_epigraph=True,
            solver='ipopt'
        )


        case_name = 'simple_case'
        figure_folder = f'{PROJECT_ROOT}\\results\\MPC\\{case_name}\\figures'
        os.makedirs(figure_folder, exist_ok=True)

        n_train = 501
        def training_callback(error_calculator, epoch=None):
            len_his = len(error_calculator.training_history['feas'])
            print(f"Iter {error_calculator._iter}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
                f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")

        # 误差计算器

        # 训练参数配置
        trainer_configure = {
            "call_interval": 5,
            "training_callback": training_callback,
            "optimizer": 'SGD',
            "lr": 1e-1,
            # "lr_b":5e-2,
            # "lr_A":3e-3,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 200, "gamma": 0.98},
            "n_cal": 3,
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
            'A_hat': A_hat,
            'b_hat': errorcalculator.b_hat,
            'errorcalculator': errorcalculator,
            'trainer_configure': trainer_configure,
            'params': params,
            'result_path': f'{PROJECT_ROOT}\\results\\MPC\\{case_name}\\{model_type}_weights_obj.pth',
            'n_train': n_train
        }

    def solve_mpc(self, x0):

        # fix 初始状态变量
        for i in range(self.dim_x):
            self.model.x[0, i].fix(x0[i])

        # 求解
        solver = pyo.SolverFactory('gurobi')
        result = solver.solve(self.model, tee=False)

        if result.solver.status != pyo.SolverStatus.ok or result.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError("MPC求解失败")
        # print(max(list(pyo.value(sum(self.model.x[k, i] ** 2 for i in range(self.dim_x))) for k in self.model.K)))

        # 结果提取
        u_opt = np.array([[pyo.value(self.model.u[k, i]) for i in range(self.dim_u)]
                          for k in self.model.K_controls])

        # 解锁 x0，以便下一步重新 fix
        for i in range(self.dim_x):
            self.model.x[0, i].unfix()
        print((self.model.x[0,0](), self.model.x[0,1](), u_opt[0], self.model.objective()))
        return u_opt[0], self.model.objective()

    def solve_mpc_apx(self, x0, A_fr, b_fr, A_obj, b_obj):
        apx_model = pyo.ConcreteModel()

        dim_x = self.dim_x
        dim_u = self.dim_u
        n_constraints = A_fr.shape[0]
        n_obj = A_obj.shape[0]

        # 集合定义
        apx_model.I_x = pyo.RangeSet(0, dim_x - 1)
        apx_model.I_u = pyo.RangeSet(0, dim_u - 1)
        apx_model.I_c = pyo.RangeSet(0, n_constraints - 1)
        apx_model.I_obj = pyo.RangeSet(0, n_obj - 2)

        # 变量定义
        apx_model.x = pyo.Var(apx_model.I_x)
        apx_model.u = pyo.Var(apx_model.I_u)
        apx_model.f = pyo.Var()


        # 约束1：线性不等式 A[x; u] <= b
        def ineq_constraint_rule(model, i):
            expr = sum(A_fr[i, j] * (model.x[j]-self.x_range[0])/(self.x_range[1]-self.x_range[0]) for j in model.I_x) + \
                   sum(A_fr[i, dim_x + j] * (model.u[j]-self.u_range[0])/(self.u_range[1]-self.u_range[0]) for j in model.I_u)
            return expr <= b_fr[i]

        apx_model.ineq_constraints = pyo.Constraint(apx_model.I_c, rule=ineq_constraint_rule)

        # 约束2：目标函数 A[x; u; f] <= b
        def objective_rule(model, i):
            expr = sum(A_obj[i, j] * (model.x[j]-self.x_range[0])/(self.x_range[1]-self.x_range[0]) for j in model.I_x) + \
                   sum(A_obj[i, dim_x + j] * (model.u[j]-self.u_range[0])/(self.u_range[1]-self.u_range[0]) for j in model.I_u) + \
                   A_obj[i, dim_x + dim_u] * (model.f-self.f_range[0])/(self.f_range[1]-self.f_range[0])
            return expr <= b_obj[i]
        apx_model.objective_pwl = pyo.Constraint(apx_model.I_obj, rule=objective_rule)

        # fix 初始状态变量
        for i in range(self.dim_x):
            apx_model.x[i].fix(x0[i])

        # apx_model.x_next = pyo.Var(apx_model.I_x)
        # # 系统动态约束
        # def dynamics_constraint_rule(model, i):
        #     return model.x_next[i] == sum(self.A[i, j] * model.x[j] for j in range(dim_x)) + \
        #                                  sum(self.B[i, j] * model.u[j] for j in range(dim_u))
        # apx_model.dynamics = pyo.Constraint(range(dim_x), rule=dynamics_constraint_rule)
        #
        # # 约束3：下一步的可行性约束 A[x_next; u] <= b
        # def ineq_constraint_next_rule(model, i):
        #     expr = sum(A_fr[i, j] * (model.x_next[j]-self.x_range[0])/(self.x_range[1]-self.x_range[0]) for j in model.I_x) + \
        #            sum(A_fr[i, dim_x + j] * (model.u[j]-self.u_range[0])/(self.u_range[1]-self.u_range[0]) for j in model.I_u)
        #     return expr <= b_fr[i]
        #
        # apx_model.ineq_constraints_next = pyo.Constraint(apx_model.I_c, rule=ineq_constraint_next_rule)

        # 目标函数：min f
        apx_model.obj = pyo.Objective(expr=apx_model.f, sense=pyo.minimize)

        # 求解
        solver = pyo.SolverFactory('gurobi')
        result = solver.solve(apx_model, tee=False)

        # 结果提取
        if result.solver.status != pyo.SolverStatus.ok or result.solver.termination_condition != pyo.TerminationCondition.optimal:
            raise RuntimeError("MPC_apx求解失败")
        u_opt = np.array([pyo.value(apx_model.u[i]) for i in apx_model.I_u])
        f_opt = pyo.value(apx_model.f)
        print((apx_model.x[0](),apx_model.x[1](), u_opt[0], f_opt))

        return u_opt, f_opt

    def receding_horizon_control(self, x0, steps=20, is_apx = False, apx_params = None):
        """滚动时域控制，返回系统轨迹和控制输入序列"""
        # 存储结果
        x_history = [np.array(x0).flatten()]
        u_history = []

        # 滚动优化
        x_current = np.array(x0).flatten()
        for s in range(steps):

            # 求解当前MPC问题
            if is_apx:
                u_opt, _ = self.solve_mpc_apx(x_current, A_fr = apx_params['A_fr'], b_fr = apx_params['b_fr'], A_obj = apx_params['A_obj'], b_obj = apx_params['b_obj'], )
            else:
                u_opt, _ = self.solve_mpc(x_current)

            # 应用第一个控制输入
            u_current = u_opt.flatten()
            u_history.append(u_current)

            # 系统演化
            x_next = self.A @ x_current + self.B @ u_current
            x_history.append(x_next.flatten())

            # 更新当前状态
            x_current = x_next.flatten()
            # print(u_current)

        return np.array(x_history), np.array(u_history)

    def plot_results(self, x_history, u_history):
        """绘制控制结果"""
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(x_history[:, 0], 'b-', linewidth=2)
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(x_history[:, 1], 'r-', linewidth=2)
        plt.grid(True)

        plt.subplot(3, 1, 3)
        plt.step(range(len(u_history)), u_history[:, 0], 'g-', linewidth=2, where='post')
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# 主程序
if __name__ == "__main__":
    # 创建MPC控制器实例
    mpc = MPCcase()
    # mpc.build_simplecase(include_obj=True)

    mpc.build_simplecase_fr()
    # # 初始状态 (文章Example 1中使用的初始状态)
    # x0 = [1.0,-1.4]
    #
    # # 滚动时域控制
    # x_history, u_history = mpc.receding_horizon_control(x0, steps=40)
    #
    # # 绘制结果
    # mpc.plot_results(x_history, u_history)
