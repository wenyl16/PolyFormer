import numpy as np
from pathlib import Path
from Simulator.Approximator import ErrorCalculator, pyomo_params_to_numpy
import torch
from torch.utils.data import Dataset, DataLoader
from pyomo.environ import *
from Simulator.Plotter import ShapeDrawer_2D
import matplotlib.pyplot as plt
from Simulator import PROJECT_ROOT
import os
import pickle
from Simulator.Plotter import ErrorVisualizer

def case_polygon(
        total_samples=200, noise_scale=0.1, batch_size=1,
        model_type='pretrainnet', device='cpu', save_artifacts=True,
        result_root=None):
    """固定参数的原始案例实现"""
    # 固定初始化参数
    A_init = np.array([
        [1, 0], [0, 1], [-1, 0], [0, -1],
        [1, 1], [-1, -1], [1, -1], [-1, 1]
    ])
    b_init = np.array([1, 1, 0, 0, 1.5, -0.5, 0.7, 0.7])

    # 自动推导维度
    dim = A_init.shape[1]
    ncons = A_init.shape[0]
    dim_theta = 2
    num_b = ncons - dim_theta

    # 构建模型
    model = ConcreteModel()

    # 参数定义
    model.A = Param(range(ncons), range(dim),
                    initialize={(i, j): A_init[i, j] for i, j in np.ndindex(A_init.shape)},
                    mutable=True)
    model.b = Param(range(num_b),
                    initialize={i: b_init[i] for i in range(num_b)},
                    mutable=True)
    model.theta = Param(range(dim_theta),
                        initialize={i: b_init[num_b + i] for i in range(2)},
                        mutable=True)

    # 变量定义
    model.var_proj = Var(range(dim), domain=Reals)

    # 约束定义
    def constraint_rule(model, i, is_adjustable):
        idx = i + num_b if is_adjustable else i
        param = model.theta[i] if is_adjustable else model.b[i]
        return sum(model.A[idx, j] * model.var_proj[j] for j in range(dim)) <= param

    model.con_fixed = Constraint(range(num_b), rule=lambda m, i: constraint_rule(m, i, False))
    model.con_adj = Constraint(range(dim_theta), rule=lambda m, i: constraint_rule(m, i, True))

    original_model = {
        'model': model,
        # 'baseline': baseline,
    }

    # 数据集配置
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # theta维度固定为2

    A_hat = np.vstack([
        np.eye(dim),  # 上界
        -np.eye(dim),  # 下界
    ])
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='cplex',
    )
    A_list = [errorcalculator.A_hat]
    b_list = [errorcalculator.b_hat]

    visualizer = ErrorVisualizer()
    num_sample = 50
    visualizer.compute_errors(errorcalculator, num_sample=num_sample)

    case_name = 'polygon'
    result_root = PROJECT_ROOT / 'results' if result_root is None else Path(result_root)
    result_folder = result_root / case_name
    figure_folder = result_folder / 'figures'
    if save_artifacts:
        figure_folder.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plotter = ShapeDrawer_2D()
    def callback(errorcalculator, epoch):
        len_his = len(errorcalculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(errorcalculator.training_history['feas'][-min(10,len_his):]):.2e}, "
              f"OptErr={np.mean(errorcalculator.training_history['opt'][-min(10,len_his):]):.2e}")
        if model_type.lower() == 'pretrainnet':
            xlim = [-0.5, 1.5]
            ylim = [-0.5, 1.5]
            if not epoch:
                plotter.plot_polygon(A_init, b_init,
                                     facecolor='blue', xlim=xlim, ylim=ylim,
                                     label=f'Original Region',
                                     title=f'Training step = {0}',
                                     )
                plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,
                                     facecolor='green', xlim=xlim, ylim=ylim,
                                     label=f'Approximation',
                                     title=f'Training step = {epoch}'
                                     )
            else:
                plotter.remove_shape(plotter.shapes[-1]['id'])
                plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,
                                     facecolor='green', xlim=xlim, ylim=ylim,
                                     label=f'Approximation',
                                     title=f'Training step = {epoch}'
                                     )
            # 保存图片（自动创建目录）
            plotter.save(str(figure_folder / 'pretrain_process' / f'step{epoch}.png'))
            if (epoch + 20) % 100 == 0:
                A_list.append(errorcalculator.A_hat)
                b_list.append(errorcalculator.b_hat)
                visualizer.compute_errors(errorcalculator, num_sample=num_sample)
            if epoch>=980:
                with open(result_folder / 'A_list.pkl', "wb") as f:
                    pickle.dump(A_list, f)
                with open(result_folder / 'b_list.pkl', "wb") as f:
                    pickle.dump(b_list, f)
                with open(result_folder / 'error_history.pkl', "wb") as f:
                    pickle.dump(visualizer.error_history, f)

    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
                'call_interval':20,
                'training_callback':callback,
                "optimizer": "SGD",
                "lr": 0.5,
                "batch_size": 1,
                "scheduler": {
                    "type": "StepLR",
                    "step_size": 100,
                    "gamma": 0.98
                },
                "n_cal": 2,
                "cal_feas": True,
                "cal_opt": True,
                "rate_opt_feas": 1
            }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",
            "lr": 0.01,
            "batch_size": batch_size,
            "scheduler": {
                "type": "StepLR",
                "step_size": 100,
                "gamma": 0.99
            },
            "n_cal": 3,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }
    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,
        'dataloader': DataLoader(
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':dim_theta,#这里是因为A，b都是参数，就用dim_theta,
    }
    return {
        'casename':'polygon',
        'A_hat': errorcalculator.A_hat,
        'b_hat': errorcalculator.b_hat,
        'errorcalculator': errorcalculator,
        'params':params,
        'trainer_configure': trainer_configure,
        'result_path': result_folder / f'{model_type.lower()}_weights.pth',
        'metadata':{'A_init':A_init,'b_init':b_init}
    }

def case_ellipse(
        total_samples=100, noise_scale=0.3, batch_size=5,
        model_type='pretrainnet', device='cpu', save_artifacts=True,
        result_root=None):
    """椭圆约束案例实现"""
    # 固定初始化参数 (二次型矩阵)
    Sigma_init = np.array([
        [5 / 2, -3 / 2],
        [-3 / 2, 5 / 2]
    ])
    dim = 2  # 固定为二维问题
    a_init = 5 / 2
    b_init = -3 / 2

    # 构建Pyomo模型
    model = ConcreteModel()

    # 可调参数定义 (Sigma矩阵的上三角元素)
    model.a = Param(initialize=a_init, mutable=True)
    model.b = Param(initialize=b_init, mutable=True)

    # 决策变量
    model.var_proj = Var(range(dim), domain=Reals)

    # 椭圆约束 (二次型)
    model.constraints = Constraint(
        expr=(model.a * model.var_proj[0] ** 2
        + 2 * model.b * model.var_proj[0] * model.var_proj[1]
        + model.a * model.var_proj[1] ** 2) <= 1
    )

    original_model = {'model': model}

    # 数据集配置
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {"a":torch.normal(0, self.noise_scale, (1,),device=device),
                    "b":torch.normal(0, self.noise_scale, (1,),device=device),}  # theta维度固定为2
    # 近似器参数
    A_hat = np.vstack([
        np.eye(dim),  # 上界
        -np.eye(dim),  # 下界
        np.array([[1, 1], [-1, -1]])  # 对角线约束
    ])
    # 误差计算器
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='cplex'
    )
    A_list = [errorcalculator.A_hat]
    b_list = [errorcalculator.b_hat]

    visualizer = ErrorVisualizer()
    num_sample = 50
    visualizer.compute_errors(errorcalculator, num_sample=num_sample)

    case_name = 'ellipse'
    result_root = PROJECT_ROOT / 'results' if result_root is None else Path(result_root)
    result_folder = result_root / case_name
    figure_folder = result_folder / 'figures'
    if save_artifacts:
        figure_folder.mkdir(parents=True, exist_ok=True)

    # 可视化回调函数
    plt.figure(figsize=(8, 6))
    plotter = ShapeDrawer_2D()
    xlim = [-2, 2]
    ylim = [-2, 2]

    # marking_epoches = [1,6,12,25,50,100,195]
    marking_epoches = list(range(1,195,20))+[195]
    def callback(error_calculator, epoch):
        if not hasattr(callback, "idx_mark"):
            callback.idx_mark = 0  # 初始化计数器
        len_his = len(error_calculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
              f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")

        if model_type.lower() == 'pretrainnet':
            if epoch == 0:
                plotter.plot_ellipse(Sigma_init, xlim=xlim, ylim=ylim,
                                     facecolor='blue', label='Original region')
                plotter.plot_polygon(error_calculator.A_hat, error_calculator.b_hat, xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation'
                                        , title=f'Training step {epoch}')
            else:
                plotter.remove_shape(plotter.shapes[-1]['id'])
                plotter.plot_polygon(error_calculator.A_hat, error_calculator.b_hat, xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation',
                                     title=f'Training step {epoch}')
            plotter.save(str(figure_folder / f'step{epoch}.png'))
            if callback.idx_mark>=len(marking_epoches):
                with open(result_folder / 'A_list.pkl', "wb") as f:
                    pickle.dump(A_list, f)
                with open(result_folder / 'b_list.pkl', "wb") as f:
                    pickle.dump(b_list, f)
                with open(result_folder / 'error_history.pkl', "wb") as f:
                    pickle.dump(visualizer.error_history, f)
            elif epoch >= marking_epoches[callback.idx_mark]:
                callback.idx_mark+=1
                A_list.append(errorcalculator.A_hat)
                b_list.append(errorcalculator.b_hat)
                visualizer.compute_errors(errorcalculator, num_sample=num_sample)




    # 训练参数配置
    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
            "call_interval": 5,
            "training_callback": callback,
            "optimizer": "SGD",
            "lr": 0.2,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",
            # "optimizer": "sgd",
            "lr": 0.002,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }


    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,
        'dataloader': DataLoader(
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':param_count,
    }
    return {
        'casename': case_name,
        'A_hat': A_hat,
        'b_hat': errorcalculator.b_hat,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'params':params,
        'result_path': result_folder / f'{model_type.lower()}_weights.pth',
        'metadata': {
            'Sigma_init': Sigma_init,
            'dim': dim
        }
    }

def case_epigraph(total_samples=100, noise_scale=0.15, batch_size=5, model_type='pretrainnet',device = 'cpu'):
    """Epigraph问题案例实现"""
    # 初始参数设置
    theta_init = np.array([1.0, 1.0])  # theta[0]=x上界, theta[1]=x²系数
    dim = 1  # 变量维度 (x, f)
    dim_theta = 2

    # 构建Pyomo模型
    model = ConcreteModel()

    # 可调参数定义
    model.theta = Param(range(dim_theta),
                        initialize={i: theta_init[i] for i in range(dim_theta)},
                        mutable=True)

    # 决策变量

    model.var_proj = Var(range(dim+1), domain=Reals) # var_proj[0]=x, var_proj[1]=f

    model.x = model.var_proj[0]  # 前n-1个元素的表达式别名
    model.f = model.var_proj[dim]

    model.obj = Expression(expr=model.theta[1] * model.x ** 2)
    # model.f = Var(domain=Reals)

    # 约束定义
    model.constraints = ConstraintList()
    model.constraints.add(model.x >= 0)  # x下界固定
    model.constraints.add(model.x <= model.theta[0])  # 上界由theta控制

    model.constraints.add(model.f >= model.obj)  # 下界约束
    # model.constraints.add(model.f <= model.theta[1] * model.theta[0]**2)  # Epigraph上界

    original_model = {'model': model}

    # 数据集配置
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # theta维度固定为2

    # 近似器参数 (线性化约束矩阵)

    A_hat = np.vstack([
        [1, 0],  # x <= theta0
        [-1, 0],  # x >= 0
        [0, -1],  # f >= theta1*x² (需要后续处理)
        [2,-1],
        [1, -1],
    ],dtype=float) #所有的A矩阵写在这里
    A_hat = np.vstack([A_hat,[0,1.]]) #这一行不能变，这表示目标函数的上界
    # [0, 1],  # f <= theta0*theta1

    case_name = 'epigraph'
    figure_folder = f'{PROJECT_ROOT}\\results\\{case_name}\\figures'
    os.makedirs(figure_folder, exist_ok=True)

    # 可视化工具
    plotter = ShapeDrawer_2D()
    xlim = [-0.5, 1.5]
    ylim = [-0.5, 1.5]

    def callback(error_calculator, epoch):
        len_his = len(error_calculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
              f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")

        if model_type.lower() == 'pretrainnet':
            if epoch == 0:
                plotter.plot_epigraph(
                    x_range=(0, 1),
                    f_min_func=lambda x: x**2,
                    # facecolor='rgba(135,206,250,0.3)',
                    label="Original epigraph",
                    xlim = xlim,
                    ylim = ylim
                )
                plotter.plot_polygon(np.vstack([error_calculator.A_hat,[0,1]]), np.hstack([error_calculator.b_hat,[error_calculator.fmax]]),
                                     xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation',
                                     title=f'Training step {epoch}')
            else:
                plotter.remove_shape(plotter.shapes[-1]['id'])
                plotter.plot_polygon(np.vstack([error_calculator.A_hat,[0,1]]), np.hstack([error_calculator.b_hat,[error_calculator.fmax]]),
                                     xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation',
                                     title=f'Training step {epoch}')
            plotter.save(f"{figure_folder}/step{epoch}.png")

    # 误差计算器
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        is_epigraph=True,
        solver='ipopt'
    )

    # 训练参数配置
    if model_type.lower() == 'pretrainnet':
        trainer_configure = {
            "call_interval": 10,
            "training_callback": callback,
            "optimizer": "SGD",
            "lr": 0.15,
            "batch_size": 3,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.97},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",
            "lr": 0.01,
            # "optimizer": "SGD",
            # "lr": 0.05,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.99},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
            # "theta_init": theta_init
        }
    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,
        'dataloader': DataLoader(
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':param_count,
    }
    return {
        'casename': case_name,
        'A_hat': A_hat,
        'b_hat': errorcalculator.b_hat,
        'fmax':errorcalculator.fmax,
        'params':params,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'result_path': f'{PROJECT_ROOT}/results/{case_name}/{model_type.lower()}_weights.pth',
    }

def case_nonconvex(
        total_samples=200, noise_scale=0.3, batch_size=5,
        model_type='pretrainnet', device='cpu', save_artifacts=True,
        result_root=None):
    """非凸优化问题案例实现"""
    # 初始参数设置
    theta_init = np.array([1, 1, 1])  # theta[0]: 圆心x偏移,theta[1]: 圆心y偏移, theta[2]: 最小半径
    dim = 2  # 变量维度
    dim_theta = len(theta_init)

    # 构建Pyomo模型
    model = ConcreteModel()

    # 定义可变参数
    model.theta = Param(range(dim_theta),
                        initialize={i: theta_init[i] for i in range(dim_theta)},
                        mutable=True)
    # 定义变量及非对称边界
    def variable_bounds(m, i):
        return (-1, 1)  # var_proj[0] ∈ [-1,2], var_proj[1] ∈ [-2,4]

    model.var_proj = Var(range(dim), domain=Reals, bounds=variable_bounds)

    # 非凸约束定义
    model.constraints = ConstraintList()
    # 单位圆约束 (凸)
    model.constraints.add(model.var_proj[0] ** 2 + model.var_proj[1] ** 2 <= 1)
    # 动态环形约束 (非凸)
    model.constraints.add(
        (model.var_proj[0] - model.theta[0]) ** 2 + (model.var_proj[1] - model.theta[1]) ** 2 >= model.theta[2] ** 2
    )

    original_model = {'model': model}

    # 数据集配置（生成theta扰动）
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # theta维度固定为2

    # 近似器矩阵（包含边界约束）
    A_hat = np.vstack([
        np.eye(dim),  # 上界
        -np.eye(dim),  # 下界
        [[1, 1], [-1, -1]]  # 对角线约束
    ])
    # 误差计算器（支持非凸约束评估）
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='ipopt',  # 使用支持非凸的求解器
    )
    A_list = [errorcalculator.A_hat]
    b_list = [errorcalculator.b_hat]

    visualizer = ErrorVisualizer()
    num_sample = 50
    visualizer.compute_errors(errorcalculator, num_sample=num_sample)

    case_name = 'nonconvex'
    result_root = PROJECT_ROOT / 'results' if result_root is None else Path(result_root)
    result_folder = result_root / case_name
    figure_folder = result_folder / 'figures'
    if save_artifacts:
        figure_folder.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 6))
    # 可视化回调函数
    plotter = ShapeDrawer_2D()
    xlim = (-1.5, 1.5)
    ylim = (-1.5, 1.5)
    marking_epoches = list(range(1,300,30))+[280]

    def callback(error_calculator, epoch):
        if not hasattr(callback, "idx_mark"):
            callback.idx_mark = 0  # 初始化计数器
        len_his = len(error_calculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(error_calculator.training_history['feas'][-min(10, len_his):]):.2e}, "
              f"OptErr={np.mean(error_calculator.training_history['opt'][-min(10, len_his):]):.2e}")
        if model_type.lower() == 'pretrainnet':
            # 绘制原始区域
            if epoch == 0:
                # 单位圆
                plotter.plot_circle_regions(
                    theta=theta_init,
                    xlim=xlim,  # 包含两个圆的可视范围
                    ylim=ylim,
                    edgecolor='skyblue',
                    facecolor='skyblue',  # 区域填充色
                    alpha=0.3,  # 透明度
                    label = 'Nonconvex region'
                )
                plotter.plot_polygon(error_calculator.A_hat, error_calculator.b_hat, xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation'
                                     , title=f'Training step {epoch}')
            else:
                plotter.remove_shape(plotter.shapes[-1]['id'])
                plotter.plot_polygon(error_calculator.A_hat, error_calculator.b_hat, xlim=xlim, ylim=ylim,
                                     facecolor='green', label='Approximation',
                                     title=f'Training step {epoch}')
            plotter.save(str(figure_folder / f'step{epoch}.png'))

            if callback.idx_mark>=len(marking_epoches):
                with open(result_folder / 'A_list.pkl', "wb") as f:
                    pickle.dump(A_list, f)
                with open(result_folder / 'b_list.pkl', "wb") as f:
                    pickle.dump(b_list, f)
                with open(result_folder / 'error_history.pkl', "wb") as f:
                    pickle.dump(visualizer.error_history, f)
            elif epoch >= marking_epoches[callback.idx_mark]:
                callback.idx_mark+=1
                A_list.append(errorcalculator.A_hat)
                b_list.append(errorcalculator.b_hat)
                visualizer.compute_errors(errorcalculator, num_sample=num_sample)




    if model_type.lower() == 'pretrainnet':
        # 训练参数配置
        trainer_configure = {
            "call_interval": 20,
            "training_callback": callback,
            # "optimizer": "Adam",
            "optimizer": "sgd",
            "lr": 0.2,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.99},
            "n_cal": 2,  # 减少校准次数提高稳定性
            "cal_feas": True,
            "cal_opt": True,  # 非凸问题暂不优化目标
            "rate_opt_feas": 1
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",
            "lr": 0.006,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},
            "n_cal": 5,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1,
        }
    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,
        'dataloader': DataLoader(
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':param_count,
    }
    return {
        'casename': case_name,
        'A_hat': A_hat,
        'b_hat': errorcalculator.b_hat,
        'params':params,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'result_path': result_folder / f'{model_type.lower()}_weights.pth',
    }


def case_ball(
        dim=2, total_samples=200, noise_scale=0.15, batch_size=5,
        model_type='pretrainnet', device='cpu', save_artifacts=True,
        result_root=None):
    """非凸优化问题案例实现"""
    # 初始参数设置
    R_init = 1.0  # theta 圆半径
    dim_theta = 1

    # 构建Pyomo模型
    model = ConcreteModel()

    # 定义可变参数
    model.R = Param(initialize= R_init, mutable=True)
    # 定义变量及非对称边界
    def variable_bounds(m, i):
        return (-5, 5)

    model.var_proj = Var(range(dim), domain=Reals, bounds=variable_bounds)

    # 非凸约束定义
    model.constraints = ConstraintList()
    # 单位圆约束 (凸)
    model.constraints.add(expr = sum(model.var_proj[i] ** 2 for i in range(dim))<= model.R)

    original_model = {'model': model}

    # 数据集配置（生成theta扰动）
    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_scale = noise_scale

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'theta':torch.normal(0, self.noise_scale, (dim_theta,),device=device)}  # theta维度固定为2

    # 近似器矩阵（包含边界约束）
    # A_hat = np.vstack([
    #     np.eye(dim),  # 上界
    #     -np.eye(dim),  # 下界
    # ])
    A_hat = np.vstack([
        np.eye(dim),  # 上界
        -np.eye(dim),  # 下界
    ])
    A_hat += np.random.normal(loc=0, scale=0.5/np.sqrt(dim), size=A_hat.shape)


    case_name = 'ball'
    result_root = PROJECT_ROOT / 'results' if result_root is None else Path(result_root)
    results_folder = result_root / case_name
    if save_artifacts:
        results_folder.mkdir(parents=True, exist_ok=True)

    def callback(error_calculator, epoch):
        len_his = len(error_calculator.training_history['feas'])
        e_feas = np.mean(error_calculator.training_history['feas'][-min(10, len_his):])
        e_opt = np.mean(error_calculator.training_history['opt'][-min(10, len_his):])
        print(f"Iter {epoch}: FeasErr={e_feas:.2e}, "
              f"OptErr={e_opt:.2e}")
    # 误差计算器（支持非凸约束评估）
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='gurobi',  # 使用支持非凸的求解器
    )


    if model_type.lower() == 'pretrainnet':
        # 训练参数配置
        trainer_configure = {
            "call_interval": 5,
            "training_callback": callback,
            # "optimizer": "Adam",
            "optimizer": "sgd",
            "lr": 0.3,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 1.00},
            "n_cal": 2,  # 减少校准次数提高稳定性
            "cal_feas": True,
            "cal_opt": True,  # 非凸问题暂不优化目标
            "rate_opt_feas": 1
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": callback,
            "optimizer": "adam",
            "lr": 0.006/dim,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 20, "gamma": 1.0},
            "n_cal": 5,
            "cal_feas": True,
            "cal_opt": True,
            "rate_opt_feas": 1.0,
        }
    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,
        'dataloader': DataLoader(
            CaseData(),
            batch_size=batch_size,
            shuffle=True
        ),
        'count':param_count,
    }
    return {
        'casename': case_name,
        'A_hat': errorcalculator.A_hat,
        'b_hat': errorcalculator.b_hat,
        'params':params,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'result_path': results_folder,
    }

def case_cube(
        dim=2, model_type='pretrainnet', device='cpu', save_artifacts=True,
        result_root=None):
    """非凸优化问题案例实现"""
    # 初始参数设置
    d_init = 1.0

    # 构建Pyomo模型
    model = ConcreteModel()
    def variable_bounds(m, i):
        return (-1, 1)

    model.var_proj = Var(range(dim), domain=Reals, bounds=variable_bounds)

    # 非凸约束定义
    model.constraints = ConstraintList()

    original_model = {'model': model}

    # 近似器矩阵（包含边界约束）
    A_hat = np.vstack([
        np.eye(dim),  # 上界
        -np.eye(dim),  # 下界
    ])

    A_hat += np.random.normal(loc=0, scale=0.5/sqrt(dim), size=A_hat.shape)
    # 误差计算器（支持非凸约束评估）
    errorcalculator = ErrorCalculator(
        original_model=original_model,
        A_hat=A_hat,
        solver='gurobi',  # 使用支持非凸的求解器
    )

    case_name = 'cube'
    result_root = PROJECT_ROOT / 'results' if result_root is None else Path(result_root)
    results_folder = result_root / case_name
    if save_artifacts:
        results_folder.mkdir(parents=True, exist_ok=True)

    def callback(error_calculator, epoch):
        # 初始化end_flag（如果尚未存在）
        if not hasattr(callback, 'end_flag'):
            callback.end_flag = False
            callback.start_flag = True

        # 打开文件用于追加写入
        len_his = len(error_calculator.training_history['feas'])
        e_feas = np.mean(error_calculator.training_history['feas'][-min(10, len_his):])
        e_opt = np.mean(error_calculator.training_history['opt'][-min(10, len_his):])
        print(f"Iter {epoch}: FeasErr={e_feas:.2e}, "
              f"OptErr={e_opt:.2e}")
        with open(results_folder / f'results_dim{dim}.txt', 'a') as f:
            if callback.start_flag:
                f.write(f"Initial: FeasErr={e_feas:.4e}, OptErr={e_opt:.4e}\n")
                callback.start_flag = False
            if (e_feas + e_opt) / 2 < 1e-6 and (not callback.end_flag):
                print(epoch)
                callback.end_flag = True
                # 写入epoch
                f.write(f"Converged at epoch: {epoch}\n")


    trainer_configure = {
        "call_interval": 10,
        "training_callback": callback,
        # "optimizer": "Adam",
        "optimizer": "sgd",
        "lr": 0.3,
        "batch_size": 1,
        "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 1.05},
        "n_cal": 3,  # 减少校准次数提高稳定性
        "cal_feas": True,
        "cal_opt": True,  # 非凸问题暂不优化目标
        "rate_opt_feas": 1
    }
    params_dict, param_count = pyomo_params_to_numpy(model)
    params = { #名字，初值，误差数据集
        'params_dict':params_dict,
        'dataloader': [None],
        'count':param_count,
    }
    return {
        'casename': case_name,
        'A_hat': A_hat,
        'b_hat': errorcalculator.b_hat,
        'params':params,
        'errorcalculator': errorcalculator,
        'trainer_configure': trainer_configure,
        'result_path': results_folder,
    }
