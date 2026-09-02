import numpy as np
import torch
from pathlib import Path

from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator, pyomo_params_to_numpy
from Simulator.Plotter import ShapeDrawer_2D
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import pyomo.environ as pyo

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import pandas as pd
from scipy.stats import truncnorm

def generate_truncated_normal_samples(mean, std, lower, upper, size):
    """
    生成截断正态分布样本
    """
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm.rvs(a, b, loc=mean, scale=std, size=size)

def generate_asset_samples(params=None):
    """
    生成分组的资产收益样本

    Returns:
        df_samples: pd.DataFrame, 样本数据（含均值和分组信息）
    """
    if params is None:
        params = dict(K=5, N_var=10, N_samples=10,
                      mean_range=(0.03, 0.04), std_range=(0.01, 0.05),
                      lower=-0.1, upper=0.1, do_save=True,
                      data_path=f'{PROJECT_ROOT}/data/DRCC/r_samples',
                      overwrite=False)
    # metadata = params.copy()

    # 逐项解压参数
    K = params["K"]
    N_var = params["N_var"]
    N_samples = params["N_samples"]
    mean_range = params["mean_range"]
    std_range = params["std_range"]
    lower = params["lower"]
    upper = params["upper"]
    do_save = params["do_save"]
    data_path = params["data_path"]
    output_path = Path(f'{data_path}.csv')
    if do_save and output_path.exists() and not params.get('overwrite', False):
        raise FileExistsError(
            f"Refusing to overwrite existing DRCC data: {output_path}. "
            "Choose a new data_path or set overwrite=True explicitly."
        )

    base = N_var // K
    remainder = N_var % K
    assets_per_group = [base + 1 if i < remainder else base for i in range(K)]

    means = np.linspace(mean_range[0], mean_range[1], K)
    stds = np.linspace(std_range[0], std_range[1], K)

    group_risk_params = [
        {"mean": float(mu), "std": float(sigma), "level": i + 1}
        for i, (mu, sigma) in enumerate(zip(means, stds))
    ]

    r_samples_list = []
    group_indices = np.zeros(N_var, dtype=int)
    df_rows = []

    idx = 0
    for k in range(K):
        mean = group_risk_params[k]["mean"]
        std = group_risk_params[k]["std"]
        for _ in range(assets_per_group[k]):
            samples = generate_truncated_normal_samples(
                mean, std, lower=lower, upper=upper, size=N_samples
            )
            r_samples_list.append(samples)
            group_indices[idx] = k
            idx += 1
            df_rows.append([k, np.mean(samples)]+list(samples))

    # 转为 pandas.DataFrame
    df_samples = pd.DataFrame(df_rows,
                              index=[f'{i}' for i in range(N_var)],
                              columns=['group', 'mean_r']+[f'sample_{j+1}' for j in range(N_samples)])
    if do_save:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_samples.to_csv(output_path, index=False)
    return df_samples

def generate_test_samples(N_test, params):
    params_test = params.copy()
    params_test['N_samples'] = N_test
    params_test['data_path']+= '_test'
    return generate_asset_samples(params=params_test)
class DRCCModelBuilder:
    def __init__(self, params):
        r_samples = params['r_samples']
        self.N_vars = r_samples.shape[0]
        sample_cols = [col for col in r_samples.columns if col.startswith("sample_")]

        self.group_dataset = {}
        grouped = r_samples.groupby("group")
        self.group_number = len(grouped)

        for group_id, group_df in grouped:
            mean_r = group_df["mean_r"].values.astype(float)
            samples = group_df[sample_cols].values.astype(float)
            R_limits = params['R_limits'][group_id]

            # 初值计算
            R_mean = np.mean(samples)
            R_std = np.std(samples)
            R_min_init = max(R_limits[0], R_mean - 1.5 * R_std)
            eps_init = 0.1
            rho_init = 10 ** (-4)
            max_group_total_init = (min(1.0, 1.0 / self.group_number + 0.2) + min(1.0, 1.0 / self.group_number + 0.4)) / 2

            self.group_dataset[group_id] = {
                "group_id": group_id,
                "indices": [int(idx) for idx in group_df.index],
                "mean_r": mean_r,
                "samples": samples,
                "R_limits": R_limits,
                'R_min_range': np.array((max(R_limits[0],R_mean-1.6*R_std), max(R_limits[0],R_mean-1.4*R_std))),
                "R_min": R_min_init,
                "eps_range": np.array((0.07, 0.13)),
                "eps": eps_init,
                "rho_range": np.array((10 ** (-6),10 ** (-3.5))),
                "rho": rho_init,
                "max_group_total_range":np.array((min(1.0,1.0/self.group_number+0.2),min(1.0,1.0/self.group_number+0.4))),
                "max_group_total": max_group_total_init,
            }

    def build_original_model(self):
        self.model = pyo.ConcreteModel()
        # Sets
        self.model.K = pyo.Set(initialize=self.group_dataset.keys())  # group indices
        self.model.N_vars = pyo.RangeSet(0,self.N_vars-1)
        self.model.x = pyo.Var(range(self.N_vars),domain = pyo.NonNegativeReals)
        self.model.DRCC = pyo.Block(self.model.K)

        self.model.x_couple = pyo.ConstraintList()
        for k in self.model.K:
            block_data = self.group_dataset[k]
            self.build_drcc_block(block_data)
            for j in self.model.DRCC[k].N_vars:
                self.model.x_couple.add(expr=self.model.DRCC[k].x[j] == self.model.x[block_data['indices'][j]])
        # 统一目标函数（例如，期望利润）
        def objective_rule(m):
            return sum(self.group_dataset[k]['mean_r'][j]*m.DRCC[k].x[j] for k in m.K for j in m.DRCC[k].N_vars)

        self.model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        # 统一约束 总投资=1
        def total_allocation_rule(m):
            return sum(m.x[j] for j in m.N_vars) <= 1
        self.model.allocation = pyo.Constraint(rule=total_allocation_rule)

    def build_drcc_block(self, block_data):
        group_id = block_data['group_id']
        block = self.model.DRCC[group_id]
        N_vars, N_samples = block_data['samples'].shape[0], block_data['samples'].shape[1]
        N_uncertain = N_vars
        # Sets
        block.N_samples = pyo.RangeSet(0, N_samples - 1)
        block.N_vars = pyo.RangeSet(0, N_vars - 1)
        block.N_uncertain = pyo.RangeSet(0, N_uncertain - 1)

        R_limits = block_data['R_limits']

        block.R_min = pyo.Param(initialize=block_data['R_min'], mutable=True, domain=pyo.Reals)
        block.eps = pyo.Param(initialize=block_data['eps'], mutable=True, domain=pyo.Reals)
        block.rho = pyo.Param(initialize=block_data['rho'], mutable=True, domain=pyo.Reals)
        block.max_group_total = pyo.Param(initialize= block_data['max_group_total'], mutable=True, domain=pyo.Reals)

        H = np.vstack([np.eye(N_uncertain), -np.eye(N_uncertain)])
        h = np.hstack([R_limits[1]*np.ones(N_uncertain), -R_limits[0]*np.ones(N_uncertain)])
        block.dim_h = pyo.RangeSet(0, len(h)- 1)

        block.x = pyo.Var(block.N_vars, domain=pyo.NonNegativeReals)  # 决策变量
        block.s = pyo.Var(block.N_samples, domain=pyo.Reals)  # 决策变量
        block.v = pyo.Var(within=pyo.NonNegativeReals)
        block.beta = pyo.Var(within=pyo.Reals)
        block.gamma = pyo.Var(block.N_samples,range(len(h)), domain=pyo.NonNegativeReals)  # 决策变量

        # Constraint 1: Wasserstein term
        def wasserstein_rule(m):
            return m.v * m.rho + (1 / N_samples) * sum(m.s[i] for i in m.N_samples) <= 0

        block.wasserstein_c = pyo.Constraint(rule=wasserstein_rule)

        # Constraint 2: beta ≤ s_i
        def beta_le_si_rule(m, i):
            return m.beta <= m.s[i]
        block.beta_si_c = pyo.Constraint(block.N_samples, rule=beta_le_si_rule)

        # Constraint 3: DRCC inequality
        def drcc_rule(m, i):
            term1 = -sum(m.x[j] * block_data['samples'][j, i] for j in m.N_vars)
            term2 = m.R_min * sum(m.x[j] for j in m.N_vars)
            term3 = (m.eps - 1) * m.beta
            term4 = m.eps * sum(
                m.gamma[i, l] * (h[l] - sum(H[l, j] * block_data['samples'][j, i] for j in m.N_uncertain))
                for l in m.dim_h
            )
            rhs = m.eps * m.s[i]
            return term1 + term2 + term3 + term4 <= rhs

        block.drcc_c = pyo.Constraint(block.N_samples, rule=drcc_rule)

        # Constraint 4a: dual norm
        def dual_norm_upper_rule(m, i, j):
            norm_expr = m.eps * sum(H[l, j] * m.gamma[i, l] for l in m.dim_h) + m.x[j]
            return norm_expr <= m.eps * m.v

        block.norm_upper = pyo.Constraint(block.N_samples, block.N_uncertain, rule=dual_norm_upper_rule)
        # Constraint 4a: dual norm
        def dual_norm_lower_rule(m, i, j):
            norm_expr = m.eps * sum(H[l, j] * m.gamma[i, l] for l in m.dim_h) + m.x[j]
            return norm_expr >= - m.eps * m.v

        block.norm_lower = pyo.Constraint(block.N_samples, block.N_uncertain, rule=dual_norm_lower_rule)

        # Constraint 5: max_group_total
        def max_group_total_rule(m):
            return sum(m.x[j] for j in m.N_vars)<=m.max_group_total

        block.max_group_total_cons = pyo.Constraint(rule=max_group_total_rule)

    def update_original_params(self, update_params_dict):
        """
        更新模型中每个 DRCC block 的参数，并返回更新差值。

        每次先从 self.group_dataset 中读取初始值，再进行指定更新。
        返回值是每个 group_id 的参数变化量（更新值 - 原始初始值），
        顺序为 [R_min, eps, rho, max_group_total]。

        参数:
            update_params_dict: dict[group_id -> dict[param_name -> new_value]]

        返回:
            dict[group_id -> list of 4 deltas]
        """
        meta_data_dict = {}
        def _normalize(x, xrange):
            xmin, xmax = xrange
            if xmax == xmin:
                return 0.5
            return (x - xmin) / (xmax - xmin)
        def _denormalize(x_normalized, xrange):
            xmin, xmax = xrange
            return (x_normalized)*(xmax - xmin)+xmin

        for group_id, block_data in self.group_dataset.items():
            block = self.model.DRCC[group_id]

            # 获取原始初始值
            init_R_min = block_data['R_min']
            init_eps = block_data['eps']
            init_rho = block_data['rho']
            init_max_group_total = block_data['max_group_total']

            # 初始化为初始值
            block.R_min.set_value(init_R_min)
            block.eps.set_value(init_eps)
            block.rho.set_value(init_rho)
            block.max_group_total.set_value(init_max_group_total)

            # 初始为零变化
            R_min_delta = 0.0
            eps_delta = 0.0
            rho_delta = 0.0
            max_group_total_delta = 0.0

            # 如果有更新，则替换并计算差值
            if group_id in update_params_dict:
                updates = update_params_dict[group_id]
                if 'R_min' in updates:
                    new_val = updates['R_min']
                    R_min_delta = _normalize(new_val, block_data['R_min_range'])-_normalize(init_R_min, block_data['R_min_range'])
                    block.R_min.set_value(new_val)
                if 'eps' in updates:
                    new_val = updates['eps']
                    eps_delta = _normalize(new_val, block_data['eps_range'])-_normalize(init_eps, block_data['eps_range'])
                    block.eps.set_value(new_val)
                if 'rho' in updates:
                    new_val = updates['rho']
                    rho_delta = _normalize(new_val, block_data['rho_range']) - _normalize(init_rho, block_data['rho_range'])
                    block.rho.set_value(new_val)
                if 'max_group_total' in updates:
                    new_val = updates['max_group_total']
                    max_group_total_delta = _normalize(new_val, block_data['max_group_total_range']) - _normalize(init_max_group_total, block_data['max_group_total_range'])
                    block.max_group_total.set_value(new_val)

            # 保存 delta
            meta_data_dict[group_id] = [
                R_min_delta,
                eps_delta,
                rho_delta,
                max_group_total_delta
            ]

        return meta_data_dict

    def build_drcc_train(
            self, casedata, model_type='pretrainnet', plot_flag=False,
            total_samples=100, batch_size=5, device='cpu', save_artifacts=True,
            result_root=None):
        group_id = casedata['group_id']
        group_number = self.group_number
        model = pyo.ConcreteModel()
        N_vars, N_samples = casedata['samples'].shape[0], casedata['samples'].shape[1]
        N_uncertain = N_vars
        # Sets
        model.N_samples = pyo.RangeSet(0, N_samples - 1)
        model.N_vars = pyo.RangeSet(0, N_vars - 1)
        model.N_uncertain = pyo.RangeSet(0, N_uncertain - 1)
        R_limits = casedata['R_limits']

        def _normalize(x, xrange):
            xmin, xmax = xrange
            if xmax == xmin:
                return 0.5
            return (x - xmin) / (xmax - xmin)
        R_min_meta_init = _normalize(casedata['R_min'], casedata['R_min_range'])
        eps_meta_init = _normalize(casedata['eps'], casedata['eps_range'])
        rho_meta_init = _normalize(casedata['rho'], casedata['rho_range'])
        max_total_meta_init = _normalize(casedata['max_group_total'], casedata['max_group_total_range'])

        model.R_min_meta = pyo.Param(initialize=R_min_meta_init, mutable=True, domain=pyo.Reals)
        model.eps_meta = pyo.Param(initialize=eps_meta_init, mutable=True, domain=pyo.Reals)
        model.rho_meta = pyo.Param(initialize=rho_meta_init, mutable=True, domain=pyo.Reals)
        model.max_total_meta = pyo.Param(initialize= max_total_meta_init, mutable=True, domain=pyo.Reals)

        def _denormalize(x_normalized, xrange):
            xmin, xmax = xrange
            return (x_normalized)*(xmax - xmin)+xmin

        model.R_min = pyo.Expression(expr=_denormalize(model.R_min_meta, casedata['R_min_range']))
        model.eps = pyo.Expression(expr=_denormalize(model.eps_meta, casedata['eps_range']))
        model.rho = pyo.Expression(expr=_denormalize(model.rho_meta, casedata['rho_range']))
        model.max_total = pyo.Expression(expr=_denormalize(model.max_total_meta, casedata['max_group_total_range']))

        H = np.vstack([np.eye(N_uncertain), -np.eye(N_uncertain)])
        h = np.hstack([R_limits[1]*np.ones(N_uncertain), -R_limits[0]*np.ones(N_uncertain)])
        model.dim_h = pyo.RangeSet(0, len(h)- 1)

        model.x = pyo.Var(model.N_vars, domain=pyo.NonNegativeReals)  # 决策变量
        model.var_proj = pyo.Var(model.N_vars, domain=pyo.NonNegativeReals)  # 投影变量
        model.s = pyo.Var(model.N_samples, domain=pyo.Reals)  # 决策变量
        model.v = pyo.Var(within=pyo.NonNegativeReals)
        model.beta = pyo.Var(within=pyo.Reals)
        model.gamma = pyo.Var(model.N_samples,range(len(h)), domain=pyo.NonNegativeReals)  # 决策变量

        # Constraint 1: Wasserstein term
        def wasserstein_rule(m):
            return m.v * m.rho + (1 / N_samples) * sum(m.s[i] for i in m.N_samples) <= 0

        model.wasserstein_c = pyo.Constraint(rule=wasserstein_rule)

        # Constraint 2: beta ≤ s_i
        def beta_le_si_rule(m, i):
            return m.beta <= m.s[i]
        model.beta_si_c = pyo.Constraint(model.N_samples, rule=beta_le_si_rule)

        # Constraint 3: DRCC inequality
        def drcc_rule(m, i):
            term1 = -sum(m.x[j] * casedata['samples'][j, i] for j in m.N_vars)
            term2 = m.R_min * sum(m.x[j] for j in m.N_vars)
            term3 = (m.eps - 1) * m.beta
            term4 = m.eps * sum(
                m.gamma[i, l] * (h[l] - sum(H[l, j] * casedata['samples'][j, i] for j in m.N_uncertain))
                for l in m.dim_h
            )
            rhs = m.eps * m.s[i]
            return term1 + term2 + term3 + term4 <= rhs

        model.drcc_c = pyo.Constraint(model.N_samples, rule=drcc_rule)

        # Constraint 4a: dual norm <=
        def dual_norm_upper_rule(m, i, j):
            norm_expr = m.eps * sum(H[l, j] * m.gamma[i, l] for l in m.dim_h) + m.x[j]
            return norm_expr <= model.eps * m.v

        model.norm_upper = pyo.Constraint(model.N_samples, model.N_uncertain, rule=dual_norm_upper_rule)
        # Constraint 4a: dual norm >=
        def dual_norm_lower_rule(m, i, j):
            norm_expr = m.eps * sum(H[l, j] * m.gamma[i, l] for l in m.dim_h) + m.x[j]
            return norm_expr >= - m.eps * m.v

        model.norm_lower = pyo.Constraint(model.N_samples, model.N_uncertain, rule=dual_norm_lower_rule)

        # Constraint 5: max_group_total
        def max_total_rule(m):
            return sum(m.x[j] for j in m.N_vars)<= m.max_total

        model.max_total_cons = pyo.Constraint(rule=max_total_rule)

        def projection_rule(m, j):
            return m.var_proj[j] == m.x[j]
        model.projection = pyo.Constraint(model.N_vars, rule=projection_rule)


        class CaseData(Dataset):
            def __init__(self, size=total_samples):
                self.size = size

            def __len__(self):
                return self.size

            def __getitem__(self, idx):
                return {
                    'R_min_meta': torch.rand(1, device=device)-R_min_meta_init,  # 归一化值 ∈ [0,1]
                    'eps_meta': torch.rand(1, device=device)-eps_meta_init,
                    'rho_meta': torch.rand(1, device=device)-rho_meta_init,
                    'max_total_meta': torch.rand(1, device=device)-max_total_meta_init,
                }
        dim = N_vars
        A_hat = np.vstack([
            np.eye(dim),
            -np.eye(dim),
            (np.ones((dim,dim))-np.eye(dim))/np.sqrt(dim-1),
            -(np.ones((dim,dim))+np.eye(dim))/np.sqrt(dim-1),
            np.ones(dim)/np.sqrt(dim),
            -np.ones(dim)/np.sqrt(dim),
        ])
        errorcalculator = ErrorCalculator(
            original_model={'model': model},
            A_hat=A_hat,
            solver='gurobi',
        )

        case_name = f'x{self.N_vars}g{group_number}s{N_samples}'
        result_root = PROJECT_ROOT / 'results' if result_root is None else Path(result_root)
        result_folder = result_root / 'DRCC' / case_name / f'g{group_id}'
        if save_artifacts:
            result_folder.mkdir(parents=True, exist_ok=True)

        if model_type.lower() == 'pretrainnet' and plot_flag and save_artifacts:
            plt.figure(figsize=(8, 6))
            xlim = np.array((0.,1.))
            ylim = np.array((0.,1.))
            plotter = ShapeDrawer_2D()
            plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,
                                 facecolor='green', xlim=xlim, ylim=ylim,
                                 label=f'Approximation',
                                 title=f'Training step = {0}'
                                 )

            pretrain_folder = result_folder / 'figures' / 'pretrain_process'
            pretrain_folder.mkdir(parents=True, exist_ok=True)
            plotter.save(pretrain_folder / f'step0{0}.png')

        n_train = 30

        def training_callback(errorcalculator, epoch):
            len_his = len(errorcalculator.training_history['feas'])
            print(f"Iter {epoch}: FeasErr={np.mean(errorcalculator.training_history['feas'][-min(10, len_his):]):.2e}, "
                  f"OptErr={np.mean(errorcalculator.training_history['opt'][-min(10, len_his):]):.2e}")
            if model_type.lower() == 'pretrainnet' and plot_flag and save_artifacts:
                plotter.remove_shape(plotter.shapes[-1]['id'])
                plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,
                                     facecolor='green', xlim=xlim, ylim=ylim,
                                     label=f'Approximation',
                                     title=f'Training step = {epoch}'
                                     )
                plotter.save(pretrain_folder / f'step{epoch}.png')
        # 训练参数配置
        if model_type.lower() == 'pretrainnet':
            trainer_configure = {
                "call_interval": 5,
                "training_callback": training_callback,
                "optimizer": 'sgd',
                "lr": 2e-5,
                "batch_size": 1,
                "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.98},
                "n_cal": 5,
                "cal_feas": True,
                "cal_opt": True,
                'feas_tol': 1e-10,
                'opt_tol': 1e-10,
                "rate_opt_feas": 1.0
            }
        else:
            trainer_configure = {
                "call_interval": 1,
                "training_callback": training_callback,
                "optimizer": "adam",
                # "optimizer": "sgd",
                "lr": 4e-7,
                "batch_size": batch_size,
                "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.97},
                "n_cal": 2,
                "cal_feas": True,
                "cal_opt": True,
                'feas_tol': 1e-10,
                'opt_tol': 1e-10,
                "rate_opt_feas": 1.0,
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
            'A_hat': A_hat,
            'b_hat': errorcalculator.b_hat,
            'errorcalculator': errorcalculator,
            'trainer_configure': trainer_configure,
            'params': params,
            'result_path': result_folder / f'{model_type}_weights.pth',
            'n_train': n_train,
            'metadata': {
                'dscasedata': casedata,
            }
        }

    def build_apx_model(self, apx_data):
        self.apx_model = pyo.ConcreteModel()

        # Sets
        self.apx_model.K = pyo.Set(initialize=self.group_dataset.keys())  # group indices
        self.apx_model.N_vars = pyo.RangeSet(0, self.N_vars - 1)
        self.apx_model.x = pyo.Var(self.apx_model.N_vars, domain=pyo.NonNegativeReals)
        self.apx_model.Apx = pyo.Block(self.apx_model.K)

        # Constraint list to couple block variables with global x
        self.apx_model.x_couple = pyo.ConstraintList()

        for k in self.apx_model.K:
            A = apx_data[k]['A']  # 2D numpy array
            b = apx_data[k]['b']  # 1D numpy array

            block_data = self.group_dataset[k]
            indices = block_data['indices']

            n_constr, n_vars = A.shape
            block = self.apx_model.Apx[k]

            block.N_constr = pyo.RangeSet(0, n_constr - 1)
            block.N_vars = pyo.RangeSet(0, n_vars - 1)

            # Local x vars for compatibility (not used in objective)
            block.x = pyo.Var(block.N_vars, domain=pyo.NonNegativeReals)

            # A and b as mutable Params
            block.A = pyo.Param(block.N_constr, block.N_vars, initialize={
                (i, j): float(A[i, j]) for i in range(n_constr) for j in range(n_vars)
            }, mutable=True)

            block.b = pyo.Param(block.N_constr, initialize={
                i: float(b[i]) for i in range(n_constr)
            }, mutable=True)

            # A x <= b constraints
            def apx_constraint_rule(m, i):
                return sum(m.A[i, j] * m.x[j] for j in m.N_vars) <= m.b[i]

            block.constraints = pyo.Constraint(block.N_constr, rule=apx_constraint_rule)

            # Bind block.x to global x
            for local_j, global_j in enumerate(indices):
                self.apx_model.x_couple.add(block.x[local_j] == self.apx_model.x[global_j])

        # Objective function: same as original
        def objective_rule(m):
            return sum(self.group_dataset[k]['mean_r'][j] * m.Apx[k].x[j]
                       for k in m.K for j in m.Apx[k].N_vars)

        self.apx_model.obj = pyo.Objective(rule=objective_rule, sense=pyo.maximize)

        # Total allocation constraint
        def total_allocation_rule(m):
            return sum(m.x[j] for j in m.N_vars) <= 1.0

        self.apx_model.allocation = pyo.Constraint(rule=total_allocation_rule)

    def update_apx_params(self, new_apx_data):
        """
        用于更新近似模型中每个 group 的 A 和 b 参数。

        参数：
            new_apx_data: dict[group_id -> {'A': np.array, 'b': np.array}]
        """
        for group_id, data in new_apx_data.items():
            block = self.apx_model.Apx[group_id]
            # 更新 A 参数
            if group_id in new_apx_data:
                if 'A' in new_apx_data[group_id]:
                    A_new = new_apx_data[group_id]['A']
                    n_constr, n_vars = A_new.shape
                    for i in range(n_constr):
                        for j in range(n_vars):
                            block.A[i, j] = float(A_new[i, j])

            if 'b' in new_apx_data[group_id]:
                b_new = new_apx_data[group_id]['b']
                n_constr = len(b_new)
                for i in range(n_constr):
                    block.b[i] = float(b_new[i])

    def solve(self, is_apx=False, solver_name='gurobi', tee=True):
        import time
        import tracemalloc

        solver = pyo.SolverFactory(solver_name)

        # 选择模型
        model = self.apx_model if is_apx else self.model

        # 启动资源监控
        tracemalloc.reset_peak()
        tracemalloc.start()
        start_solve = time.time()

        # 求解
        results = solver.solve(model, tee=tee)
        end_solve = time.time()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # 提取求解器状态
        status = results.solver.status
        termination = results.solver.termination_condition

        # 检查是否最优
        if (status == pyo.SolverStatus.ok) and (termination == pyo.TerminationCondition.optimal):
            x_vals = np.array([pyo.value(model.x[i]) for i in range(self.N_vars)])
            obj_val = pyo.value(model.obj)
            num_constraints = sum(
                1 for _ in model.component_data_objects(pyo.Constraint, active=True, descend_into=True))
            num_vars = sum(1 for _ in model.component_data_objects(pyo.Var, active=True, descend_into=True))

            # print(f"[SOLVE SUCCESS] Objective = {obj_val:.6f}")
            return {
                "x": x_vals,
                "obj": obj_val,
                "num_constraints": num_constraints,
                "num_vars": num_vars,
                "solve_time": end_solve - start_solve,
                "peak_memory_MB": peak / 1024 / 1024,
            }

        else:
            # print(f"[SOLVE FAILED] Status: {status}, Termination: {termination}")
            return None

    def evaluate_solution(self, x_vals, r_test_samples):
        """
        评估给定解 x 在测试集上的目标值和违反DRCC约束的概率。
        """

        group_violation_prob = {}
        group_violation_value = {}

        total_obj = 0.0

        sample_cols = [col for col in r_test_samples.columns if col.startswith("sample_")]

        for group_id, group_df in r_test_samples.groupby("group"):
            if group_id not in self.group_dataset:
                raise ValueError(f"Group {group_id} not found in training data.")

            group_data = self.group_dataset[group_id]
            indices = group_data['indices']
            R_k = group_data['R_min']  # 注意我们用 R_min 作下界

            group_x = np.array([x_vals[i] for i in indices])
            sum_x = group_x.sum()

            # mean objective contribution
            mean_r = group_df["mean_r"].values.astype(float)
            total_obj += np.dot(mean_r, group_x)

            # test samples
            sample_matrix = group_df[sample_cols].values.astype(float)  # shape: (n_assets, n_samples)
            # 转置为 shape: (n_samples, n_assets)
            sample_matrix = sample_matrix.T

            violations = 0
            sum_violations = 0
            for r_sample in sample_matrix:
                lhs = np.dot(r_sample, group_x)
                rhs = R_k * sum_x
                if lhs < rhs:
                    violations += 1
                    sum_violations +=rhs-lhs

            prob_violation = violations / sample_matrix.shape[0]
            mean_violation = sum_violations / sample_matrix.shape[0]
            group_violation_prob[group_id] = prob_violation
            group_violation_value[group_id] = mean_violation

        return {
            "test_objective": total_obj,
            "violation_probabilities": group_violation_prob,
            "violation_value": group_violation_value
        }


if __name__ == "__main__":
    print(
        "Use `python -m Simulator.runners.main_drcc --help` for the paper DRCC "
        "workflows. The sample-generation functions remain available as an API."
    )


