import numpy as np
from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator
import os
import pyomo.environ as pyo
import Simulator.cases.TD_case as TD_case
from Simulator.Approximator import FullNet, PreTrainNet
from Simulator import PROJECT_ROOT
import torch
import pandas as pd

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tracemalloc

tscases = [
           'case4gs_ts',
           'case118_ts',
           'case300_ts'
           ]

dscases = [
           'case10ba_ds',
           # 'case17me_ds',
           # 'case33bw_ds',
           # 'case51ga_ds',
           # 'case74_ds',
           # 'case118zh_ds',
           # 'case136ma_ds',
           # 'case533mt_hi_ds'
           ]
model_type = 'fullnet'
res_list = []
solver = pyo.SolverFactory('ipopt')
for dsppc_name in dscases:
    dsppc = getattr(TD_case,dsppc_name)()
    for tsppc_name in tscases:
        tsppc = getattr(TD_case, tsppc_name)()
        model_base = TD_case.TScase(tscasedata=tsppc, is_base=True)
        solver.solve(model_base, tee=True)

        dscasedata_dict = TD_case.define_td_case_data(tsppc, dsppc, ds_percent=0.75, load_threshold=20)  # 注意这里会改tsppc的负荷
        res = {'tscasename':tsppc['casename'],
               'num_ds':len(dscasedata_dict),
                'dscasename':dsppc['casename'],
               'base_obj':model_base.obj()}


        case = TD_case.DScase_train (casedata=dsppc, model_type=model_type)
        # 这块是加入近似：
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dscasedata_apx_dict = dscasedata_dict.copy()
        weights_path = f'{PROJECT_ROOT}\\results\\ds_proj\\{case["casename"]}\\{model_type}_weights.pth'

        if model_type == 'pretrainnet':
            APA = PreTrainNet(case['A_hat'], case['b_hat'], is_epigraph=False, device=device)
            APA.load_state_dict(torch.load(weights_path, map_location=device))
            A, b = APA()
            A, b = A[0].detach().cpu().numpy(), b[0].detach().cpu().numpy()
            for key, dscasedata in dscasedata_apx_dict.items():
                dscasedata_apx_dict[key] = {'baseMVA': dscasedata['baseMVA'], 'A_hat': A, 'b_hat': b}
        else:  # model_type == 'fullnet'
            APA = FullNet(case['params']['count'], case['A_hat'], case['b_hat'],
                                is_epigraph=False, n_hidden=128, device=device)
            APA.load_state_dict(torch.load(weights_path, map_location=device))
            for key, dscasedata in dscasedata_apx_dict.items():
                A, b = APA(torch.tensor([model_base.V[key]() - 1.0]))
                A, b = A[0].detach().cpu().numpy(), b[0].detach().cpu().numpy()
                dscasedata_apx_dict[key] = {'baseMVA': dscasedata['baseMVA'], 'A_hat': A, 'b_hat': b}
        model_base = None
        #近似模型
        model = TD_case.TDcase(tscasedata=tsppc, dscasedata_dict=dscasedata_apx_dict, is_apx=True)  # is_apx控制是否为近似
        num_constraints = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True, descend_into=True))
        num_vars = sum(1 for _ in model.component_data_objects(pyo.Var, active=True, descend_into=True))
        tracemalloc.reset_peak()
        tracemalloc.start()
        results = solver.solve(model, tee=True)
        solver_time = results['Solver'][0]['Time']
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        res['apx_ncons'] = num_constraints
        res['apx_nvars'] = num_vars
        res['apx_obj'] = model.obj()
        res['apx_peak_memory_MB']=peak / 1024 / 1024
        res['apx_time'] = solver_time
        errors = []
        for tsnode, dscasedata in dscasedata_dict.items():
            P_target = model.DS[tsnode].Pn[1]()
            Q_target = model.DS[tsnode].Qn[1]()
            v_target = model.V[tsnode]()
            errors.append(TD_case.disagg_DS(P_target, Q_target, v_target, dscasedata))

        res['mean_error'] = np.mean(errors)
        res['max_error'] = np.max(errors)

        model = None

        #原始模型

        model = TD_case.TDcase(tscasedata=tsppc, dscasedata_dict=dscasedata_dict, is_apx=False)  # is_apx控制是否为近似
        num_constraints = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True, descend_into=True))
        num_vars = sum(1 for _ in model.component_data_objects(pyo.Var, active=True, descend_into=True))

        tracemalloc.reset_peak()
        tracemalloc.start()
        results = solver.solve(model, tee=True)
        solver_time = results['Solver'][0]['Time']
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        res.update({
               'full_ncons': num_constraints,
               'full_nvars':num_vars,
               'ipopt_obj':model.obj(),
               'ipopt_peak_memory_MB':peak / 1024 / 1024,
               'ipopt_time':solver_time
               })
        model = None

        res_list.append(res)

df = pd.DataFrame(res_list)
print(df)
output_path = f'{PROJECT_ROOT}\\results\\ds_proj\\td_results\\{model_type}.xlsx'
df.to_excel(output_path, index=False)  # index=False避免写入行索引
print(f"数据已保存至: {output_path}")