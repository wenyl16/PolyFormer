import time

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
import Simulator.cases.DS_case_3phase as DS_case_3phase

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import tracemalloc

tscases = [
           'case4gs_ts',
           'case118_ts',
           'case300_ts'
           ]

dscases = [
           'case36real_3phase_ds',
           ]
V_list_path = f'{PROJECT_ROOT}\\results\\ds_proj\\td_results\\V_root.pkl'
import pickle
with open(V_list_path, 'rb') as f:
    V_list = pickle.load(f)
res_list = []
solver = pyo.SolverFactory('ipopt')
for dsppc_name in dscases:
    dsppc = getattr(DS_case_3phase,dsppc_name)()

    for tsppc_name in tscases:
        print(dsppc_name,tsppc_name)
        tsppc = getattr(TD_case, tsppc_name)()
        # model_base = TD_case.TScase(tscasedata=tsppc, is_base=True)
        # solver.solve(model_base, tee=True)
        dscasedata_dict = TD_case.define_td_case_data(tsppc, dsppc, ds_percent=0.75, load_threshold=20)  # 注意这里会改tsppc的负荷
        res = {'tscasename':tsppc['casename'],
               'num_ds':len(dscasedata_dict),
                'dscasename':dsppc['casename'],
               # 'base_obj':model_base.obj()
               }

        # 这块是加入近似：
        dscasedata_apx_dict = dscasedata_dict.copy()
        # vertices = TD_case.generate_vertices(dscasedata = dsppc)
        # start_count = time.perf_counter()
        # vertices = TD_case.generate_vertices(dscasedata=dsppc)
        # vtx_time = time.perf_counter() - start_count
        vtx_time = 0
        for key, dscasedata in dscasedata_apx_dict.items():
            start_count = time.perf_counter()
            vertices = TD_case.generate_vertices(dscasedata=dsppc, V_root=V_list[tsppc_name][key])
            vtx_time += time.perf_counter() - start_count
            dscasedata_apx_dict[key] = {'baseMVA': dscasedata['baseMVA'], 'vertex': vertices}
        # model_base = None
        res['vtx_time'] = vtx_time/len(dscasedata_apx_dict)
        #近似模型
        model = TD_case.TDcase(tscasedata=tsppc, dscasedata_dict=dscasedata_apx_dict)  # is_apx控制是否为近似
        num_constraints = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True, descend_into=True))
        num_vars = sum(1 for _ in model.component_data_objects(pyo.Var, active=True, descend_into=True))
        tracemalloc.reset_peak()
        tracemalloc.start()
        results = solver.solve(model, tee=False)
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

        # model = None
        #
        # #原始模型
        # model = TD_case.TDcase(tscasedata=tsppc, dscasedata_dict=dscasedata_dict, is_apx=False)  # is_apx控制是否为近似
        # num_constraints = sum(1 for _ in model.component_data_objects(pyo.Constraint, active=True, descend_into=True))
        # num_vars = sum(1 for _ in model.component_data_objects(pyo.Var, active=True, descend_into=True))
        #
        # tracemalloc.reset_peak()
        # tracemalloc.start()
        # results = solver.solve(model, tee=True)
        # solver_time = results['Solver'][0]['Time']
        # current, peak = tracemalloc.get_traced_memory()
        # tracemalloc.stop()
        # res.update({
        #        'full_ncons': num_constraints,
        #        'full_nvars':num_vars,
        #        'ipopt_obj':model.obj(),
        #        'ipopt_peak_memory_MB':peak / 1024 / 1024,
        #        'ipopt_time':solver_time
        #        })
        model = None

        res_list.append(res)

df = pd.DataFrame(res_list)
print(df)
output_path = f'{PROJECT_ROOT}\\results\\ds_proj\\td_results\\vertex2.xlsx'
df.to_excel(output_path, index=False)  # index=False避免写入行索引
print(f"数据已保存至: {output_path}")