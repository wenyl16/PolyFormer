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

from scipy.io import loadmat

def define_nodal_flex_P(ppc, percent = 0.3, rate = 0.3):
    # Create list of (node, load) tuples
    node_loads = [(int(bus_data[0]), bus_data[2]) for bus_data in ppc["bus"]]
    # Sort nodes by load in descending order
    sorted_nodes = sorted(node_loads, key=lambda x: x[1], reverse=True)
    # Determine the top 50% nodes with the highest load
    num_nodes = len(sorted_nodes)
    top_percent = round(num_nodes*percent)
    top_nodes = {node for node, _ in sorted_nodes[:top_percent]}

    # Create node_flex_dict
    node_flex_dict = {}
    for node, _ in node_loads:
        if node in top_nodes:
            node_flex_dict[node] = {"type": 2, "rate": (rate,rate)}
        else:
            node_flex_dict[node] = {"type": 0, "rate": None}
    ppc["node_flex_dict"] = node_flex_dict



load_file_path = PROJECT_ROOT / 'data' / 'real_dis_data' / 'load_file.xls'
volt_file_path = PROJECT_ROOT / 'data' / 'real_dis_data' / 'volt_file.xls'
target_time = "2024-09-30 14:00"
def case36real_3phase_ds(flex_percent = 0.5, flex_rate = 0.3):
    """PyPower case data for 36-node system with separated PQ and R/X data"""
    missing_inputs = [
        path for path in (load_file_path, volt_file_path) if not path.is_file()
    ]
    if missing_inputs:
        formatted = "\n  - ".join(
            path.relative_to(PROJECT_ROOT).as_posix() for path in missing_inputs
        )
        raise FileNotFoundError(
            "The real-world three-phase case requires two anonymized input "
            f"workbooks that are not present:\n  - {formatted}\n"
            "See the Data inventory section in README.md."
        )
    baseMVA = 1.0
    basekV = 10
    low_volt_norm = 0.23
    R_dict = {}
    X_dict = {}
    line_info = {'JKLYJ-150':{'R':0.226, 'X':0.557, 'Xm':0.223},
                 'JKLYJ-185':{'R':0.183, 'X':0.5995, 'Xm':0.2725},
                 'JKLJ-120':{'R':0.235, 'X':0.6912, 'Xm':0.3142},
                 'JKLYJ-120':{'R':0.260, 'X':0.756, 'Xm':0.419}
                 }
    for key, value in line_info.items():
        R_dict[key] = value['R']*np.eye(3)
        X = value['X']
        Xm = value['Xm']
        X_dict[key] = np.array([[X, Xm, Xm], [Xm, X, Xm], [Xm, Xm, X]])

    branch_info  = [
    {"length": 261.2, "type": "JKLYJ-150"},
    {"length": 577.6, "type": "JKLYJ-150"},
    {"length": 615.7, "type": "JKLYJ-150"},
    {"length": 265.07, "type": "JKLYJ-185"},
    {"length": 108.8, "type": "JKLYJ-185"},
    {"length": 562.0, "type": "JKLYJ-185"},
    {"length": 140.6, "type": "JKLYJ-185"},
    {"length": 120.6, "type": "JKLYJ-185"},
    {"length": 703.2, "type": "JKLYJ-150"},
    {"length": 82.85, "type": "JKLYJ-150"},

    {"length": 420.7, "type": "JKLYJ-185"},
    {"length": 102.5, "type": "JKLYJ-185"},
    {"length": 218.17, "type": "JKLYJ-185"},
    {"length": 224.95, "type": "JKLYJ-185"},
    {"length": 206.395, "type": "JKLYJ-185"},
    {"length": 527.33, "type": "JKLYJ-185"},
    {"length": 19.7027, "type": "JKLYJ-185"},

    {"length": 55.966, "type": "JKLJ-120"},
    {"length": 353.8, "type": "JKLJ-120"},
    {"length": 1291.7, "type": "JKLJ-120"},
    {"length": 714.0, "type": "JKLJ-120"},
    {"length": 1157.27, "type": "JKLJ-120"},
    {"length": 1399.12, "type": "JKLJ-120"},

    {"length": 46.422, "type": "JKLJ-120"},
    {"length": 36.956, "type": "JKLJ-120"},
    {"length": 198.0, "type": "JKLJ-120"},
    {"length": 43.76, "type": "JKLJ-120"},
    {"length": 141.75, "type": "JKLJ-120"},
    {"length": 61.62, "type": "JKLJ-120"},
    {"length": 19.99, "type": "JKLJ-120"},

    {"length": 44.21, "type": "JKLYJ-120"},
    {"length": 55.407, "type": "JKLYJ-120"},
    {"length": 43.767, "type": "JKLYJ-120"},
    {"length": 44.026, "type": "JKLYJ-120"},
    {"length": 143.23, "type": "JKLYJ-120"}
]

    # 构建bus矩阵 (使用索引引用PQ_list)

    root_voltage = get_voltage_at_time(volt_file_path, target_time)

    bus = np.array([
        [1, 3, 0, 0, 0, 0, 1, root_voltage/basekV, 0, basekV, 1, 1.100, 0.900],
        [2, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [3, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [4, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [5, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [6, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [7, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [8, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [9, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [10, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [11, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [12, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [13, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [14, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [15, 2, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [16, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [17, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [18, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [19, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [20, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [21, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [22, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [23, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [24, 2, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [25, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [26, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [27, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [28, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [29, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [30, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [31, 2, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [32, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [33, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [34, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [35, 1, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900],
        [36, 2, 0, 0, 0, 0, 1, 1, 0, low_volt_norm, 1, 1.100, 0.900]
    ], dtype=float)

    node_load_mapping = {
        5: [27],
        6: [26],
        7: [18],
        8: [20],
        11: [13],
        13: [1, 6, 8],
        14: [16],
        17: [19],
        18: [25],
        21: [24],
        22: [9, 14],
        24: [10, 12],
        26: [28],
        28: [15, 21],
        30: [17],
        32: [7],
        33: [2, 5, 11],
        34: [3]
    }

    # node_flex_dict = {# 0: No_flex, 1:Pflex, 2:PQflex,
    #     5: 0,
    #     6: 0,
    #     7: 0,
    #     8: 0,
    #     11: 0,
    #     13: 0,
    #     14: 0,
    #     17: 1,
    #     18: 1,
    #     21: 1,
    #     22: 1,
    #     24: 1,
    #     26: 1,
    #     28: 1,
    #     30: 1,
    #     32: 1,
    #     33: 1,
    #     34: 1
    # }


    n_bus = bus.shape[0]
    power_data = read_phase_power(load_file_path, target_time)
    load_volt = [power_data[i]['Ua'] for i in range(len(power_data))]
    bus_P = np.zeros([n_bus,3])
    bus_Q = np.zeros([n_bus, 3])
    for key,value in node_load_mapping.items():
        for load_idx in value:
            bus_P[key-1,:]+= np.array([power_data[load_idx-1]['Pa'], power_data[load_idx-1]['Pb'], power_data[load_idx-1]['Pc']])
            bus_Q[key-1,:]+= np.array([power_data[load_idx-1]['Qa'], power_data[load_idx-1]['Qb'], power_data[load_idx-1]['Qc']])
    # 单相有功/无功功率的独立列表 (便于集中修改)
    PQ_list = np.hstack((np.sum(bus_P, axis=1, keepdims=True),np.sum(bus_Q   , axis=1, keepdims=True)))
    # 注入PQ_list中的单相有功/无功数据
    bus[:, 2] = PQ_list[:, 0]*1e-3  # 注入有功(P, p.u.)
    bus[:, 3] = PQ_list[:, 1]*1e-3   # 注入无功(Q, p.u.)

    branch_R_mat = []
    branch_X_mat = []
    for i in range(len(branch_info)):
        branch_R_mat.append(R_dict[branch_info[i]['type']] * branch_info[i]['length']*1e-3)
        branch_X_mat.append(X_dict[branch_info[i]['type']] * branch_info[i]['length']*1e-3)
    # 构建branch矩阵 (初始将电阻、电抗设为0)
    branch = np.array([
        [1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路1
        [2, 3, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路2
        [3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路3
        [4, 5, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路4
        [5, 6, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路5
        [6, 7, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路6
        [7, 8, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路7
        [8, 9, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路8
        [9, 10, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路9
        [10, 11, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路10
        [11, 12, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路11
        [12, 13, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路12
        [13, 14, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路13
        [14, 15, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路14
        [2, 16, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路15
        [16, 17, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路16
        [17, 18, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路17
        [8, 19, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路18
        [19, 20, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路19
        [20, 21, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路20
        [21, 22, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路21
        [22, 23, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路22
        [23, 24, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路23
        [23, 25, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路24
        [25, 26, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路25
        [26, 27, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路26
        [27, 28, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路27
        [28, 29, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路28
        [29, 30, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路29
        [30, 31, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路30
        [12, 32, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路31
        [32, 33, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路32
        [33, 34, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路33
        [34, 35, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360],  # 支路34
        [35, 36, 0, 0, 0, 0, 0, 0, 0, 0, 1, -360, 360]  # 支路35
    ], dtype=float)
    # 注入RX_list中的单相电阻、电抗数据
    RX_list = np.array([[mat[0][0] for mat in branch_R_mat],[mat[0][0] for mat in branch_X_mat]])
    Vbase = basekV*1e3
    Sbase = baseMVA*1e6
    Zbase = Vbase**2/Sbase
    branch[:, 2] = RX_list[0,:].T/Zbase  # 注入电阻(R)
    branch[:, 3] = RX_list[1,:].T/Zbase  # 注入电抗(X)

    branch_R_mat_rated = [mat/Zbase for mat in branch_R_mat]
    branch_X_mat_rated = [mat / Zbase for mat in branch_X_mat]

    # 发电机数据
    gen = np.array([
        [1, 0, 0, 10, -10, root_voltage/basekV, 100, 1, 10, -10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=float)

    # 发电机成本数据
    gencost = np.array([
        [2, 0, 0, 1, 0.48, 0]
    ], dtype=float)
    ppc = {
        'version': '2',
        'casename':'case36real_3phase_ds',
        'baseMVA': baseMVA,
        'bus': bus,
        'branch': branch,
        'gen': gen,
        'gencost': gencost,
        # 'loadvolt':load_volt,
        'branch_R_mat':branch_R_mat_rated,
        'branch_X_mat':branch_X_mat_rated,
        'bus_P': bus_P*1e-3,
        'bus_Q': bus_Q*1e-3,
        # 'node_flex_dict':node_flex_dict
        # 返回参数列表便于外部修改
        # 'PQ_list': PQ_list.tolist(),
        # 'RX_list': RX_list.tolist()
    }
    define_nodal_flex_P(ppc, percent=flex_percent, rate=flex_rate)
    return ppc
#


import pandas as pd
# from datetime import datetime

def read_phase_power(file_path, target_time_str):
    """Read all load-point sheets and calculate per-phase P/Q at one time."""
    expected_columns = [
        'management_unit', 'service_center', 'customer_id', 'customer_name',
        'timestamp', 'asset_id', 'instantaneous_active_power',
        'phase_a_current', 'phase_b_current', 'phase_c_current',
        'neutral_current', 'phase_a_voltage', 'phase_b_voltage',
        'phase_c_voltage', 'total_power_factor',
        'cumulative_forward_active_energy',
        'cumulative_reverse_active_energy', 'quadrant_i_reactive_energy',
        'quadrant_iv_reactive_energy', 'ct_ratio', 'pt_ratio',
        'logical_address', 'is_supplemental_reading', 'ingestion_timestamp'
    ]
    expected_sheets = [f'Load_Point_{index:02d}' for index in range(1, 29)]
    all_sheets = pd.read_excel(file_path, sheet_name=None)
    if list(all_sheets) != expected_sheets:
        raise ValueError(
            "The load workbook must contain Load_Point_01 through "
            "Load_Point_28 in that order."
        )

    target_time = pd.to_datetime(target_time_str)
    results = []
    for sheet_name, df in all_sheets.items():
        if list(df.columns) != expected_columns:
            raise ValueError(f"Unexpected column schema in worksheet '{sheet_name}'.")
        timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
        time_data = df[timestamps.dt.floor('min') == target_time]
        if time_data.empty:
            raise ValueError(
                f"No load record found at {target_time_str} in worksheet "
                f"'{sheet_name}'."
            )

        row = time_data.iloc[0]
        required = [
            'phase_a_current', 'phase_b_current', 'phase_c_current',
            'phase_a_voltage', 'phase_b_voltage', 'phase_c_voltage',
            'total_power_factor'
        ]
        numeric = pd.to_numeric(row[required], errors='coerce')
        if numeric.isna().any():
            missing = ', '.join(numeric.index[numeric.isna()])
            raise ValueError(
                f"Non-numeric values in worksheet '{sheet_name}': {missing}."
            )

        current_scale = 1 if sheet_name == 'Load_Point_10' else 120
        Ia = numeric['phase_a_current'] * current_scale
        Ib = numeric['phase_b_current'] * current_scale
        Ic = numeric['phase_c_current'] * current_scale

        Ua_raw = numeric['phase_a_voltage']
        Ub_raw = numeric['phase_b_voltage']
        Uc_raw = numeric['phase_c_voltage']
        voltage_scale = 2.3 if any(
            phase_voltage - 100 <= 10
            for phase_voltage in (Ua_raw, Ub_raw, Uc_raw)
        ) else 1.0
        Ua = Ua_raw * voltage_scale
        Ub = Ub_raw * voltage_scale
        Uc = Uc_raw * voltage_scale
        cos_phi = np.clip(numeric['total_power_factor'], -1.0, 1.0)
        sin_phi = np.sqrt(1 - cos_phi ** 2)

        Pa = Ia * Ua * cos_phi * 1e-3
        Pb = Ib * Ub * cos_phi * 1e-3
        Pc = Ic * Uc * cos_phi * 1e-3
        Qa = Ia * Ua * sin_phi * 1e-3
        Qb = Ib * Ub * sin_phi * 1e-3
        Qc = Ic * Uc * sin_phi * 1e-3

        results.append({
            'sheet_name': sheet_name,
            'target_time': target_time_str,
            'Pa': round(Pa, 4),
            'Pb': round(Pb, 4),
            'Pc': round(Pc, 4),
            'Qa': round(Qa, 4),
            'Qb': round(Qb, 4),
            'Qc': round(Qc, 4),
            'Ua': round(Ua, 4),
            'Ub': round(Ub, 4),
            'Uc': round(Uc, 4),
            'cos_phi': round(cos_phi, 4)
        })

    return results


def get_voltage_at_time(file_path, target_time_str):
    """Return the feeder line-to-line voltage at the requested timestamp."""
    df = pd.read_excel(file_path)
    expected_columns = [
        'timestamp', 'feeder_active_power', 'feeder_reactive_power',
        'feeder_current', 'bus_line_voltage_ab'
    ]
    if list(df.columns) != expected_columns:
        raise ValueError("Unexpected column schema in the feeder workbook.")
    target_time = pd.to_datetime(target_time_str)
    timestamps = pd.to_datetime(df['timestamp'], errors='coerce')
    match = df[timestamps == target_time]
    if match.empty:
        raise ValueError(f"No feeder record found at {target_time_str}.")
    voltage = pd.to_numeric(match['bus_line_voltage_ab'].iloc[0], errors='coerce')
    if pd.isna(voltage):
        raise ValueError(f"Non-numeric feeder voltage at {target_time_str}.")
    return float(voltage)

def DScase_3phase_train(
        casedata, model_type='pretrainnet', plot_flag=True, total_samples=100,
        noise_range=(-0.05, 0.05), batch_size=5, device='cpu', save_artifacts=True,
        result_root=None):
    baseMVA = casedata['baseMVA']
    bus_data = casedata['bus']  # 每行: [bus_i, type, Pd, Qd, ...]
    branch_data = casedata['branch']  # 每行: [from, to, ..., rateA, angle_min, angle_max]
    # 预先计算的支路阻抗矩阵列表
    # case36_3phase 中应已提供：branch_R_mat, branch_X_mat
    branch_R_mat = casedata.get('branch_R_mat')
    branch_X_mat = casedata.get('branch_X_mat')
    node_flex_dict = casedata.get('node_flex_dict')
    bus_P = casedata.get('bus_P')
    bus_Q = casedata.get('bus_Q')
    # 相相索引映射
    phase_list = ['a', 'b', 'c']
    phase_dict = {'a': 0, 'b': 1, 'c': 2}
    ph_idx = {ph: i for i, ph in enumerate(phase_list)}

    # 节点和支路数
    bus_ids = [int(row[0]) for row in bus_data]
    line_ids = list(range(len(branch_data)))

    # Pyomo 模型
    model = pyo.ConcreteModel()
    model.BUS = pyo.Set(initialize=bus_ids)
    model.LINE = pyo.Set(initialize=line_ids)
    model.PH = pyo.Set(initialize=phase_list)

    # 支路起止索引映射
    model.from_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 0]) for l in model.LINE}, within=model.BUS, mutable=False)
    model.to_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 1]) for l in model.LINE}, within=model.BUS, mutable=False)

    # 支路阻抗参数: R[(l,phi,psi)], X[(l,phi,psi)]
    def R_init(model, l, ph, ps):
        return branch_R_mat[l][ph_idx[ph], ph_idx[ps]]

    def X_init(model, l, ph, ps):
        return branch_X_mat[l][ph_idx[ph], ph_idx[ps]]

    model.R = pyo.Param(model.LINE, model.PH, model.PH, initialize=R_init, mutable=False)
    model.X = pyo.Param(model.LINE, model.PH, model.PH, initialize=X_init, mutable=False)

    # 负荷注入（正为注入至网络，Pd,Qd 为负载取负）
    def Pd_init(model, i, ph):
        idx = bus_ids.index(i)
        # print(bus_P[idx, phase_dict[ph]])
        return bus_P[idx, ph_idx[ph]] / baseMVA

    def Qd_init(model, i, ph):
        idx = bus_ids.index(i)
        return bus_Q[idx, ph_idx[ph]] / baseMVA

    model.Pd = pyo.Param(model.BUS, model.PH, initialize=Pd_init, mutable=False)
    model.Qd = pyo.Param(model.BUS, model.PH, initialize=Qd_init, mutable=False)

    model.V_root = pyo.Param(initialize=1.0, mutable=True)
    # 变量定义
    model.V2 = pyo.Var(model.BUS, model.PH, within=pyo.NonNegativeReals)  # 节点电压平方
    model.Pf = pyo.Var(model.LINE, model.PH, within=pyo.Reals)  # 支路有功功率流
    model.Qf = pyo.Var(model.LINE, model.PH, within=pyo.Reals)  # 支路无功功率流
    model.I2 = pyo.Var(model.LINE, model.PH, within=pyo.NonNegativeReals)  # 支路电流平方

    # model.P_total = pyo.Var(model.PH, within=pyo.Reals)  # 根节点有功功率流
    # model.Q_total = pyo.Var(model.PH, within=pyo.Reals)  # 根节点无功功率流
    model.Pn = pyo.Var(model.BUS, model.PH, within=pyo.Reals)  # 节点有功注入
    model.Qn = pyo.Var(model.BUS, model.PH, within=pyo.Reals)  # 节点无功注入

    model.var_proj = pyo.Var(range(2), within=pyo.Reals)  # 聚合功率变量

    # 电压上下限 (pu^2)
    Vmin = 0.9 ** 2
    Vmax = 1.1 ** 2

    # 约束
    model.constraints = pyo.ConstraintList()

    # 电压边界约束
    for i in model.BUS:
        for ph in model.PH:
            model.constraints.add(expr=model.V2[i, ph] >= Vmin)
            model.constraints.add(expr=model.V2[i, ph] <= Vmax)

    # 电压下降方程（简化形式）
    for l in model.LINE:
        i = model.from_bus[l]
        j = model.to_bus[l]
        for ph in model.PH:
            # 2*sum_phi' (R_ij^ph,ph' * P_ij^ph' + X_ij^ph,ph' * Q_ij^ph') + sum_phi' |Z_ij^ph,ph'|^2 * I2
            lin_loss = sum(2 * (model.R[l, ph, ps] * model.Pf[l, ps] + model.X[l, ph, ps] * model.Qf[l, ps])
                           for ps in model.PH)
            quad_loss = sum((model.R[l, ph, ps] ** 2 + model.X[l, ph, ps] ** 2) * model.I2[l, ps] for ps in model.PH)
            model.constraints.add(
                expr=model.V2[j, ph] == model.V2[i, ph] - lin_loss + quad_loss
                # expr = (model.V2[j, ph] == model.V2[i, ph] - lin_loss)
            )

    # 电流-功率确切关系
    for l in model.LINE:
        i = model.from_bus[l]
        for ph in model.PH:
            # 等式转化为不等式有数值问题
            # model.constraints.add(
            #     expr=model.I2[l, ph] * model.V2[i, ph] >= model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            # )
            # model.constraints.add(
            #     expr=model.I2[l, ph] * model.V2[i, ph] <= model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            # )
            model.constraints.add(
                expr=model.I2[l, ph] * model.V2[i, ph] == model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            )

    # 功率平衡约束
    for n in model.BUS:
        if not n == 1:
            for ph in model.PH:
                inflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.to_bus[l] == n)
                loss_P = sum(sum(model.R[l, ph, ps] * model.I2[l, ps]
                                 for ps in model.PH)
                             for l in model.LINE if model.to_bus[l] == n)
                outflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.from_bus[l] == n)
                model.constraints.add(expr=(inflow_P - loss_P - outflow_P + model.Pn[n, ph] == 0.0))

                inflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.to_bus[l] == n)
                loss_Q = sum(sum(model.X[l, ph, ps] * model.I2[l, ps]
                                 for ps in model.PH)
                             for l in model.LINE if model.to_bus[l] == n)
                outflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.from_bus[l] == n)
                model.constraints.add(expr=(inflow_Q - loss_Q - outflow_Q + model.Qn[n, ph] == 0.0))

                node_flex_info = node_flex_dict.get(n, 0)
                if not node_flex_info['type']:
                    model.constraints.add(expr=(model.Pn[n, ph] == -model.Pd[n, ph]))
                    model.constraints.add(expr=(model.Qn[n, ph] == -model.Qd[n, ph]))
                elif node_flex_info['type'] == 1:
                    model.constraints.add(expr=(model.Pn[n, ph] <= -model.Pd[n, ph] + node_flex_info['rate'] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Pn[n, ph] >= -model.Pd[n, ph] - node_flex_info['rate'] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] == -model.Qd[n, ph]))
                elif node_flex_info['type'] == 2:
                    model.constraints.add(expr=(model.Pn[n, ph] <= -model.Pd[n, ph] + node_flex_info['rate'][0] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Pn[n, ph] >= -model.Pd[n, ph] - node_flex_info['rate'][0] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] <= -model.Qd[n, ph] + node_flex_info['rate'][1] * abs(model.Qd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] >= -model.Qd[n, ph] - node_flex_info['rate'][1] * abs(model.Qd[n, ph])))
    # 平衡节点（母线）功率和电压设置，假设 bus 1 为参考
    for ph in model.PH:
        outflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.from_bus[l] == 1)
        outflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.from_bus[l] == 1)
        model.constraints.add(expr=(model.Pn[1, ph] - outflow_P == model.Pd[1, ph]))
        model.constraints.add(expr=(model.Qn[1, ph] - outflow_Q == model.Qd[1, ph]))
        model.constraints.add(expr=model.V2[bus_ids[0], ph] == model.V_root ** 2)

    model.constraints.add(expr=(model.var_proj[0] == sum(model.Pn[1, ph] for ph in model.PH)))
    model.constraints.add(expr=(model.var_proj[1] == sum(model.Qn[1, ph] for ph in model.PH)))

    class CaseData(Dataset):
        def __init__(self, size=total_samples):
            self.size = size
            self.noise_range = noise_range

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            return {'V_root':torch.rand(1, device=device) * (noise_range[1]-noise_range[0]) +noise_range[0]}

    dim = 2
    A_hat = np.vstack([
        np.eye(dim),
        -np.eye(dim),
        [1,1],
        [-1,-1],
        [1, -1],
        [-1, 1],
    ])
    errorcalculator = ErrorCalculator(
        original_model={'model': model},
        A_hat=A_hat,
        solver='ipopt',
    )

    case_name = casedata['casename']
    result_root = PROJECT_ROOT / 'results' if result_root is None else Path(result_root)
    result_folder = result_root / 'ds_proj' / case_name
    figure_folder = result_folder / 'figures'
    if save_artifacts:
        figure_folder.mkdir(parents=True, exist_ok=True)


    n_train = 500

    if plot_flag and save_artifacts:
        plt.figure(figsize=(8, 6))
        total_P = sum(sum(bus_P))
        total_Q = sum(sum(bus_Q))
        xlim = np.array([total_P - 0.5*abs(total_P),total_P + 0.5*abs(total_P)])/ baseMVA
        ylim = np.array([total_Q - 0.5*abs(total_Q),total_Q + 0.5*abs(total_Q)])/ baseMVA
        plotter = ShapeDrawer_2D()
        plotter.plot_polygon(errorcalculator.A_hat, errorcalculator.b_hat,
                             facecolor='green', xlim=xlim, ylim=ylim,
                             label=f'Approximation',
                             title=f'Training step = {0}'
                             )

        pretrain_folder = figure_folder / 'pretrain_process'
        pretrain_folder.mkdir(parents=True, exist_ok=True)
        plotter.save(pretrain_folder / f'step0{0}.png')
    def training_callback(errorcalculator, epoch):
        len_his = len(errorcalculator.training_history['feas'])
        print(f"Iter {epoch}: FeasErr={np.mean(errorcalculator.training_history['feas'][-min(10, len_his):]):.2e}, "
              f"OptErr={np.mean(errorcalculator.training_history['opt'][-min(10, len_his):]):.2e}")
        # print(errorcalculator.b_hat)
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
            "lr": 2e-1,
            "batch_size": 1,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.98},
            "n_cal": 5,
            "cal_feas": True,
            "cal_opt": True,
            'feas_tol': 1e-10,
            'opt_tol': 1e-10,
            "rate_opt_feas": 0.6
        }
    else:
        trainer_configure = {
            "call_interval": 1,
            "training_callback": training_callback,
            "optimizer": "adam",
            # "optimizer": "sgd",
            "lr": 4e-4,
            "batch_size": batch_size,
            "scheduler": {"type": "StepLR", "step_size": 100, "gamma": 0.95},
            "n_cal": 2,
            "cal_feas": True,
            "cal_opt": True,
            'feas_tol': 1e-10,
            'opt_tol': 1e-10,
            "rate_opt_feas": 0.6,
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
        'params': params,
        'result_path': result_folder / f'{model_type}_weights.pth',
        'n_train': n_train,
        'metadata': {
            'dscasedata': casedata,
        }
    }

def DScase_3phase(model, casedata):
    baseMVA = casedata['baseMVA']
    bus_data = casedata['bus']  # 每行: [bus_i, type, Pd, Qd, ...]
    branch_data = casedata['branch']  # 每行: [from, to, ..., rateA, angle_min, angle_max]
    # 预先计算的支路阻抗矩阵列表
    # case36_3phase 中应已提供：branch_R_mat, branch_X_mat
    branch_R_mat = casedata.get('branch_R_mat')
    branch_X_mat = casedata.get('branch_X_mat')
    node_flex_dict = casedata.get('node_flex_dict')
    bus_P = casedata.get('bus_P')
    bus_Q = casedata.get('bus_Q')
    # 相相索引映射
    phase_list = ['a', 'b', 'c']
    phase_dict = {'a':0, 'b':1, 'c':2}
    ph_idx = {ph: i for i, ph in enumerate(phase_list)}

    # 节点和支路数
    bus_ids = [int(row[0]) for row in bus_data]
    line_ids = list(range(len(branch_data)))

    # Pyomo 模型
    # model = pyo.ConcreteModel() #模型是输入的
    model.BUS = pyo.Set(initialize=bus_ids)
    model.LINE = pyo.Set(initialize=line_ids)
    model.PH = pyo.Set(initialize=phase_list)

    # 支路起止索引映射
    model.from_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 0]) for l in model.LINE}, within=model.BUS)
    model.to_bus = pyo.Param(model.LINE, initialize={l: int(branch_data[l, 1]) for l in model.LINE}, within=model.BUS)


    # 支路阻抗参数: R[(l,phi,psi)], X[(l,phi,psi)]
    def R_init(model, l, ph, ps):
        return branch_R_mat[l][ph_idx[ph], ph_idx[ps]]

    def X_init(model, l, ph, ps):
        return branch_X_mat[l][ph_idx[ph], ph_idx[ps]]


    model.R = pyo.Param(model.LINE, model.PH, model.PH, initialize=R_init, mutable=False)
    model.X = pyo.Param(model.LINE, model.PH, model.PH, initialize=X_init, mutable=False)



    # 负荷注入（正为注入至网络，Pd,Qd 为负载取负）
    def Pd_init(model, i, ph):
        idx = bus_ids.index(i)
        # print(bus_P[idx, phase_dict[ph]])
        return bus_P[idx, ph_idx[ph]] / baseMVA


    def Qd_init(model, i, ph):
        idx = bus_ids.index(i)
        return bus_Q[idx, ph_idx[ph]] / baseMVA


    model.Pd = pyo.Param(model.BUS, model.PH, initialize=Pd_init, mutable=False)
    model.Qd = pyo.Param(model.BUS, model.PH, initialize=Qd_init, mutable=False)

    # 变量定义
    model.V2 = pyo.Var(model.BUS, model.PH, within=pyo.NonNegativeReals)  # 节点电压平方
    model.Pf = pyo.Var(model.LINE, model.PH, within=pyo.Reals)  # 支路有功功率流
    model.Qf = pyo.Var(model.LINE, model.PH, within=pyo.Reals)  # 支路无功功率流
    model.I2 = pyo.Var(model.LINE, model.PH, within=pyo.NonNegativeReals)  # 支路电流平方

    # model.P_total = pyo.Var(model.PH, within=pyo.Reals)  # 根节点有功功率流
    # model.Q_total = pyo.Var(model.PH, within=pyo.Reals)  # 根节点无功功率流
    model.Pn = pyo.Var(model.BUS, model.PH, within=pyo.Reals)  # 节点有功注入
    model.Qn = pyo.Var(model.BUS, model.PH, within=pyo.Reals)  # 节点无功注入

    model.var_proj = pyo.Var(range(2),within = pyo.Reals)  # 聚合功率变量

    # 电压上下限 (pu^2)
    Vmin = 0.90 ** 2
    Vmax = 1.10 ** 2

    # 约束
    model.constraints = pyo.ConstraintList()

    # 电压边界约束
    for i in model.BUS:
        for ph in model.PH:
            model.constraints.add(expr = model.V2[i, ph] >= Vmin)
            model.constraints.add(expr = model.V2[i, ph] <= Vmax)

    # 电压下降方程（简化形式）
    for l in model.LINE:
        i = model.from_bus[l]
        j = model.to_bus[l]
        for ph in model.PH:
            # 2*sum_phi' (R_ij^ph,ph' * P_ij^ph' + X_ij^ph,ph' * Q_ij^ph') + sum_phi' |Z_ij^ph,ph'|^2 * I2
            lin_loss = sum(2 * (model.R[l, ph, ps] * model.Pf[l, ps] + model.X[l, ph, ps] * model.Qf[l, ps])
                           for ps in model.PH)
            quad_loss = sum((model.R[l, ph, ps] ** 2 + model.X[l, ph, ps] ** 2) * model.I2[l, ps] for ps in model.PH)
            model.constraints.add(
                expr = model.V2[j, ph] == model.V2[i, ph] - lin_loss + quad_loss
                # expr = (model.V2[j, ph] == model.V2[i, ph] - lin_loss)
            )

    # 电流-功率确切关系
    for l in model.LINE:
        i = model.from_bus[l]
        for ph in model.PH:
            # model.constraints.add(
            #     expr = model.I2[l, ph] * model.V2[i, ph] >= model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            # )
            # model.constraints.add(
            #     expr = model.I2[l, ph] * model.V2[i, ph] <= model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            # )
            model.constraints.add(
                expr = model.I2[l, ph] * model.V2[i, ph] == model.Pf[l, ph] ** 2 + model.Qf[l, ph] ** 2
            )
    # 功率平衡约束
    for n in model.BUS:
        if not n == 1:
            for ph in model.PH:
                inflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.to_bus[l] == n)
                loss_P = sum(sum(model.R[l, ph, ps] * model.I2[l, ps]
                                 for ps in model.PH)
                             for l in model.LINE if model.to_bus[l] == n)
                outflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.from_bus[l] == n)
                model.constraints.add(expr=(inflow_P - loss_P - outflow_P + model.Pn[n,ph] == 0.0))

                inflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.to_bus[l] == n)
                loss_Q = sum(sum(model.X[l, ph, ps] * model.I2[l, ps]
                                 for ps in model.PH)
                             for l in model.LINE if model.to_bus[l] == n)
                outflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.from_bus[l] == n)
                model.constraints.add(expr = (inflow_Q - loss_Q - outflow_Q + model.Qn[n,ph] == 0.0))

                node_flex_info = node_flex_dict.get(n, 0)
                if not node_flex_info['type']:
                    model.constraints.add(expr=(model.Pn[n, ph] == -model.Pd[n, ph]))
                    model.constraints.add(expr=(model.Qn[n, ph] == -model.Qd[n, ph]))
                elif node_flex_info['type'] == 1:
                    model.constraints.add(expr=(model.Pn[n, ph] <= -model.Pd[n, ph] + node_flex_info['rate'] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Pn[n, ph] >= -model.Pd[n, ph] - node_flex_info['rate'] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] == -model.Qd[n, ph]))
                elif node_flex_info['type'] == 2:
                    model.constraints.add(expr=(model.Pn[n, ph] <= -model.Pd[n, ph] + node_flex_info['rate'][0] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Pn[n, ph] >= -model.Pd[n, ph] - node_flex_info['rate'][0] * abs(model.Pd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] <= -model.Qd[n, ph] + node_flex_info['rate'][1] * abs(model.Qd[n, ph])))
                    model.constraints.add(expr=(model.Qn[n, ph] >= -model.Qd[n, ph] - node_flex_info['rate'][1] * abs(model.Qd[n, ph])))
    # 平衡节点（母线）功率和电压设置，假设 bus 1 为参考
    for ph in model.PH:
        outflow_P = sum(model.Pf[l, ph] for l in model.LINE if model.from_bus[l] == 1)
        outflow_Q = sum(model.Qf[l, ph] for l in model.LINE if model.from_bus[l] == 1)
        model.constraints.add(expr=( model.Pn[1, ph] - outflow_P == model.Pd[1, ph]))
        model.constraints.add(expr=( model.Qn[1, ph] - outflow_Q == model.Qd[1, ph]))

    model.constraints.add(expr=(model.var_proj[0] == sum(model.Pn[1, ph] for ph in model.PH)))
    model.constraints.add(expr=(model.var_proj[1] == sum(model.Qn[1, ph] for ph in model.PH)))

def disagg_DS_3phase(P_target, Q_target, v_target, dscasedata):
    model = pyo.ConcreteModel()
    DScase_3phase(model, dscasedata)
    for ph in model.PH:
        model.constraints.add(v_target ** 2 == model.V2[dscasedata['bus'][0,0], ph]) # 0 号节点为根节点
    model.obj = pyo.Objective(
        expr=(P_target-model.var_proj[0]) ** 2 + (Q_target-model.var_proj[1]) ** 2,
        sense=pyo.minimize
    )
    solver = pyo.SolverFactory('ipopt')
    solver.solve(model, tee=True)

    return model.obj()

if __name__ == "__main__":
    print(
        "Use `python -m Simulator.runners.main_ds_3phase --help` for the "
        "anonymized three-phase workflow."
    )
