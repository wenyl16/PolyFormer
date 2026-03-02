import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from Simulator import PROJECT_ROOT
from Simulator.Approximator import ErrorCalculator, pyomo_params_to_numpy
from Simulator.Plotter import ShapeDrawer_2D
import matplotlib.pyplot as plt
import os
import pyomo.environ as pyo

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from scipy.io import loadmat
def define_nodal_flex_P(ppc, percent = 0.5, rate = 0.3):
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
def case17me_ds(root_voltage = 1.0, radial = True, flex_percent = 0.5, flex_rate = 0.3):
    ppc = {"version": '2'}
    ppc['casename'] = 'case17me_ds'
    # ppc["baseMVA"]=10
    # ppc["basekV"] = 13.8
    data = loadmat(f'{PROJECT_ROOT}\\data\\TD_OPF\\ds_data\\case17me_ds.mat')
    ppc["baseMVA"] = data["mpc"]["baseMVA"][0, 0][0, 0]
    ppc["branch"] = data["mpc"]["branch"][0, 0]
    ppc["branch"][:, 2:4] /= 2
    if radial:
        ppc["branch"] = ppc["branch"][ppc["branch"][:,10]==1,:] #radial
    ppc["bus"] = data["mpc"]["bus"][0, 0]
    ppc["bus"][:,-2] = 1.15
    ppc["bus"][:,-1] = 0.85
    ppc["gen"] = data["mpc"]["gen"][0, 0]
    define_nodal_flex_P(ppc,percent = flex_percent, rate = flex_rate)
    ppc["root_voltage"] = root_voltage
    return ppc

case17me_ds()