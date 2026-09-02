import os
import numpy as np
import pandas as pd

from Simulator.Plotter import ErrorVisualizer
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
from Simulator.cases.aggregation_case import Aggregator
from Simulator import  PROJECT_ROOT
# result_path = f'{PROJECT_ROOT}\\results\\aggregation\\discrete\\'
result_path = f'{PROJECT_ROOT}\\results\\aggregation\\'
is_discrete = False
import pickle

# with open(result_path+'error_history.pkl', 'rb') as f:  # 'rb' 表示二进制读取
#     res_dict= pickle.load(f)
# res_dict['error_feas'] = res_dict['error_feas'][::2]
# res_dict['error_opt'] = res_dict['error_opt'][::2]
# visualizer = ErrorVisualizer()
# visualizer.error_history = res_dict
# visualizer.error_history['iterations'] = list(range(0,801,100))
#
# visualizer.plot_dual_violin(save_path=f'{result_path}/errors_violin.svg', is_discrete=is_discrete)

# with open(result_path+'error_history_cube.pkl', 'rb') as f:  # 'rb' 表示二进制读取
#     res_dict_cube= pickle.load(f)
# error_cube = res_dict_cube['error_opt'][-1]
# with open(result_path+'error_history.pkl', 'rb') as f:  # 'rb' 表示二进制读取
#     res_dict_apa= pickle.load(f)
# error_apa = res_dict_apa['error_opt'][-1]
# visualizer = ErrorVisualizer()
# visualizer.plot_error_vs_constraints(cube_error=error_cube, apa_error=error_apa, save_path=f'{result_path}/errors_diff_method.svg', background=False)

visualizer = ErrorVisualizer()
visualizer.plot_count_discrete( save_path=f'{result_path}/errors_diff_method.svg')
# visualizer.plot_constraint_bar( save_path=f'{result_path}/constraint_bar_diff_method.svg')


