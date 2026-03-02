import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
import matplotlib.pyplot as plt
from Simulator.Plotter import ShapeDrawer_2D
import torch
from Simulator.cases.basic_cases import case_polygon
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type = 'pretrainnet'
case = case_polygon(model_type = model_type)
dim_theta = 2
pretrainnet =  PreTrainNet(case['A_hat'],case['b_hat'])
pretrainnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth', map_location=device))
A_pretrained,b_pretrained = pretrainnet()
A_pretrained = A_pretrained[0].detach().cpu().numpy()
b_pretrained = b_pretrained[0].detach().cpu().numpy()
biasnet = BiasNet(dim_theta = dim_theta, b_init = b_pretrained)
biasnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\biasnet_weights.pth', map_location=device))
fullnet = FullNet(dim_theta = dim_theta, A_init=A_pretrained,b_init = b_pretrained)
fullnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\fullnet_weights.pth', map_location=device))
plt.figure(figsize=(8, 6))
plotter = ShapeDrawer_2D()


dtheta = [-0.1,0.1]
xlim = [-0.5, 1.5]
ylim = [-0.5, 1.5]
b = np.array([1, 1, 0, 0, 1.5, -0.5, 0.7+dtheta[0] , 0.7+dtheta[1] ])

figure_folder_name = f'{PROJECT_ROOT}\\results\\{case['casename']}\\figures\\comparison\\'
os.makedirs(figure_folder_name, exist_ok=True)

facecolor = 'blue'
edgecolor = 'blue'
plotter.plot_polygon(case['metadata']['A_init'], b, edgecolor=edgecolor,
                     facecolor=facecolor, xlim=xlim, ylim=ylim,
                     label=f'Original region',  # 透明度
                     # title=f'Training step = {0}',
                     )
A_pred, b_pred = fullnet(torch.tensor(dtheta))
A_pred = A_pred[0].detach().cpu().numpy()
b_pred = b_pred[0].detach().cpu().numpy()

facecolor = 'red'
edgecolor = 'red'
plotter.plot_polygon(A_pred, b_pred,edgecolor=edgecolor,
                     facecolor=facecolor, xlim=xlim, ylim=ylim,
                     label=f'fullnet',
                     # title=f'Training step = {0}',
                     )
plotter.save(figure_folder_name+f'fullnet dtheta = {dtheta[0]:.2e}, {dtheta[1]:.2e}.svg', show_legend=False)


