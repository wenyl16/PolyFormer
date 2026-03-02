import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
import matplotlib.pyplot as plt
from Simulator.Plotter import ShapeDrawer_2D
import os
import torch
from Simulator.cases.basic_cases import case_nonconvex
model_type = 'pretrainnet'
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
case = case_nonconvex(model_type = model_type)
dim_theta = 3
pretrainnet =  PreTrainNet(case['A_hat'],case['b_hat'])
pretrainnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth', map_location=device))
A_pretrained,b_pretrained = pretrainnet()
A_pretrained = A_pretrained[0].detach().cpu().numpy()
b_pretrained = b_pretrained[0].detach().cpu().numpy()
biasnet = BiasNet(dim_theta = dim_theta, b_init = b_pretrained)
biasnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\biasnet_weights.pth', map_location=device))
fullnet = FullNet(dim_theta = dim_theta, A_init=A_pretrained,b_init = b_pretrained)
fullnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\fullnet_weights.pth', map_location=device))
plt.figure(figsize=(6, 6))
plotter = ShapeDrawer_2D()

dtheta = [0.,0.,0.]
xlim = [-1.5, 1.5]
ylim = [-1.5, 1.5]
theta = np.array(
    [1+dtheta[0], 1+dtheta[1],1+dtheta[2],
     ])

figure_folder_name = f'{PROJECT_ROOT}\\results\\{case['casename']}\\figures\\comparison\\'
os.makedirs(figure_folder_name, exist_ok=True)

facecolor = 'blue'
edgecolor = 'blue'

plotter.plot_circle_regions(
    theta=theta,
    xlim=xlim,  # 包含两个圆的可视范围
    ylim=ylim,
    edgecolor=edgecolor ,
    facecolor=facecolor,  # 区域填充色
    alpha=0.3,  # 透明度
    label='Nonconvex region'
)
A_pred, b_pred = fullnet(torch.tensor([dtheta,dtheta]))
A_pred = A_pred[0].detach().cpu().numpy()
b_pred = b_pred[0].detach().cpu().numpy()

facecolor = 'red'
edgecolor = 'red'
plotter.plot_polygon(A_pred, b_pred,edgecolor=edgecolor ,
                     facecolor=facecolor, xlim=xlim, ylim=ylim,
                     label=f'fullnet',  # 透明度
                     # title=f'Training step = {0}',
                     )
plotter.save(figure_folder_name+f'fullnet dtheta = [{dtheta[0]:.2e},{dtheta[1]:.2e},{dtheta[2]:.2e}].svg', show_legend=False)


