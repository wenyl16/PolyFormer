import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
import matplotlib.pyplot as plt
from Simulator.Plotter import ShapeDrawer_2D
import os
import torch
from Simulator.cases.basic_cases import case_ellipse
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type = 'pretrainnet'
case = case_ellipse(model_type = model_type)
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

dtheta = [0.0,0.0]
xlim = [-5,5]
ylim = [-5,5]
Sigma = np.array([
    [5 / 2+dtheta[0], -4.8 / 2+dtheta[1]],
    [-4.8 / 2+dtheta[1], 5 / 2+dtheta[0]]
])

figure_folder_name = f'{PROJECT_ROOT}\\results\\{case['casename']}\\figures\\comparison\\'
os.makedirs(figure_folder_name, exist_ok=True)

plotter.plot_ellipse(Sigma, xlim=xlim, ylim=ylim,edgecolor='red',facecolor='blue', label='Original region')
plotter.save('1.png')
plotter.plot_polygon(A_pretrained, b_pretrained,
                     facecolor='green', xlim=xlim, ylim=ylim,
                     label=f'pretrained',
                     # title=f'Training step = {0}',
                     )
plotter.save(figure_folder_name+f'pretrain dtheta = [{dtheta[0]:.2e},{dtheta[1]:.2e}].png')
plotter.remove_shape(plotter.shapes[-1]['id'])

b_pred = biasnet(torch.tensor(dtheta))
b_pred = b_pred.detach().cpu().numpy()
plotter.plot_polygon(A_pretrained, b_pred,
                     facecolor='yellow', xlim=xlim, ylim=ylim,
                     label=f'biasnet',
                     # title=f'Training step = {0}',
                     )
plotter.save(figure_folder_name+f'biasnet dtheta = [{dtheta[0]:.2e},{dtheta[1]:.2e}].png')
plotter.remove_shape(plotter.shapes[-1]['id'])

A_pred, b_pred = fullnet(torch.tensor([dtheta,dtheta]))
A_pred = A_pred[0].detach().cpu().numpy()
b_pred = b_pred[0].detach().cpu().numpy()

plotter.plot_polygon(A_pred, b_pred,
                     facecolor='red', xlim=xlim, ylim=ylim,
                     label=f'fullnet',
                     # title=f'Training step = {0}',
                     )
plotter.save(figure_folder_name+f'fullnet dtheta = [{dtheta[0]:.2e},{dtheta[1]:.2e}].png')


