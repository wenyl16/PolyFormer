import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
import matplotlib.pyplot as plt
from Simulator.Plotter import ShapeDrawer_2D
import os
import torch
from Simulator.cases.basic_cases import case_epigraph
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type = 'pretrainnet'
case = case_epigraph(model_type = model_type)
dim_theta = 2
pretrainnet =  PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=True)
pretrainnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth', map_location=device))
A_pretrained,b_pretrained = pretrainnet()
A_pretrained = A_pretrained[0].detach().cpu().numpy()
b_pretrained = b_pretrained[0].detach().cpu().numpy()
biasnet = BiasNet(dim_theta = dim_theta, b_init = b_pretrained,n_hidden=128)
biasnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\biasnet_weights.pth', map_location=device))
fullnet = FullNet(dim_theta = dim_theta, A_init=A_pretrained,b_init = b_pretrained,is_epigraph=True,n_hidden=128)
fullnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\fullnet_weights.pth', map_location=device))
plt.figure(figsize=(8, 6))
plotter = ShapeDrawer_2D()

dtheta = [-0.2,0.3]
xlim = [-0.5, 1.5]
ylim = [-0.5, 1.5]
theta = np.array(
    [1+dtheta[0], 1+dtheta[1],
     ])

figure_folder_name = f'{PROJECT_ROOT}\\results\\{case['casename']}\\figures\\comparison\\'
os.makedirs(figure_folder_name, exist_ok=True)

plotter.plot_epigraph(
    x_range=(0, theta[0]),
    f_min_func=lambda x: theta[1]*x ** 2,
    # facecolor='rgba(135,206,250,0.3)',
    label="Original epigraph",
    xlim=xlim,
    ylim=ylim
)
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


