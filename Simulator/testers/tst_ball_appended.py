import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet, BiasNet, FullNet
import numpy as np
from Simulator.Plotter import ShapeDrawer_2D
from Simulator.Counter import PolyBallHausdorffCalculator
import os
import torch
from Simulator.cases.basic_cases import case_ball
model_type = 'pretrainnet'
from Simulator import PROJECT_ROOT
import pickle
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# dim_list = np.hstack([2,4,6,8,10,15,20,30,40,50,60,70,80,90,100,120,140,160,180,200])
dim_list = np.hstack([50])
results = []
for dim in dim_list:
    case = case_ball(dim = dim, model_type = model_type,device=device)
    pretrainnet =  PreTrainNet(case['A_hat'],case['b_hat'])
    pretrainnet.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_appended_weights_dim{dim}.pth', map_location=device))
    A_pretrained,b_pretrained = pretrainnet()
    A_pretrained = A_pretrained[0].detach().cpu().numpy()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()
    calculator_final = PolyBallHausdorffCalculator(
        A=A_pretrained,
        b=b_pretrained,
        R=case['params']['params_dict']['R']['initial_value'],
        solver_name='ipopt',  # 推荐使用支持二阶锥的求解器
        M=1e3,
        delta_A=1e-4,
        delta_b=1e-4
    )

    calculator_final.create_model()
    # res = calculator.compute_hausdorff(sensitivity=False)
    res_final = calculator_final.compute_hausdorff(sensitivity=False)
    print(res_final)
    # results.append(res)


# np.savetxt('sensitivity_A.csv', results[0]['sensitivity_A'], delimiter=',', fmt='%.3f')

