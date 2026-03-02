import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.MPC_case import MPCcase
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parallel= False
model_type = 'pretrainnet'
mpc = MPCcase()
is_epigraph = False
case = mpc.build_simplecase_fr(model_type=model_type)

if model_type=='pretrainnet':
    n_train = 500
    model   = PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=is_epigraph, device = device)
else:
    n_train = 20
    model = PreTrainNet(case['A_hat'], case['b_hat'],is_epigraph=is_epigraph,device=device)
    model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth',map_location=device))
    A_pretrained, b_pretrained = model()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()
    if model_type == 'biasnet':
        A_pretrained = A_pretrained[0].detach().to(device)
        case['trainer_configure'].update(A_pretrained = A_pretrained)
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,n_hidden=128,device = device).to(device)
    elif model_type == 'fullnet':
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,is_epigraph=is_epigraph, n_hidden=128,device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)

trainer.configure(**case['trainer_configure'])
# trainer.configure(theta_init = case['theta_init'])
trainer.initialize()
trainer.train(n_train = n_train , params_data=case['params'],parallel=parallel)

trainer.configure(rate_opt_feas = 1e-3)
trainer.train(n_train = n_train , params_data=case['params'],parallel=parallel)

torch.save(model.state_dict(), case['result_path'])
