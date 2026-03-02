import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
import Simulator.cases.DS_case_3phase as DS_case_3phase
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = 'fullnet'
ppc = DS_case_3phase.case36real_3phase_ds()
print(sum(ppc['bus'][:,2])/ppc['baseMVA'])
case = DS_case_3phase.DScase_3phase_train(casedata=ppc, model_type = model_type, noise_range=[-0.04,0.04],device=device)
parallel = True
P_rated = sum(ppc['bus'][:,2])/ppc['baseMVA']
# lr = 2e-1 #这是pretrainnet
lr = 5e-5/P_rated
rate_opt_feas = 0.6

if model_type=='pretrainnet':
    n_train = 500
    model  = PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=False, device = device)
else:
    n_train = 20
    model = PreTrainNet(case['A_hat'], case['b_hat'],is_epigraph=False,device=device)
    model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\ds_proj\\{case['casename']}\\pretrainnet_weights.pth',map_location=device))
    A_pretrained, b_pretrained = model()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()
    if model_type == 'biasnet':
        A_pretrained = A_pretrained[0].detach().to(device)
        case['trainer_configure'].update(A_pretrained = A_pretrained)
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,n_hidden=128,device = device).to(device)
    elif model_type == 'fullnet':
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,is_epigraph=False,n_hidden=128,device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)

trainer.configure(**case['trainer_configure'])
trainer.configure(lr=lr)
trainer.configure(rate_opt_feas=rate_opt_feas)
trainer.initialize()
trainer.train(n_train=n_train * 4, params_data=case['params'], parallel=parallel)
torch.save(model.state_dict(), case['result_path'])