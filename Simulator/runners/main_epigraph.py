import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.basic_cases import case_epigraph
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parallel= True
model_type = 'pretrainnet'
case = case_epigraph(model_type = model_type,device=device)
if model_type=='pretrainnet':
    n_train = 500
    model   = PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=True, device = device)
else:
    n_train = 20
    model = PreTrainNet(case['A_hat'], case['b_hat'],is_epigraph=True,device=device)
    model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth',map_location=device))
    A_pretrained, b_pretrained = model()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()
    if model_type == 'biasnet':
        A_pretrained = A_pretrained[0].detach().to(device)
        case['trainer_configure'].update(A_pretrained = A_pretrained)
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,n_hidden=128,device = device).to(device)
    elif model_type == 'fullnet':
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,is_epigraph=True,n_hidden=128,device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)

trainer.configure(**case['trainer_configure'])
# trainer.configure(theta_init = case['theta_init'])
trainer.initialize()
trainer.train(n_train = n_train , params_data=case['params'],parallel=parallel)
# torch.save(model.state_dict(), case['result_path'])

# trainer.configure(feas_tol = 1e-4,opt_tol = 2e-2,lr = case['trainer_configure']['lr']/50)
# trainer.initialize()
# trainer.train(n_train = 3 , dataloader = case['dataloader'],parallel=parallel)
#
# torch.save(model.state_dict(), case['result_path']+'feas')
