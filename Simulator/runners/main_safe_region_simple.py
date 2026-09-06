import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.safe_region_case import safe_region_case
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
parallel= False
model_type = 'fullnet'
# model_type = 'pretrainnet'

sfc = safe_region_case()
is_epigraph = False
case = sfc.build_case_1(x_only=True, model_type=model_type)
# case = sfc.build_case_2(x_only=True, model_type=model_type)
if model_type=='pretrainnet':
    n_train = 500
    model   = PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=is_epigraph, device = device)
else:
    n_train = 10
    model = PreTrainNet(case['A_hat'], case['b_hat'],is_epigraph=is_epigraph,device=device)
    model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\safe_region\\{case['casename']}\\pretrainnet_weights_fr.pth',map_location=device))
    A_pretrained, b_pretrained = model()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()
    if model_type == 'biasnet':
        A_pretrained = A_pretrained[0].detach().to(device)
        case['trainer_configure'].update(A_pretrained = A_pretrained)
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,n_hidden=128,device = device).to(device)
    elif model_type == 'fullnet':
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        model= FullNet(dim_theta = case['params']['count'], A_init=case['A_hat'],b_init = case['b_hat'],is_epigraph=is_epigraph, n_hidden=128,device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)

trainer.configure(**case['trainer_configure'])
trainer.configure(
    # optimizer = 'sgd',
    lr= 1e-3
)
trainer.initialize()
trainer.train(n_train = n_train , params_data=case['params'],parallel=parallel)

trainer.configure(rate_opt_feas = 1e-1,lr = 1e-4)
trainer.initialize()
trainer.train(n_train = 1 , params_data=case['params'],parallel=parallel)

torch.save(model.state_dict(), case['result_path'])

path = f'{PROJECT_ROOT}\\results\\safe_region\\{case['casename']}\\training_history.pkl'
import pickle
with open(path, "wb") as f:
    pickle.dump(case['errorcalculator'].training_history, f)