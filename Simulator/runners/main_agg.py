import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.aggregation_case import Aggregator
from Simulator import  PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model_type = 'pretrainnet'
agg = Aggregator(seed=0, discrete_rate = 0.1)

# 生成5个电动汽车
agg.gen_EV(60)
agg.gen_TCL(40)
agg.gen_ESS(5)

# agg = Aggregator(seed=0, discrete_rate = 0.0)
#
# # 生成5个电动汽车
# agg.gen_EV(600)
# agg.gen_TCL(400)
case = agg.case_aggregator(model_type=model_type,)

if model_type=='pretrainnet':
    n_train = case.get('n_train',1000)
    model   = PreTrainNet(case['A_hat'],case['b_hat'],device = device).to(device)
else:
    n_train = case.get('n_train',1000)
    model = PreTrainNet(case['A_hat'], case['b_hat'])
    model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\{case['casename']}\\pretrainnet_weights.pth'))
    A_pretrained, b_pretrained = model()
    b_pretrained = b_pretrained[0].detach().cpu().numpy()
    if model_type == 'biasnet':
        A_pretrained = A_pretrained[0].detach().to(device)
        case['trainer_configure'].update(A_pretrained = A_pretrained)
        model = BiasNet(dim_theta=case['params']['count'], b_init=b_pretrained,device = device).to(device)
    elif model_type == 'fullnet':
        A_pretrained = A_pretrained[0].detach().cpu().numpy()
        model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,device = device).to(device)

trainer = Trainer(
    model=model,
    error_calculator=case['errorcalculator'],
    compute_loss=compute_loss,
)

import time
training_start = time.time()
trainer.configure(**case['trainer_configure'])
trainer.configure(
    #feas_tol = 1e-1,
                  rate_opt_feas = 1.0,
                  lr = 1e-2,
)
trainer.initialize()
trainer.train(n_train = n_train , params_data = case['params'])

trainer.configure(**case['trainer_configure'])
trainer.configure(
    #feas_tol = 1e-1,
                  rate_opt_feas = 0.1,
                  lr = 1e-2,
)
trainer.initialize()
trainer.train(n_train = 200 , params_data = case['params'])

trainer.configure(**case['trainer_configure'])
trainer.configure(
    #feas_tol = 1e-1,
                  rate_opt_feas = 0.0001,
                  lr = 1e-2,
)
trainer.initialize()
trainer.train(n_train = 100 , params_data = case['params'])

training_end = time.time()
print('总耗时',training_end-training_start)
torch.save(model.state_dict(), case['result_path']+'disc')
