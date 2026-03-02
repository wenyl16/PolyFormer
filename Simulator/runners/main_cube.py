import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.basic_cases import case_cube
from Simulator import PROJECT_ROOT
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_type ='pretrainnet'
parallel= False
dim_list = np.hstack([2,4,6,8,10,15,20,30,40,50,60,70,80,90,100,120,140,160,180,200])
# dim_list = np.hstack([6,8,10,15,20,30,40, 50])
for dim in dim_list:
    case = case_cube(dim = dim, model_type = model_type,device=device)

    model   = PreTrainNet(case['A_hat'],case['b_hat'],device = device).to(device)
    trainer = Trainer(
        model=model,
        error_calculator=case['errorcalculator'],
        compute_loss=compute_loss,
    )

    n_train = int(50*np.sqrt(dim/2))
    trainer.configure(**case['trainer_configure'])
    trainer.configure(lr=2e-2/(np.sqrt(dim/2)))
    # trainer.configure(lr_b=1.6)
    trainer.initialize()
    trainer.train(n_train=n_train, params_data=case["params"], parallel=parallel)

    n_train = int(400*np.sqrt(dim/2))
    trainer.configure(**case['trainer_configure'])
    trainer.configure(lr=3e-1/(np.sqrt(dim/2)))
    # trainer.configure(lr_b=1.6)
    trainer.initialize()
    trainer.train(n_train=n_train, params_data=case["params"], parallel=parallel)


    torch.save(model.state_dict(), case['result_path']+f'/{model_type.lower()}_weights_dim{dim}.pth')