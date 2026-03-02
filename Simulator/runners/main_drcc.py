import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
from Simulator.cases.DRCC_case import DRCCModelBuilder
from Simulator import PROJECT_ROOT
import pandas as pd
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
N_var = 400
N_levels = 8
N_samples = 1280
data_path = f'{PROJECT_ROOT}/data/DRCC/r_samples_x{N_var}g{N_levels}s{N_samples}'
# N_levels = 2
r_samples = pd.read_csv(data_path+'.csv')

# params = {'r_samples':r_samples,
#           'R_min':[0.015]*N_levels,
#           'R_limits':[(-0.1,0.1)]*N_levels,
#           'eps':[0.1]*N_levels,
#           'rho':[0.0001]*N_levels,
#           'max_group_total':[1.0]*N_levels}
N_levels = len(set(r_samples['group']))
params = {'r_samples':r_samples,
          # 'R_min':[None]*N_levels,
          'R_limits':[(-0.1,0.1)]*N_levels,
          # 'eps':[None]*N_levels,
          # 'rho':[None]*N_levels,
          # 'max_group_total':[None]*N_levels
          }


# plot_flag = True if len(r_samples)==2 else False
plot_flag = False

portfolio =  DRCCModelBuilder(params)
parallel = False
model_type = 'fullnet'
# model_type='pretrainnet'
# for group_idx in range(portfolio.group_number):
for group_idx in range(2,portfolio.group_number):
    case = portfolio.build_drcc_train(portfolio.group_dataset[group_idx], model_type=model_type, plot_flag=plot_flag,device=device)
    if model_type=='pretrainnet':
        self_configure = dict(
            lr=5e-1,
            rate_opt_feas=1.0,
        )
        n_train = 500
        model  = PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=False, device = device)
    else:
        self_configure = dict(
            lr=2e-4,
            rate_opt_feas=1.,
            optimizer='adam',
        )  # full
        n_train = 30

        # self_configure_2 = dict(
        #     lr=5e-4,
        #     rate_opt_feas=1.,
        #     optimizer='adam',
        # )  # full
        # n_train_2 = 40

        # model = PreTrainNet(case['A_hat'], case['b_hat'],is_epigraph=False,device=device)
        # model.load_state_dict(torch.load(f'{PROJECT_ROOT}\\results\\DRCC\\{case['casename']}\\g{group_idx}\\pretrainnet_weights.pth',map_location=device))
        # A_pretrained, b_pretrained = model()
        # b_pretrained = b_pretrained[0].detach().cpu().numpy()
        if model_type == 'biasnet':
            A_pretrained = A_pretrained[0].detach().to(device)
            case['trainer_configure'].update(A_pretrained = A_pretrained)
            model = BiasNet(dim_theta=case['params']['count'], b_init=case['b_hat'],n_hidden=128,device = device).to(device)
        elif model_type == 'fullnet':
            # A_pretrained = A_pretrained[0].detach().cpu().numpy()
            # model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,is_epigraph=False,n_hidden=128,device = device).to(device)
            model = FullNet(dim_theta=case['params']['count'], A_init=case['A_hat'], b_init=case['b_hat'],
                            is_epigraph=False, n_hidden=128, device=device).to(device)
    trainer = Trainer(
        model=model,
        error_calculator=case['errorcalculator'],
        compute_loss=compute_loss,
    )

    trainer.configure(**case['trainer_configure'])
    trainer.configure(**self_configure)
    trainer.initialize()
    trainer.train(n_train=n_train, params_data=case['params'], parallel=parallel)

    # trainer.configure(**case['trainer_configure'])
    # trainer.configure(**self_configure_2)
    # trainer.initialize()
    # trainer.train(n_train=n_train_2, params_data=case['params'], parallel=parallel)
    torch.save(model.state_dict(), case['result_path'])