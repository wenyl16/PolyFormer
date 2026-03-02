import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from Simulator.Approximator import PreTrainNet,BiasNet,FullNet,compute_loss,Trainer
import torch
import Simulator.cases.TD_case as TD_case
from Simulator import PROJECT_ROOT
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_type = 'fullnet'
parallel= False

dscases = {
    # 'case10ba_ds': TD_case.case10ba_ds(),
    # 'case17me_ds': TD_case.case17me_ds(),
    # 'case33bw_ds': TD_case.case33bw_ds(),
    'case51ga_ds': TD_case.case51ga_ds(),
    'case74_ds': TD_case.case74_ds(),
    'case118zh_ds': TD_case.case118zh_ds(),
    'case136ma_ds': TD_case.case136ma_ds(),
    'case533mt_hi_ds': TD_case.case533mt_hi_ds()
}

for casename, ppc in dscases.items():
    P_rated = sum(ppc['bus'][:,2])/ppc['baseMVA']
    # lr = 1e-1/P_rated
    lr = 2e-5/P_rated#full net
    rate_opt_feas = 1.0
    # case = TD_case.DScase_train(casedata=ppc, model_type = model_type, device=device, plot_flag=True, noise_range=[-0.04,0.04]) #, noise_range=[-0.02,0.02]
    case = TD_case.DScase_train(casedata=ppc, model_type = model_type, device=device, plot_flag=False, noise_range=[-0.05,0.05])

    if model_type=='pretrainnet':
        n_train = 500
        model   = PreTrainNet(case['A_hat'],case['b_hat'],is_epigraph=False, device = device)
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
            model= FullNet(dim_theta = case['params']['count'], A_init=A_pretrained,b_init = b_pretrained,n_hidden=128,device = device).to(device)

    trainer = Trainer(
        model=model,
        error_calculator=case['errorcalculator'],
        compute_loss=compute_loss,
    )

    trainer.configure(**case['trainer_configure'])
    trainer.configure(lr = lr)
    trainer.configure(rate_opt_feas = rate_opt_feas)
    trainer.initialize()
    trainer.train(n_train = n_train*2 , params_data=case['params'], parallel= parallel)
    trainer.configure(rate_opt_feas = 1e-3, lr = lr/2)
    trainer.initialize()
    trainer.train(n_train = 10 , params_data=case['params'], parallel= parallel)

    torch.save(model.state_dict(), case['result_path'])