"""Train PreTrainNet or FullNet for the microgrid safe-region case."""

import argparse
import os
import pickle
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch

from Simulator import PROJECT_ROOT
from Simulator.Approximator import PreTrainNet, FullNet, compute_loss, Trainer
from Simulator.cases.safe_region_case import safe_region_case


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model-type', choices=('pretrainnet', 'fullnet'), default='fullnet',
        help='Training mode (default: fullnet, matching the original main runner).',
    )
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    model_type = args.model_type
    device = torch.device(
        ('cuda' if torch.cuda.is_available() else 'cpu')
        if args.device == 'auto' else args.device
    )
    parallel = False
    is_epigraph = False

    T = 12
    Delta_t = 5 / 60
    params = {
        'num_cont_tcl': 2,
        'num_disc_tcl': 2,
        'num_ess': 2,
        'C_L': 0.0,
        'C_U': 30.0,

        # Continuous TCLs.
        'C_cont': [316.11, 140.56],
        'eta_cont': [4.0, 4.0],
        'H_cont': [3.892, 0.92092],
        'p_cont_max_tcl': [17.4, 7.69],
        'theta_min_cont': [21.0, 21.0],
        'theta_max_cont': [25.0, 25.0],
        'theta_set_cont': [23.0, 23.0],

        # Discrete TCLs.
        'C_disc': [95.5, 170.42],
        'eta_disc': [3.6, 3.6],
        'H_disc': [1.96, 2.10],
        'p_disc_max_tcl': [15.73, 10.46],
        'theta_min_disc': [21.0, 21.0],
        'theta_max_disc': [25.0, 25.0],
        'theta_set_disc': [23.0, 23.0],

        # ESSs; preserve the different lower bounds in the two original runners.
        'eta_chg': [0.97, 0.98],
        'eta_dis': [0.98, 0.97],
        'pmax_chg_ess': [25.0, 10.0],
        'pmax_dis_ess': [25.0, 10.0],
        'e_min': [0.0, 0.0] if model_type == 'pretrainnet' else [10.0, 10.0],
        'e_max': [50.0, 40.0],
    }
    with np.load(PROJECT_ROOT / 'data' / 'profiles_data' / 'profiles_data.npz') as profiles:
        data = {
            'full_theta_amb': profiles['temp_data'],
            'full_p_bl': 20 * profiles['load_data'],
            'full_p_pv': 40 * profiles['pv_data'],
        }

    sfc = safe_region_case()
    case = sfc.build_mg_case(
        T=T, Delta_t=Delta_t, data=data, params=params,
        x_only=True, model_type=model_type, device=device,
    )
    if model_type == 'pretrainnet':
        n_train = 500
        model = PreTrainNet(
            case['A_hat'], case['b_hat'], is_epigraph=is_epigraph, device=device,
        )
    else:
        n_train = 20
        # The original FullNet runner starts from the case polytope, not a checkpoint.
        model = FullNet(
            dim_theta=case['params']['count'],
            A_init=case['A_hat'], b_init=case['b_hat'],
            is_epigraph=is_epigraph, n_hidden=128, device=device,
        ).to(device)

    trainer = Trainer(
        model=model,
        error_calculator=case['errorcalculator'],
        compute_loss=compute_loss,
    )
    trainer.configure(**case['trainer_configure'])
    trainer.configure(lr=1e-1 if model_type == 'pretrainnet' else 2e-3)
    trainer.initialize()
    trainer.train(n_train=n_train, params_data=case['params'], parallel=parallel)

    if model_type == 'fullnet':
        trainer.configure(rate_opt_feas=2e-2, lr=2e-4)
        trainer.initialize()
        trainer.train(n_train=1, params_data=case['params'], parallel=parallel)

    torch.save(model.state_dict(), case['result_path'])
    if model_type == 'fullnet':
        history_path = Path(case['result_path']).parent / 'training_history.pkl'
        with open(history_path, 'wb') as stream:
            pickle.dump(case['errorcalculator'].training_history, stream)
    return trainer


if __name__ == '__main__':
    main()
