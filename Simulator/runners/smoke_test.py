"""Fast, non-writing smoke tests for the three paper applications."""

import argparse

from Simulator.runners.main_agg import main as run_aggregation
from Simulator.runners.main_drcc import main as run_drcc


def run_distribution():
    """Check the shared T-D training components without executing main_ds.py."""
    import numpy as np
    import torch

    from Simulator import PROJECT_ROOT
    from Simulator.Approximator import FullNet, PreTrainNet, Trainer, compute_loss
    from Simulator.cases import TD_case

    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cpu')
    ppc = TD_case.case10ba_ds()
    p_rated = sum(ppc['bus'][:, 2]) / ppc['baseMVA']
    case = TD_case.DScase_train(
        casedata=ppc, model_type='fullnet', device=device,
        plot_flag=False, noise_range=(-0.04, 0.04),
        total_samples=1, batch_size=1, save_artifacts=False,
    )
    pretrained = PreTrainNet(case['A_hat'], case['b_hat'], device=device)
    weights_path = (
        PROJECT_ROOT / 'results' / 'ds_proj' / case['casename'] /
        'pretrainnet_weights.pth'
    )
    pretrained.load_state_dict(
        torch.load(weights_path, map_location=device, weights_only=True)
    )
    with torch.no_grad():
        A, b = pretrained()
    model = FullNet(
        dim_theta=case['params']['count'],
        A_init=A[0].cpu().numpy(), b_init=b[0].cpu().numpy(),
        n_hidden=128, device=device,
    ).to(device)
    trainer = Trainer(
        model=model, error_calculator=case['errorcalculator'], compute_loss=compute_loss,
    )
    trainer.configure(**case['trainer_configure'])
    trainer.configure(
        lr=2e-5 / p_rated, rate_opt_feas=1.0,
        n_cal=1, training_callback=None,
    )
    trainer.initialize()
    trainer.train(n_train=1, params_data=case['params'], parallel=False)
    if len(trainer.loss_history) != 1:
        raise RuntimeError('The T-D smoke test must complete exactly one update.')
    print(f"TD_SMOKE_OK case={case['casename']} model=fullnet updates=1")


RUNNERS = {
    'aggregation': lambda: run_aggregation(['--smoke', '--device', 'cpu']),
    'td': run_distribution,
    'drcc': lambda: run_drcc(['--smoke', '--device', 'cpu']),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--case', choices=(*RUNNERS, 'all'), default='all',
        help='Application smoke test to run.',
    )
    args = parser.parse_args(argv)
    selected = list(RUNNERS) if args.case == 'all' else [args.case]
    for name in selected:
        print(f'=== smoke: {name} ===')
        RUNNERS[name]()
    print(f"SMOKE_TESTS_OK cases={selected}")


if __name__ == '__main__':
    main()
