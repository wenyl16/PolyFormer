"""Train PolyFormer for the eight balanced distribution-network cases."""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from Simulator import PROJECT_ROOT
from Simulator.Approximator import FullNet, PreTrainNet, Trainer, compute_loss
import Simulator.cases.TD_case as TD_case


CASE_BUILDERS = {
    'case10ba_ds': TD_case.case10ba_ds,
    'case17me_ds': TD_case.case17me_ds,
    'case33bw_ds': TD_case.case33bw_ds,
    'case51ga_ds': TD_case.case51ga_ds,
    'case74_ds': TD_case.case74_ds,
    'case118zh_ds': TD_case.case118zh_ds,
    'case136ma_ds': TD_case.case136ma_ds,
    'case533mt_hi_ds': TD_case.case533mt_hi_ds,
}
SMALL_CASES = {'case10ba_ds', 'case17me_ds', 'case33bw_ds'}
LARGE_CASE = 'case533mt_hi_ds'


def _device(name: str):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--case', choices=CASE_BUILDERS, action='append',
        help='Repeat for multiple networks; omit to run all eight.',
    )
    parser.add_argument('--model-type', choices=('pretrainnet', 'fullnet'), default='pretrainnet')
    parser.add_argument('--variant', choices=('feasible', 'moderate'), default='feasible')
    parser.add_argument('--epochs', type=int, help='Override the paper epoch count for the first phase.')
    parser.add_argument('--phase2-epochs', type=int, help='Override the feasible variant second phase.')
    parser.add_argument('--theta-samples', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--output-root', type=Path,
        default=PROJECT_ROOT / 'results',
        help='Root of the existing results tree (default: PROJECT_ROOT/results).',
    )
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument(
        '--smoke', action='store_true',
        help='Run one pretraining update on case10ba_ds without writing files.',
    )
    return parser


def _fullnet_schedule(case_name, p_rated, variant, epochs, phase2_epochs):
    if variant == 'moderate':
        return [(epochs if epochs is not None else 60, 5e-5 / p_rated, 0.6)]

    phase1_lr = (5e-5 if case_name in SMALL_CASES else 1e-5) / p_rated
    if case_name in SMALL_CASES:
        phase2_lr = 5e-5 / p_rated
    elif case_name == LARGE_CASE:
        phase2_lr = 1e-6 / p_rated
    else:
        phase2_lr = 2e-6 / p_rated
    phase2_ratio = 1e-2 if case_name == LARGE_CASE else 1e-3
    return [
        (epochs if epochs is not None else 40, phase1_lr, 1.0),
        (phase2_epochs if phase2_epochs is not None else 10, phase2_lr, phase2_ratio),
    ]


def main(argv=None):
    args = build_parser().parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    case_names = args.case or list(CASE_BUILDERS)
    model_type = args.model_type
    save_artifacts = not args.no_save
    if args.smoke:
        case_names = ['case10ba_ds']
        model_type = 'pretrainnet'
        save_artifacts = False

    device = _device(args.device)
    started = time.time()
    for case_name in case_names:
        ppc = CASE_BUILDERS[case_name]()
        p_rated = sum(ppc['bus'][:, 2]) / ppc['baseMVA']
        case = TD_case.DScase_train(
            casedata=ppc,
            model_type=model_type,
            plot_flag=args.plot,
            total_samples=1 if args.smoke else args.theta_samples,
            noise_range=(-0.05, 0.05),
            batch_size=1 if args.smoke else args.batch_size,
            device=device,
            save_artifacts=save_artifacts,
            result_root=args.output_root,
        )

        if model_type == 'pretrainnet':
            model = PreTrainNet(case['A_hat'], case['b_hat'], is_epigraph=False, device=device).to(device)
            schedule = [(
                1 if args.smoke else (args.epochs if args.epochs is not None else 500),
                0.1 / p_rated,
                1.0,
            )]
        else:
            generated_pretrain_path = (
                args.output_root / 'ds_proj' / case_name / 'pretrainnet_weights.pth'
            )
            archived_pretrain_path = (
                PROJECT_ROOT / 'results' / 'ds_proj' / case_name / 'pretrainnet_weights.pth'
            )
            pretrain_path = (
                generated_pretrain_path
                if generated_pretrain_path.is_file()
                else archived_pretrain_path
            )
            if not pretrain_path.is_file():
                raise FileNotFoundError(
                    f"Pretrained weights not found: {pretrain_path}. Run --model-type pretrainnet first."
                )
            pretrain_model = PreTrainNet(case['A_hat'], case['b_hat'], is_epigraph=False, device=device).to(device)
            pretrain_model.load_state_dict(
                torch.load(pretrain_path, map_location=device, weights_only=True)
            )
            A_pretrained, b_pretrained = pretrain_model()
            model = FullNet(
                dim_theta=case['params']['count'],
                A_init=A_pretrained[0].detach().cpu().numpy(),
                b_init=b_pretrained[0].detach().cpu().numpy(),
                is_epigraph=False,
                n_hidden=128,
                device=device,
            ).to(device)
            schedule = _fullnet_schedule(
                case_name, p_rated, args.variant, args.epochs, args.phase2_epochs
            )

        trainer = Trainer(model=model, error_calculator=case['errorcalculator'], compute_loss=compute_loss)
        for epochs, learning_rate, rate_opt_feas in schedule:
            if epochs == 0:
                continue
            trainer.configure(**case['trainer_configure'])
            trainer.configure(lr=learning_rate, rate_opt_feas=rate_opt_feas)
            if args.smoke:
                trainer.configure(n_cal=1, call_interval=1)
            trainer.initialize()
            trainer.train(
                n_train=epochs,
                params_data=case['params'],
                parallel=args.parallel and not args.smoke,
            )

        if save_artifacts:
            if model_type == 'fullnet' and args.variant == 'moderate':
                result_path = (
                    args.output_root / 'ds_proj_original' /
                    case_name / 'fullnet_weights.pth'
                )
            else:
                result_path = case['result_path']
            result_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), result_path)
        print(
            f"DS_CASE_OK case={case_name} model={model_type} variant={args.variant} "
            f"updates={len(trainer.loss_history)}"
        )

    print(f"DS_OK cases={case_names} elapsed_s={time.time() - started:.2f}")


if __name__ == '__main__':
    main()
