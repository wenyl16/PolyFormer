"""Train PolyFormer for the anonymized real three-phase network case."""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np
import torch

from Simulator import PROJECT_ROOT
from Simulator.Approximator import FullNet, PreTrainNet, Trainer, compute_loss
import Simulator.cases.DS_case_3phase as DS_case_3phase


def _device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(name)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-type', choices=('pretrainnet', 'fullnet'), default='pretrainnet')
    parser.add_argument('--variant', choices=('feasible', 'moderate'), default='feasible')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--phase2-epochs', type=int)
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
    parser.add_argument('--smoke', action='store_true')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = _device(args.device)
    save_artifacts = not args.no_save and not args.smoke
    model_type = 'pretrainnet' if args.smoke else args.model_type

    ppc = DS_case_3phase.case36real_3phase_ds()
    p_rated = sum(ppc['bus'][:, 2]) / ppc['baseMVA']
    case = DS_case_3phase.DScase_3phase_train(
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
        pretrain_epochs = 500 if args.epochs is None else args.epochs
        schedule = [(1 if args.smoke else pretrain_epochs, 0.1 / p_rated, 1.0)]
    else:
        generated_pretrain_path = (
            args.output_root / 'ds_proj' / case['casename'] / 'pretrainnet_weights.pth'
        )
        archived_pretrain_path = (
            PROJECT_ROOT / 'results' / 'ds_proj' /
            case['casename'] / 'pretrainnet_weights.pth'
        )
        pretrain_path = (
            generated_pretrain_path
            if generated_pretrain_path.is_file()
            else archived_pretrain_path
        )
        if not pretrain_path.is_file():
            raise FileNotFoundError(
                f"Pretrained weights not found: {pretrain_path}. Run pretraining first."
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
        if args.variant == 'moderate':
            schedule = [((60 if args.epochs is None else args.epochs), 5e-5 / p_rated, 0.6)]
        else:
            schedule = [
                ((40 if args.epochs is None else args.epochs), 1e-5 / p_rated, 1.0),
                ((10 if args.phase2_epochs is None else args.phase2_epochs), 2e-6 / p_rated, 1e-3),
            ]

    trainer = Trainer(model=model, error_calculator=case['errorcalculator'], compute_loss=compute_loss)
    started = time.time()
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
                case['casename'] / 'fullnet_weights.pth'
            )
        else:
            result_path = case['result_path']
        result_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), result_path)

    print(
        f"DS_3PHASE_OK model={model_type} variant={args.variant} "
        f"updates={len(trainer.loss_history)} elapsed_s={time.time() - started:.2f}"
    )


if __name__ == '__main__':
    main()
