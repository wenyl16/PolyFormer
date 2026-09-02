"""Train PolyFormer for one or more DRCC portfolio groups."""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

from Simulator import PROJECT_ROOT
from Simulator.Approximator import FullNet, Trainer, compute_loss
from Simulator.cases.DRCC_case import DRCCModelBuilder


PAPER_CASES = {
    "50x2x150": (50, 2, 150),
    "150x3x300": (150, 3, 300),
    "300x5x900": (300, 5, 900),
    "400x8x1280": (400, 8, 1280),
}


def _device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=PAPER_CASES, default="400x8x1280")
    parser.add_argument(
        "--group",
        type=int,
        action="append",
        help="Zero-based group id. Repeat to train multiple groups; omit for all groups.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--theta-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "results",
        help="Root of the existing results tree (default: PROJECT_ROOT/results).",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one update for group 0 of the tiny x2g1s10 fixture without writing files.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.smoke:
        n_assets, n_groups, n_samples = 2, 1, 10
        groups = [0]
        epochs = 1
        theta_samples = 1
        batch_size = 1
        save_artifacts = False
    else:
        n_assets, n_groups, n_samples = PAPER_CASES[args.case]
        groups = args.group
        epochs = args.epochs
        theta_samples = args.theta_samples
        batch_size = args.batch_size
        save_artifacts = not args.no_save

    data_path = PROJECT_ROOT / 'data' / 'DRCC' / f'r_samples_x{n_assets}g{n_groups}s{n_samples}.csv'
    if not data_path.is_file():
        raise FileNotFoundError(f"DRCC input data not found: {data_path}")
    r_samples = pd.read_csv(data_path)
    actual_groups = len(set(r_samples['group']))
    if actual_groups != n_groups:
        raise ValueError(
            f"{data_path.name} contains {actual_groups} groups; expected {n_groups}."
        )

    params = {
        'r_samples': r_samples,
        'R_limits': [(-0.1, 0.1)] * n_groups,
    }
    portfolio = DRCCModelBuilder(params)
    group_ids = list(range(portfolio.group_number)) if groups is None else groups
    invalid = [group for group in group_ids if group not in portfolio.group_dataset]
    if invalid:
        raise ValueError(f"Invalid group id(s) {invalid}; valid ids are 0..{portfolio.group_number - 1}.")

    device = _device(args.device)
    started = time.time()
    for group_id in group_ids:
        case = portfolio.build_drcc_train(
            portfolio.group_dataset[group_id],
            model_type='fullnet',
            plot_flag=False,
            total_samples=theta_samples,
            batch_size=batch_size,
            device=device,
            save_artifacts=save_artifacts,
            result_root=args.output_root,
        )
        model = FullNet(
            dim_theta=case['params']['count'],
            A_init=case['A_hat'],
            b_init=case['b_hat'],
            is_epigraph=False,
            n_hidden=128,
            device=device,
        ).to(device)
        trainer = Trainer(model=model, error_calculator=case['errorcalculator'], compute_loss=compute_loss)
        trainer.configure(**case['trainer_configure'])
        trainer.configure(lr=2e-4, rate_opt_feas=1.0, optimizer='adam')
        if args.smoke:
            trainer.configure(n_cal=1, call_interval=1)
        trainer.initialize()
        trainer.train(
            n_train=epochs,
            params_data=case['params'],
            parallel=args.parallel and not args.smoke,
        )

        if save_artifacts:
            case['result_path'].parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), case['result_path'])
        print(
            f"DRCC_GROUP_OK case={case['casename']} group={group_id} "
            f"epochs={epochs} updates={len(trainer.loss_history)}"
        )

    print(f"DRCC_OK groups={group_ids} elapsed_s={time.time() - started:.2f}")


if __name__ == "__main__":
    main()
