"""Train PolyFormer approximators for the paper's cube scaling cases."""

# KMP_DUPLICATE_LIB_OK must be set before importing torch.
# ruff: noqa: I001

import argparse
import os
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch

from Simulator.Approximator import PreTrainNet, Trainer, compute_loss
from Simulator import PROJECT_ROOT


DEFAULT_DIMENSIONS = (2, 4, 6, 8, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200)
MODEL_TYPE = 'pretrainnet'


def _positive_dimension(value):
    dimension = int(value)
    if dimension <= 0:
        raise argparse.ArgumentTypeError('dimensions must be positive integers')
    return dimension


def _device(name):
    if name == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('--device cuda was requested, but CUDA is not available.')
    return torch.device(name)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dimensions', nargs='+', type=_positive_dimension,
        default=list(DEFAULT_DIMENSIONS), metavar='N',
        help='Dimensions to train (default: the paper scaling sweep).',
    )
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument(
        '--output-root', type=Path,
        default=PROJECT_ROOT / 'results',
        help='Root of the existing results tree (default: PROJECT_ROOT/results).',
    )
    parallel_group = parser.add_mutually_exclusive_group()
    parallel_group.add_argument('--parallel', dest='parallel', action='store_true')
    parallel_group.add_argument('--no-parallel', dest='parallel', action='store_false')
    parser.set_defaults(parallel=False)
    parser.add_argument('--no-save', action='store_true', help='Do not write weights or callback results.')
    parser.add_argument(
        '--smoke', action='store_true',
        help='Run one update at the smallest selected dimension without saving.',
    )
    return parser


def _build_model(case, device):
    return PreTrainNet(case['A_hat'], case['b_hat'], device=device).to(device)


def _schedule(dimension, smoke):
    scale = np.sqrt(dimension / 2)
    if smoke:
        return [(1, 2e-2 / scale)]
    return [
        (int(50 * scale), 2e-2 / scale),
        (int(400 * scale), 3e-1 / scale),
    ]


def main(argv=None):
    args = build_parser().parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    from Simulator.cases.basic_cases import case_cube

    dimensions = [min(args.dimensions)] if args.smoke else args.dimensions
    save_artifacts = not (args.no_save or args.smoke)
    device = _device(args.device)

    for dimension in dimensions:
        case = case_cube(
            dim=dimension, model_type=MODEL_TYPE, device=device,
            save_artifacts=save_artifacts, result_root=args.output_root,
        )
        model = _build_model(case, device)
        trainer = Trainer(
            model=model,
            error_calculator=case['errorcalculator'],
            compute_loss=compute_loss,
        )

        for n_train, learning_rate in _schedule(dimension, args.smoke):
            trainer.configure(**case['trainer_configure'])
            trainer.configure(lr=learning_rate)
            if not save_artifacts:
                trainer.configure(training_callback=None)
            if args.smoke:
                trainer.configure(n_cal=1, call_interval=1)
            trainer.initialize()
            trainer.train(
                n_train=n_train,
                params_data=case['params'],
                parallel=args.parallel,
            )

        if save_artifacts:
            output_path = Path(case['result_path']) / f'{MODEL_TYPE}_weights_dim{dimension}.pth'
            torch.save(
                model.state_dict(),
                output_path,
            )
            print(f'Saved cube weights to {output_path}')
        else:
            print(f'Cube dimension {dimension} completed; artifacts were not saved.')


if __name__ == '__main__':
    main()
