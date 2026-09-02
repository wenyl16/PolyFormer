"""Train the two resource-aggregation experiments reported in the paper."""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

from Simulator.Approximator import PreTrainNet, Trainer, compute_loss
from Simulator.cases.aggregation_case import Aggregator
from Simulator import PROJECT_ROOT


PAPER_SCENARIOS = {
    "continuous": {
        "n_ev": 600,
        "n_hp": 400,
        "n_bss": 0,
        "n_discrete_ev": 0,
        "n_discrete_hp": 0,
        "lr": 2e-2,
    },
    "mixed": {
        "n_ev": 60,
        "n_hp": 40,
        "n_bss": 5,
        "n_discrete_ev": 6,
        "n_discrete_hp": 4,
        "lr": 1e-2,
    },
}


def _device(name: str):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=PAPER_SCENARIOS, default="mixed")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--phase1-epochs", type=int, default=500)
    parser.add_argument("--phase2-epochs", type=int, default=200)
    parser.add_argument("--phase3-epochs", type=int, default=100)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "results",
        help="Root of the existing results tree (default: PROJECT_ROOT/results).",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not write weights or snapshots.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one update on a tiny fleet for the selected scenario without writing artifacts.",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    settings = dict(PAPER_SCENARIOS[args.scenario])
    phase_epochs = [args.phase1_epochs, args.phase2_epochs, args.phase3_epochs]
    save_artifacts = not args.no_save

    if args.smoke:
        if args.scenario == "mixed":
            settings.update(
                n_ev=2,
                n_hp=1,
                n_bss=1,
                n_discrete_ev=1,
                n_discrete_hp=1,
            )
        else:
            settings.update(
                n_ev=2,
                n_hp=1,
                n_bss=0,
                n_discrete_ev=0,
                n_discrete_hp=0,
            )
        phase_epochs = [1, 0, 0]
        save_artifacts = False

    device = _device(args.device)
    aggregator = Aggregator(seed=args.seed, discrete_rate=0.0)
    aggregator.gen_EV(settings["n_ev"], n_discrete=settings["n_discrete_ev"])
    aggregator.gen_TCL(settings["n_hp"], n_discrete=settings["n_discrete_hp"])
    aggregator.gen_ESS(settings["n_bss"])
    case = aggregator.case_aggregator(
        model_type="pretrainnet", save_artifacts=save_artifacts,
        result_root=args.output_root,
    )

    model = PreTrainNet(case["A_hat"], case["b_hat"], device=device).to(device)
    trainer = Trainer(model=model, error_calculator=case["errorcalculator"], compute_loss=compute_loss)

    started = time.time()
    for epochs, rate_opt_feas in zip(phase_epochs, (1.0, 0.1, 1e-4)):
        if epochs == 0:
            continue
        trainer.configure(**case["trainer_configure"])
        trainer.configure(lr=settings["lr"], rate_opt_feas=rate_opt_feas)
        if args.smoke:
            trainer.configure(n_cal=1, call_interval=1)
        trainer.initialize()
        trainer.train(n_train=epochs, params_data=case["params"], parallel=False)

    if save_artifacts:
        case["result_path"].parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), case["result_path"])

    print(
        "AGGREGATION_OK "
        f"scenario={args.scenario} resources={settings['n_ev'] + settings['n_hp'] + settings['n_bss']} "
        f"discrete_ev={settings['n_discrete_ev']} discrete_hp={settings['n_discrete_hp']} "
        f"updates={sum(phase_epochs)} elapsed_s={time.time() - started:.2f}"
    )


if __name__ == "__main__":
    main()
