"""Train the polygon Extended Data experiment."""

# KMP_DUPLICATE_LIB_OK must be set before importing torch.
# ruff: noqa: I001

import argparse
import os
from pathlib import Path
from unittest.mock import patch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np

from Simulator import PROJECT_ROOT
from Simulator.Approximator import BiasNet, FullNet, PreTrainNet, Trainer, compute_loss
from Simulator.Plotter import ErrorVisualizer
from Simulator.cases.basic_cases import case_polygon


DEFAULT_MODEL_TYPE = "fullnet"
DEFAULT_PARALLEL = True


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train PolyFormer on the two-dimensional polygon experiment."
    )
    parser.add_argument(
        "--model-type",
        choices=("pretrainnet", "biasnet", "fullnet"),
        default=DEFAULT_MODEL_TYPE,
        help="Network variant to train (default: %(default)s).",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Torch device; auto uses CUDA when available (default: %(default)s).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "results",
        help="Root of the existing results tree (default: PROJECT_ROOT/results).",
    )
    parser.add_argument(
        "--parallel",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_PARALLEL,
        help="Enable or disable parallel error calculations (default: enabled).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one update with one sample and one error calculation; never save weights.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Run training without writing weights or callback artifacts.",
    )
    return parser.parse_args(argv)


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return torch.device(name)


def build_model(case, model_type, device):
    if model_type == "pretrainnet":
        return PreTrainNet(case["A_hat"], case["b_hat"], device=device).to(device)

    generated_pretrained_path = Path(case["result_path"]).with_name(
        "pretrainnet_weights.pth"
    )
    archived_pretrained_path = (
        Path(PROJECT_ROOT)
        / "results"
        / case["casename"]
        / "pretrainnet_weights.pth"
    )
    pretrained_path = (
        generated_pretrained_path
        if generated_pretrained_path.is_file()
        else archived_pretrained_path
    )
    if not pretrained_path.is_file():
        raise FileNotFoundError(
            f"Pretrained polygon weights are required for {model_type}: {pretrained_path}"
        )

    pretrained_model = PreTrainNet(
        case["A_hat"], case["b_hat"], device=device
    ).to(device)
    pretrained_model.load_state_dict(
        torch.load(pretrained_path, map_location=device, weights_only=True)
    )
    with torch.no_grad():
        A_pretrained, b_pretrained = pretrained_model()

    b_init = b_pretrained[0].detach().cpu().numpy()
    if model_type == "biasnet":
        A_pretrained = A_pretrained[0].detach().to(device)
        case["trainer_configure"].update(A_pretrained=A_pretrained)
        return BiasNet(
            dim_theta=case["params"]["count"], b_init=b_init, device=device
        ).to(device)

    A_init = A_pretrained[0].detach().cpu().numpy()
    return FullNet(
        dim_theta=case["params"]["count"],
        A_init=A_init,
        b_init=b_init,
        device=device,
    ).to(device)


def build_case(case_options, smoke):
    if not smoke:
        return case_polygon(**case_options)

    compute_errors = ErrorVisualizer.compute_errors

    def compute_one_initial_error_sample(visualizer, model, num_sample=50):
        del num_sample
        return compute_errors(visualizer, model, num_sample=1)

    with patch.object(
        ErrorVisualizer, "compute_errors", compute_one_initial_error_sample
    ):
        return case_polygon(**case_options)


def run(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    save_artifacts = not (args.smoke or args.no_save)
    case_options = {
        "model_type": args.model_type,
        "device": device,
        "save_artifacts": save_artifacts,
        "result_root": args.output_root,
    }
    if args.smoke:
        case_options.update(total_samples=1, batch_size=1)

    case = build_case(case_options, args.smoke)
    model = build_model(case, args.model_type, device)
    trainer = Trainer(
        model=model,
        error_calculator=case["errorcalculator"],
        compute_loss=compute_loss,
    )
    trainer.configure(**case["trainer_configure"])
    if args.smoke:
        trainer.configure(n_cal=1)
    if args.smoke or args.no_save:
        trainer.configure(training_callback=None)
    trainer.initialize()

    n_train = 1 if args.smoke else (1000 if args.model_type == "pretrainnet" else 20)
    trainer.train(
        n_train=n_train,
        params_data=case["params"],
        parallel=args.parallel,
    )

    if args.smoke or args.no_save:
        print("Polygon run completed; weights were not saved.")
    else:
        output_path = Path(case["result_path"])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output_path)
        print(f"Saved polygon weights to {output_path}")
    return trainer


def main(argv=None):
    return run(parse_args(argv))


if __name__ == "__main__":
    main()
