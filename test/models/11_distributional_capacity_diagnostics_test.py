# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]
# This file is automatically mirrored from the corresponding .ipynb for git diff purposes.
# Actual research execution should be performed in the Jupyter Notebook (.ipynb)
# or in an approved remote/server environment.

# %% [markdown]
# # 11번 위험 이벤트·분포 예측 진단
#
# 10번의 다음 15분 점예측과 경쟁하도록, 11번에서는 향후 4시간 급변·하방 위험 확률과
# 수익률 분포를 예측합니다. Double Descent는 주 연구가 아니라 보조 용량 진단으로 분리합니다.
#
# 모든 표와 그림은 notebook inline output으로만 표시하며 서버에 PNG/CSV/Markdown 파일을 저장하지 않습니다.

# %%
"""11번: 위험 이벤트·직접 분포 예측과 보조 용량 진단."""

from __future__ import annotations

import argparse
import copy
import gc
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset


def find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents, Path.home() / "personal_ai_project" / "quantitative_trading"]:
        if (candidate / "pyproject.toml").exists() and (candidate / "test").exists():
            return candidate
    return start


REPO_ROOT = find_repo_root(Path.cwd())
OBJECTIVE_BACKEND_PATH = REPO_ROOT / "test" / "models" / "10_objective_ensemble_confirmation_test.py"


def load_objective_backend():
    spec = importlib.util.spec_from_file_location("objective_ensemble_backend", OBJECTIVE_BACKEND_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"10번 backend를 불러오지 못했다: {OBJECTIVE_BACKEND_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


obj = load_objective_backend()
diag = obj.diag
base = obj.base

DEFAULT_MODELS = ["Linear", "PatchTSTLike"]
DEFAULT_DISTRIBUTIONS = ["gaussian", "student_t", "quantile"]
DEFAULT_PREPROCESSINGS = ["seasonal_diff16", "winsor_025"]
DEFAULT_EVENT_KINDS = ["downside", "absolute_move"]
DEFAULT_WIDTHS = [16, 24, 32, 48, 64, 96, 128, 192, 256]
DEFAULT_SAMPLE_FRACTIONS = [0.25, 0.50, 0.75, 1.00]
DEFAULT_SEEDS = [42, 137, 2026]
QUANTILE_LEVELS = (0.05, 0.50, 0.95)


def configure_inline_matplotlib() -> None:
    diag.configure_inline_matplotlib()


def show_figure(fig) -> None:
    diag.show_figure(fig)


def make_tensor_loader(split, batch_size: int, shuffle: bool, profile) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(split["x"]), torch.from_numpy(split["y"]))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
        drop_last=False,
    )


class DistributionalLinear(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int, output_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * n_features, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
        )
        self.head = nn.Linear(hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class DistributionalPatchTST(nn.Module):
    def __init__(
        self,
        seq_len: int,
        n_features: int,
        hidden: int,
        output_dim: int,
        patch_len: int = 8,
        heads: int = 4,
    ):
        super().__init__()
        if hidden % heads:
            raise ValueError(f"PatchTSTLike hidden={hidden}은 heads={heads}로 나누어져야 한다.")
        self.patch_len = patch_len
        self.n_patches = max(1, seq_len // patch_len)
        self.projection = nn.Linear(n_features * patch_len, hidden)
        layer = nn.TransformerEncoderLayer(
            hidden,
            heads,
            hidden * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, features = x.shape
        usable = self.n_patches * self.patch_len
        patches = x[:, -usable:].reshape(batch, self.n_patches, features * self.patch_len)
        encoded = self.encoder(self.projection(patches))
        return self.head(encoded.mean(dim=1))


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def initialize_distribution_head(model: nn.Module, distribution: str) -> None:
    layer = model.head[-1] if isinstance(model.head, nn.Sequential) else model.head
    nn.init.normal_(layer.weight, mean=0.0, std=1e-3)
    nn.init.zeros_(layer.bias)
    with torch.no_grad():
        if distribution in {"gaussian", "student_t"}:
            # sigmoid(raw) * 0.1 ~= 0.002, matching the observed 15-minute return scale.
            layer.bias[1] = math.log(0.02 / 0.98)
        if distribution == "student_t":
            layer.bias[2] = inverse_softplus(5.0 - 2.1)
        if distribution == "quantile":
            layer.bias[1] = inverse_softplus(0.003)
            layer.bias[2] = inverse_softplus(0.003)


def make_distribution_model(
    model_name: str,
    distribution: str,
    seq_len: int,
    n_features: int,
    hidden: int,
) -> nn.Module:
    output_dim = 3 if distribution in {"student_t", "quantile"} else 2
    if model_name == "Linear":
        model = DistributionalLinear(seq_len, n_features, hidden, output_dim)
    elif model_name == "PatchTSTLike":
        model = DistributionalPatchTST(seq_len, n_features, hidden, output_dim)
    else:
        raise KeyError(f"분포 예측에서 지원하지 않는 모델: {model_name}")
    initialize_distribution_head(model, distribution)
    return model


def decode_distribution(raw: torch.Tensor, distribution: str) -> dict[str, torch.Tensor]:
    if distribution == "gaussian":
        location = raw[:, 0]
        scale = 1e-5 + 0.1 * torch.sigmoid(raw[:, 1])
        return {"location": location, "scale": scale}
    if distribution == "student_t":
        location = raw[:, 0]
        scale = 1e-5 + 0.1 * torch.sigmoid(raw[:, 1])
        degrees = (2.1 + F.softplus(raw[:, 2])).clamp(2.1, 30.0)
        return {"location": location, "scale": scale, "degrees": degrees}
    if distribution == "quantile":
        median = raw[:, 0]
        lower = median - F.softplus(raw[:, 1])
        upper = median + F.softplus(raw[:, 2])
        return {"location": median, "lower": lower, "upper": upper}
    raise ValueError(f"지원하지 않는 분포: {distribution}")


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    error = target - prediction
    return torch.maximum(tau * error, (tau - 1.0) * error).mean()


def distribution_loss(raw: torch.Tensor, target: torch.Tensor, distribution: str) -> torch.Tensor:
    params = decode_distribution(raw, distribution)
    if distribution == "gaussian":
        dist = torch.distributions.Normal(params["location"], params["scale"])
        return -dist.log_prob(target).mean()
    if distribution == "student_t":
        dist = torch.distributions.StudentT(
            params["degrees"],
            params["location"],
            params["scale"],
        )
        return -dist.log_prob(target).mean()
    return (
        pinball_loss(params["lower"], target, QUANTILE_LEVELS[0])
        + pinball_loss(params["location"], target, QUANTILE_LEVELS[1])
        + pinball_loss(params["upper"], target, QUANTILE_LEVELS[2])
    ) / 3.0


def train_distribution_model(
    model: nn.Module,
    splits,
    profile,
    args: argparse.Namespace,
    distribution: str,
    batch_size: int,
) -> tuple[nn.Module, pd.DataFrame]:
    device = torch.device(profile.device)
    model = model.to(device)
    train_loader = make_tensor_loader(splits["train"], batch_size, True, profile)
    val_loader = make_tensor_loader(splits["val"], batch_size, False, profile)
    optimizer = base.make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = base.make_scheduler(optimizer, args.scheduler, args.epochs, len(train_loader), args.lr)
    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        gradients: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = distribution_loss(model(xb), yb, distribution)
            loss.backward()
            gradients.append(base.apply_gradient_policy(model, args.gradient_policy, epoch))
            optimizer.step()
            if args.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_losses.append(float(distribution_loss(model(xb), yb, distribution).detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        if scheduler is not None and args.scheduler == "plateau":
            scheduler.step(val_loss)
        elif scheduler is not None and args.scheduler not in {"onecycle", "plateau"}:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "grad_norm_mean": float(np.mean(gradients)),
            "grad_norm_max": float(np.max(gradients)),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        rows.append(row)
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
            f"grad={row['grad_norm_mean']:.4f} lr={row['lr']:.2e}"
        )
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] epoch={epoch} best_val={best_val:.6f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict_distribution(
    model: nn.Module,
    split,
    profile,
    batch_size: int,
    distribution: str,
) -> dict[str, np.ndarray]:
    device = torch.device(profile.device)
    loader = make_tensor_loader(split, batch_size, False, profile)
    collected: dict[str, list[np.ndarray]] = {}
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            params = decode_distribution(model(xb.to(device, non_blocking=True)), distribution)
            for name, value in params.items():
                collected.setdefault(name, []).append(value.detach().cpu().numpy())
    return {name: np.concatenate(chunks) for name, chunks in collected.items()}


def raw_interval(params: dict[str, np.ndarray], distribution: str, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    if distribution == "gaussian":
        quantile = float(stats.norm.ppf(1.0 - alpha / 2.0))
        return params["location"] - quantile * params["scale"], params["location"] + quantile * params["scale"]
    if distribution == "student_t":
        quantile = stats.t.ppf(1.0 - alpha / 2.0, params["degrees"])
        return params["location"] - quantile * params["scale"], params["location"] + quantile * params["scale"]
    return params["lower"], params["upper"]


def conformalize_interval(
    val_actual: np.ndarray,
    val_lower: np.ndarray,
    val_upper: np.ndarray,
    test_lower: np.ndarray,
    test_upper: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    scores = np.maximum.reduce([val_lower - val_actual, val_actual - val_upper, np.zeros_like(val_actual)])
    level = min(1.0, math.ceil((len(scores) + 1) * (1.0 - alpha)) / len(scores))
    correction = float(np.quantile(scores, level, method="higher"))
    return test_lower - correction, test_upper + correction, correction


def proper_score(
    actual: np.ndarray,
    params: dict[str, np.ndarray],
    distribution: str,
) -> float:
    if distribution == "gaussian":
        return float(-np.mean(stats.norm.logpdf(actual, loc=params["location"], scale=params["scale"])))
    if distribution == "student_t":
        standardized = (actual - params["location"]) / params["scale"]
        return float(-np.mean(stats.t.logpdf(standardized, df=params["degrees"]) - np.log(params["scale"])))
    total = 0.0
    for name, tau in [("lower", 0.05), ("location", 0.50), ("upper", 0.95)]:
        error = actual - params[name]
        total += float(np.mean(np.maximum(tau * error, (tau - 1.0) * error)))
    return total / 3.0


def pit_values(actual: np.ndarray, params: dict[str, np.ndarray], distribution: str) -> np.ndarray:
    if distribution == "gaussian":
        return stats.norm.cdf(actual, loc=params["location"], scale=params["scale"])
    if distribution == "student_t":
        standardized = (actual - params["location"]) / params["scale"]
        return stats.t.cdf(standardized, df=params["degrees"])
    return np.asarray([], dtype=np.float64)


def interval_diagnostics(
    actual: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    covered = (actual >= lower) & (actual <= upper)
    width = upper - lower
    miss_distance = np.maximum(lower - actual, 0.0) + np.maximum(actual - upper, 0.0)
    volatility_threshold = float(np.quantile(np.abs(actual), 0.75))
    high_mask = np.abs(actual) >= volatility_threshold
    low_mask = ~high_mask
    return {
        f"{prefix}_coverage": float(np.mean(covered)),
        f"{prefix}_width": float(np.mean(width)),
        f"{prefix}_miss_distance": float(np.mean(miss_distance)),
        f"{prefix}_high_vol_coverage": float(np.mean(covered[high_mask])),
        f"{prefix}_low_vol_coverage": float(np.mean(covered[low_mask])),
    }


def show_distribution_diagnostics(
    curves: pd.DataFrame,
    split,
    params: dict[str, np.ndarray],
    lower: np.ndarray,
    upper: np.ndarray,
    raw_lower: np.ndarray,
    raw_upper: np.ndarray,
    distribution: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    actual = split["y"]
    prediction = params["location"]
    width = upper - lower
    absolute_error = np.abs(actual - prediction)
    pit = pit_values(actual, params, distribution)
    n = min(300, len(actual))
    x = np.arange(n)
    fig, axes = plt.subplots(3, 2, figsize=(18, 12), dpi=140)

    axes[0, 0].plot(curves["epoch"], curves["train_loss"], label="train")
    axes[0, 0].plot(curves["epoch"], curves["val_loss"], label="validation")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("proper scoring loss")
    axes[0, 0].set_title("Distribution learning curve")
    axes[0, 0].legend()

    axes[0, 1].plot(x, actual[-n:], label="actual return")
    axes[0, 1].plot(x, prediction[-n:], label="distribution center")
    axes[0, 1].fill_between(x, raw_lower[-n:], raw_upper[-n:], alpha=0.16, label="raw 90% interval")
    axes[0, 1].fill_between(x, lower[-n:], upper[-n:], alpha=0.20, label="conformalized interval")
    axes[0, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[0, 1].set_xlabel("test time index")
    axes[0, 1].set_ylabel("next log return")
    axes[0, 1].set_title("Point and interval forecast")
    axes[0, 1].legend(fontsize=8)

    obj.show_candlestick_comparison(
        axes[1, 0],
        split,
        prediction,
        "Next-candle comparison",
        n=min(60, len(prediction)),
    )

    axes[1, 1].scatter(width, absolute_error, s=13, alpha=0.45)
    axes[1, 1].set_xlabel("prediction interval width")
    axes[1, 1].set_ylabel("absolute point error")
    axes[1, 1].set_title("Sharpness versus realized error")

    limit = max(float(np.max(np.abs(actual))), float(np.max(np.abs(prediction))), 1e-4)
    axes[2, 0].scatter(actual, prediction, s=13, alpha=0.45)
    axes[2, 0].plot([-limit, limit], [-limit, limit], color="black", linewidth=0.8)
    axes[2, 0].set_xlim(-limit, limit)
    axes[2, 0].set_ylim(-limit, limit)
    axes[2, 0].set_xlabel("actual return")
    axes[2, 0].set_ylabel("predicted center")
    axes[2, 0].set_title("Return calibration scatter")

    if len(pit):
        axes[2, 1].hist(pit, bins=10, range=(0.0, 1.0), density=True, alpha=0.75)
        axes[2, 1].axhline(1.0, color="black", linestyle="--")
        axes[2, 1].set_xlabel("PIT value")
        axes[2, 1].set_ylabel("density")
        axes[2, 1].set_title("PIT uniformity")
    else:
        nominal = np.asarray([0.05, 0.50, 0.95])
        observed = np.asarray(
            [
                np.mean(actual <= params["lower"]),
                np.mean(actual <= params["location"]),
                np.mean(actual <= params["upper"]),
            ]
        )
        axes[2, 1].plot(nominal, observed, marker="o")
        axes[2, 1].plot([0, 1], [0, 1], color="black", linestyle="--")
        axes[2, 1].set_xlabel("nominal quantile")
        axes[2, 1].set_ylabel("observed frequency")
        axes[2, 1].set_title("Quantile calibration")

    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(title)
    show_figure(fig)
    obj.display_table(
        f"{title} prediction preview",
        obj.prediction_preview_table(split, prediction, lower=lower, upper=upper),
    )


def event_preview_table(split, probability: np.ndarray, threshold: float, n_rows: int = 12) -> pd.DataFrame:
    n_rows = min(n_rows, len(probability))
    selected = slice(-n_rows, None)
    return pd.DataFrame(
        {
            "timestamp": split["timestamp"][selected],
            "prev_close": np.round(split["prev_close"][selected], 0),
            "future_end_close": np.round(split["target_close"][selected], 0),
            "future_return": np.round(split["future_return"][selected], 6),
            "event_score": np.round(split["event_score"][selected], 6),
            "event_label": split["y"][selected].astype(int),
            "predicted_probability": np.round(probability[selected], 6),
            "threshold": np.round(np.full(n_rows, threshold), 6),
        }
    )


def run_distribution_case(case, features: pd.DataFrame, profile, args: argparse.Namespace):
    base.set_seed(int(case["seed"]))
    feature_columns = base.FEATURE_SETS[args.feature_set]
    base.apply_window_preprocessing = diag.apply_preprocessing_pipeline
    data = base.build_windows(
        features,
        feature_columns,
        args.seq_len,
        str(case["preprocessing"]),
        args.normalization,
        args.max_windows,
        args.stride,
    )
    splits = base.time_split(data, args.train_ratio, args.val_ratio)
    batch_size = args.batch_size or profile.batch_size or 32

    while True:
        try:
            model = make_distribution_model(
                str(case["model"]),
                str(case["distribution"]),
                args.seq_len,
                len(feature_columns),
                int(case["hidden"]),
            )
            param_count = sum(parameter.numel() for parameter in model.parameters())
            model, curves = train_distribution_model(
                model,
                splits,
                profile,
                args,
                str(case["distribution"]),
                batch_size,
            )
            val_params = predict_distribution(model, splits["val"], profile, batch_size, str(case["distribution"]))
            test_params = predict_distribution(model, splits["test"], profile, batch_size, str(case["distribution"]))
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 4:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(4, batch_size // 2)
            print(f"[oom-retry] batch_size {batch_size} -> {next_batch}")
            batch_size = next_batch

    val_lower, val_upper = raw_interval(val_params, str(case["distribution"]), args.interval_alpha)
    raw_lower, raw_upper = raw_interval(test_params, str(case["distribution"]), args.interval_alpha)
    lower, upper, correction = conformalize_interval(
        splits["val"]["y"],
        val_lower,
        val_upper,
        raw_lower,
        raw_upper,
        args.interval_alpha,
    )
    point_metrics = base.evaluate_predictions(splits["test"], test_params["location"])
    result = {
        "case_id": (
            f"{case['model']}__{case['preprocessing']}__{case['distribution']}"
            f"__seed{case['seed']}__h{case['hidden']}"
        ),
        "suite": "distribution_probe",
        **case,
        "param_count": int(param_count),
        "batch_size": int(batch_size),
        "proper_score": proper_score(splits["test"]["y"], test_params, str(case["distribution"])),
        "conformal_correction": correction,
        **point_metrics,
        **interval_diagnostics(splits["test"]["y"], raw_lower, raw_upper, "raw_interval"),
        **interval_diagnostics(splits["test"]["y"], lower, upper, "calibrated_interval"),
    }
    if args.case_plots:
        show_distribution_diagnostics(
            curves,
            splits["test"],
            test_params,
            lower,
            upper,
            raw_lower,
            raw_upper,
            str(case["distribution"]),
            str(result["case_id"]),
        )
    return result


def subset_training_split(split, fraction: float):
    size = max(32, int(len(split["y"]) * fraction))
    start = max(0, len(split["y"]) - size)
    return {name: values[start:] for name, values in split.items()}


def evaluate_objective_loss(model, split, profile, batch_size: int, objective: str) -> float:
    device = torch.device(profile.device)
    loader = make_tensor_loader(split, batch_size, False, profile)
    losses: list[float] = []
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss, _ = obj.objective_loss(model(xb), yb, objective, 1.0)
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train_capacity_model(
    model,
    splits,
    profile,
    args: argparse.Namespace,
    batch_size: int,
) -> tuple[nn.Module, pd.DataFrame]:
    device = torch.device(profile.device)
    model = model.to(device)
    train_loader = make_tensor_loader(splits["train"], batch_size, True, profile)
    optimizer = base.make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = base.make_scheduler(optimizer, args.scheduler, args.capacity_epochs, len(train_loader), args.lr)
    best_state = None
    best_val = float("inf")
    rows: list[dict[str, float]] = []

    for epoch in range(1, args.capacity_epochs + 1):
        model.train()
        train_losses: list[float] = []
        gradients: list[float] = []
        auxiliary_scale = min(1.0, max(0.0, (epoch - 1) / max(1, args.objective_warmup_epochs)))
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = obj.objective_loss(model(xb), yb, args.capacity_objective, auxiliary_scale)
            loss.backward()
            gradients.append(base.apply_gradient_policy(model, args.gradient_policy, epoch))
            optimizer.step()
            if args.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_objective_loss(
            model,
            splits["val"],
            profile,
            batch_size,
            args.capacity_objective,
        )
        test_loss = evaluate_objective_loss(
            model,
            splits["test"],
            profile,
            batch_size,
            args.capacity_objective,
        )
        train_loss = float(np.mean(train_losses))
        if scheduler is not None and args.scheduler == "plateau":
            scheduler.step(val_loss)
        elif scheduler is not None and args.scheduler not in {"onecycle", "plateau"}:
            scheduler.step()
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "test_loss_diagnostic_only": test_loss,
                "grad_norm_mean": float(np.mean(gradients)),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
            f"test_diag={test_loss:.6f} grad={rows[-1]['grad_norm_mean']:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def show_capacity_case(curves: pd.DataFrame, result: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8), dpi=140)
    axes[0].plot(curves["epoch"], curves["train_loss"], label="train")
    axes[0].plot(curves["epoch"], curves["val_loss"], label="validation")
    axes[0].plot(curves["epoch"], curves["test_loss_diagnostic_only"], label="test diagnostic")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("objective loss")
    axes[0].set_title("Epoch-wise error")
    axes[0].legend()

    gap = curves["val_loss"] - curves["train_loss"]
    axes[1].plot(curves["epoch"], gap)
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("validation - train loss")
    axes[1].set_title("Generalization gap")

    axes[2].plot(curves["epoch"], curves["grad_norm_mean"])
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("mean gradient norm")
    axes[2].set_title("Optimization path")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(str(result["case_id"]))
    show_figure(fig)


def run_capacity_case(case, features: pd.DataFrame, profile, args: argparse.Namespace):
    base.set_seed(int(case["seed"]))
    feature_columns = base.FEATURE_SETS[args.feature_set]
    base.apply_window_preprocessing = diag.apply_preprocessing_pipeline
    data = base.build_windows(
        features,
        feature_columns,
        args.seq_len,
        str(case["preprocessing"]),
        args.normalization,
        args.max_windows,
        args.stride,
    )
    splits = base.time_split(data, args.train_ratio, args.val_ratio)
    splits["train"] = subset_training_split(splits["train"], float(case["sample_fraction"]))
    batch_size = args.batch_size or profile.batch_size or 32

    while True:
        try:
            model = base.make_model(
                str(case["model"]),
                args.seq_len,
                len(feature_columns),
                int(case["hidden"]),
            )
            param_count = sum(parameter.numel() for parameter in model.parameters())
            model, curves = train_capacity_model(model, splits, profile, args, batch_size)
            val_prediction = base.predict(model, splits["val"], profile, batch_size)
            test_prediction = base.predict(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 4:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(4, batch_size // 2)
            print(f"[oom-retry] batch_size {batch_size} -> {next_batch}")
            batch_size = next_batch

    val_metrics = base.evaluate_predictions(splits["val"], val_prediction)
    test_metrics = base.evaluate_predictions(splits["test"], test_prediction)
    best_epoch = int(curves.loc[curves["val_loss"].idxmin(), "epoch"])
    result = {
        "case_id": (
            f"{case['model']}__{case['preprocessing']}__h{case['hidden']}"
            f"__n{case['sample_fraction']}__seed{case['seed']}"
        ),
        "suite": case["suite"],
        **case,
        "param_count": int(param_count),
        "train_samples": int(len(splits["train"]["y"])),
        "parameter_sample_ratio": float(param_count / max(1, len(splits["train"]["y"]))),
        "batch_size": int(batch_size),
        "best_epoch_by_validation": best_epoch,
        "final_train_loss": float(curves.iloc[-1]["train_loss"]),
        "best_val_loss": float(curves["val_loss"].min()),
        "test_loss_at_best_val_epoch_diagnostic": float(
            curves.loc[curves["val_loss"].idxmin(), "test_loss_diagnostic_only"]
        ),
        "val_copy_risk_ratio": val_metrics["copy_risk_ratio"],
        **test_metrics,
    }
    if args.case_plots:
        show_capacity_case(curves, result)
    return result, curves


def build_risk_windows(
    features: pd.DataFrame,
    feature_columns: list[str],
    seq_len: int,
    horizon: int,
    preprocessing: str,
    normalization: str,
    max_windows: int | None,
    stride: int,
) -> dict[str, np.ndarray]:
    values = features[feature_columns].to_numpy(np.float32)
    close = features["close"].to_numpy(np.float64)
    timestamps = features["timestamp"].astype(str).to_numpy()
    end_indices = list(range(seq_len, len(features) - horizon + 1, max(1, stride)))
    if max_windows and len(end_indices) > max_windows:
        end_indices = end_indices[-max_windows:]

    xs: list[np.ndarray] = []
    future_returns: list[float] = []
    realized_volatility: list[float] = []
    downside_scores: list[float] = []
    absolute_move_scores: list[float] = []
    prev_closes: list[float] = []
    target_closes: list[float] = []
    times: list[str] = []

    for end in end_indices:
        window = values[end - seq_len : end].copy()
        window = diag.apply_preprocessing_pipeline(window, preprocessing)
        window = base.apply_window_normalization(window, normalization)
        base_close = close[end - 1]
        path = close[end : end + horizon]
        path_log_returns = np.diff(np.log(np.concatenate([[base_close], path])))
        cumulative_path = np.cumsum(path_log_returns)
        xs.append(window)
        future_returns.append(float(cumulative_path[-1]))
        realized_volatility.append(float(np.sqrt(np.sum(path_log_returns**2))))
        downside_scores.append(float(max(0.0, -np.min(cumulative_path))))
        absolute_move_scores.append(float(np.max(np.abs(cumulative_path))))
        prev_closes.append(float(base_close))
        target_closes.append(float(path[-1]))
        times.append(str(timestamps[end + horizon - 1]))

    return {
        "x": np.stack(xs).astype(np.float32),
        "y": np.asarray(future_returns, dtype=np.float32),
        "future_return": np.asarray(future_returns, dtype=np.float32),
        "realized_volatility": np.asarray(realized_volatility, dtype=np.float32),
        "downside_score": np.asarray(downside_scores, dtype=np.float32),
        "absolute_move_score": np.asarray(absolute_move_scores, dtype=np.float32),
        "prev_close": np.asarray(prev_closes, dtype=np.float32),
        "target_close": np.asarray(target_closes, dtype=np.float32),
        "timestamp": np.asarray(times),
    }


def prepare_event_splits(
    data: dict[str, np.ndarray],
    train_ratio: float,
    val_ratio: float,
    event_kind: str,
    event_quantile: float,
) -> tuple[dict[str, dict[str, np.ndarray]], float]:
    splits = base.time_split(data, train_ratio, val_ratio)
    score_name = f"{event_kind}_score"
    threshold = float(np.quantile(splits["train"][score_name], event_quantile))
    for split in splits.values():
        split["event_score"] = split[score_name].copy()
        split["y"] = (split[score_name] >= threshold).astype(np.float32)
    return splits, threshold


def binary_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    positive_weight: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=positive_weight)


def train_event_model(
    model: nn.Module,
    splits,
    profile,
    args: argparse.Namespace,
    batch_size: int,
) -> tuple[nn.Module, pd.DataFrame]:
    device = torch.device(profile.device)
    model = model.to(device)
    train_loader = make_tensor_loader(splits["train"], batch_size, True, profile)
    val_loader = make_tensor_loader(splits["val"], batch_size, False, profile)
    positives = float(np.sum(splits["train"]["y"]))
    negatives = float(len(splits["train"]["y"]) - positives)
    positive_weight = torch.tensor(
        negatives / max(1.0, positives),
        dtype=torch.float32,
        device=device,
    )
    optimizer = base.make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = base.make_scheduler(optimizer, args.scheduler, args.epochs, len(train_loader), args.lr)
    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        gradients: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = binary_cross_entropy(model(xb), yb, positive_weight)
            loss.backward()
            gradients.append(base.apply_gradient_policy(model, args.gradient_policy, epoch))
            optimizer.step()
            if args.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()
            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_losses.append(
                    float(binary_cross_entropy(model(xb), yb, positive_weight).detach().cpu())
                )
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        if scheduler is not None and args.scheduler == "plateau":
            scheduler.step(val_loss)
        elif scheduler is not None and args.scheduler not in {"onecycle", "plateau"}:
            scheduler.step()
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "grad_norm_mean": float(np.mean(gradients)),
                "lr": float(optimizer.param_groups[0]["lr"]),
            }
        )
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
            f"grad={rows[-1]['grad_norm_mean']:.4f}"
        )
        if val_loss < best_val - args.min_delta:
            best_val = val_loss
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"[early-stop] epoch={epoch} best_val={best_val:.6f}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict_event_probability(
    model: nn.Module,
    split,
    profile,
    batch_size: int,
) -> np.ndarray:
    device = torch.device(profile.device)
    loader = make_tensor_loader(split, batch_size, False, profile)
    probabilities: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(probabilities)


def average_precision(actual: np.ndarray, probability: np.ndarray) -> float:
    positives = int(np.sum(actual))
    if positives == 0:
        return float("nan")
    order = np.argsort(-probability)
    sorted_actual = actual[order]
    true_positive = np.cumsum(sorted_actual)
    precision = true_positive / np.arange(1, len(actual) + 1)
    return float(np.sum(precision * sorted_actual) / positives)


def roc_auc(actual: np.ndarray, probability: np.ndarray) -> float:
    positives = actual == 1
    negatives = actual == 0
    if not positives.any() or not negatives.any():
        return float("nan")
    ranks = stats.rankdata(probability)
    positive_rank_sum = float(np.sum(ranks[positives]))
    n_pos = int(np.sum(positives))
    n_neg = int(np.sum(negatives))
    return float((positive_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def classification_metrics(
    actual: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    predicted = probability >= threshold
    positive = actual == 1
    true_positive = float(np.sum(predicted & positive))
    false_positive = float(np.sum(predicted & ~positive))
    false_negative = float(np.sum(~predicted & positive))
    precision = true_positive / max(1.0, true_positive + false_positive)
    recall = true_positive / max(1.0, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "brier_score": float(np.mean((probability - actual) ** 2)),
        "average_precision": average_precision(actual, probability),
        "roc_auc": roc_auc(actual, probability),
        "classification_error": float(np.mean(predicted != actual)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "event_rate": float(np.mean(actual)),
        "predicted_event_rate": float(np.mean(predicted)),
    }


def select_validation_threshold(actual: np.ndarray, probability: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        f1 = classification_metrics(actual, probability, float(threshold))["f1"]
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def calibration_table(
    actual: np.ndarray,
    probability: np.ndarray,
    bins: int = 10,
) -> pd.DataFrame:
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows: list[dict[str, float]] = []
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (probability >= lower) & (
            probability <= upper if index == bins - 1 else probability < upper
        )
        if not mask.any():
            continue
        rows.append(
            {
                "lower": lower,
                "upper": upper,
                "count": int(mask.sum()),
                "mean_probability": float(np.mean(probability[mask])),
                "observed_frequency": float(np.mean(actual[mask])),
            }
        )
    return pd.DataFrame(rows)


def expected_calibration_error(calibration: pd.DataFrame) -> float:
    if calibration.empty:
        return float("nan")
    total = float(calibration["count"].sum())
    error = np.abs(calibration["mean_probability"] - calibration["observed_frequency"])
    return float(np.sum(error * calibration["count"] / total))


def selective_risk_curve(
    actual: np.ndarray,
    probability: np.ndarray,
) -> pd.DataFrame:
    confidence = np.abs(probability - 0.5) * 2.0
    predicted = probability >= 0.5
    order = np.argsort(-confidence)
    rows: list[dict[str, float]] = []
    for coverage in np.linspace(0.10, 1.00, 10):
        count = max(1, int(len(actual) * coverage))
        selected = order[:count]
        rows.append(
            {
                "coverage": float(count / len(actual)),
                "classification_error": float(np.mean(predicted[selected] != actual[selected])),
                "brier_score": float(np.mean((probability[selected] - actual[selected]) ** 2)),
            }
        )
    return pd.DataFrame(rows)


def show_event_diagnostics(
    curves: pd.DataFrame,
    split,
    probability: np.ndarray,
    validation_threshold: float,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    actual = split["y"]
    calibration = calibration_table(actual, probability)
    selective = selective_risk_curve(actual, probability)
    order = np.argsort(-probability)
    sorted_actual = actual[order]
    cumulative_tp = np.cumsum(sorted_actual)
    precision = cumulative_tp / np.arange(1, len(actual) + 1)
    recall = cumulative_tp / max(1.0, float(np.sum(actual)))
    n = min(300, len(actual))
    x = np.arange(n)

    fig, axes = plt.subplots(2, 3, figsize=(19, 9), dpi=140)
    axes[0, 0].plot(curves["epoch"], curves["train_loss"], label="train")
    axes[0, 0].plot(curves["epoch"], curves["val_loss"], label="validation")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("weighted binary cross entropy")
    axes[0, 0].set_title("Risk-event learning curve")
    axes[0, 0].legend()

    axes[0, 1].plot(x, probability[-n:], label="predicted event probability")
    axes[0, 1].scatter(
        x,
        actual[-n:],
        s=13,
        alpha=0.55,
        label="realized event",
    )
    axes[0, 1].axhline(validation_threshold, color="black", linestyle="--", label="validation threshold")
    axes[0, 1].set_xlabel("test time index")
    axes[0, 1].set_ylabel("probability / event label")
    axes[0, 1].set_title("Risk probability over time")
    axes[0, 1].legend(fontsize=8)

    axes[0, 2].plot([0, 1], [0, 1], color="black", linestyle="--")
    if not calibration.empty:
        axes[0, 2].plot(
            calibration["mean_probability"],
            calibration["observed_frequency"],
            marker="o",
        )
    axes[0, 2].set_xlabel("mean predicted probability")
    axes[0, 2].set_ylabel("observed event frequency")
    axes[0, 2].set_title("Reliability diagram")

    axes[1, 0].plot(recall, precision)
    axes[1, 0].axhline(float(np.mean(actual)), color="black", linestyle="--")
    axes[1, 0].set_xlabel("recall")
    axes[1, 0].set_ylabel("precision")
    axes[1, 0].set_title("Precision-recall curve")

    axes[1, 1].plot(selective["coverage"], selective["classification_error"], marker="o")
    axes[1, 1].set_xlabel("fraction of samples accepted")
    axes[1, 1].set_ylabel("classification error")
    axes[1, 1].set_title("Selective risk-coverage")

    axes[1, 2].scatter(split["event_score"], probability, s=13, alpha=0.45)
    axes[1, 2].set_xlabel("realized future risk score")
    axes[1, 2].set_ylabel("predicted event probability")
    axes[1, 2].set_title("Probability versus realized risk")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    fig.suptitle(title)
    show_figure(fig)
    obj.display_table(
        f"{title} event preview",
        event_preview_table(split, probability, validation_threshold),
    )


def run_risk_event_case(case, features: pd.DataFrame, profile, args: argparse.Namespace):
    base.set_seed(int(case["seed"]))
    feature_columns = base.FEATURE_SETS[args.feature_set]
    data = build_risk_windows(
        features,
        feature_columns,
        args.seq_len,
        args.horizon,
        str(case["preprocessing"]),
        args.normalization,
        args.max_windows,
        args.stride,
    )
    splits, event_threshold = prepare_event_splits(
        data,
        args.train_ratio,
        args.val_ratio,
        str(case["event_kind"]),
        args.event_quantile,
    )
    batch_size = args.batch_size or profile.batch_size or 32

    while True:
        try:
            model = base.make_model(
                str(case["model"]),
                args.seq_len,
                len(feature_columns),
                int(case["hidden"]),
            )
            param_count = sum(parameter.numel() for parameter in model.parameters())
            model, curves = train_event_model(model, splits, profile, args, batch_size)
            val_probability = predict_event_probability(model, splits["val"], profile, batch_size)
            test_probability = predict_event_probability(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 4:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(4, batch_size // 2)
            print(f"[oom-retry] batch_size {batch_size} -> {next_batch}")
            batch_size = next_batch

    validation_threshold = select_validation_threshold(splits["val"]["y"], val_probability)
    metrics = classification_metrics(splits["test"]["y"], test_probability, validation_threshold)
    calibration = calibration_table(splits["test"]["y"], test_probability)
    selective = selective_risk_curve(splits["test"]["y"], test_probability)
    train_event_rate = float(np.mean(splits["train"]["y"]))
    climatology_brier = float(np.mean((train_event_rate - splits["test"]["y"]) ** 2))
    brier_skill = 1.0 - metrics["brier_score"] / max(1e-12, climatology_brier)
    average_precision_lift = (
        metrics["average_precision"] / metrics["event_rate"]
        if metrics["event_rate"] > 0 and np.isfinite(metrics["average_precision"])
        else float("nan")
    )
    result = {
        "case_id": (
            f"{case['model']}__{case['preprocessing']}__{case['event_kind']}"
            f"__horizon{args.horizon}__seed{case['seed']}"
        ),
        "suite": "risk_event_probe",
        **case,
        "horizon": int(args.horizon),
        "event_quantile": float(args.event_quantile),
        "event_score_threshold": event_threshold,
        "train_event_rate": train_event_rate,
        "validation_probability_threshold": validation_threshold,
        "param_count": int(param_count),
        "batch_size": int(batch_size),
        "expected_calibration_error": expected_calibration_error(calibration),
        "climatology_brier_score": climatology_brier,
        "brier_skill_score": brier_skill,
        "average_precision_lift": average_precision_lift,
        "selective_error_at_50pct": float(
            selective.iloc[(selective["coverage"] - 0.5).abs().argmin()]["classification_error"]
        ),
        **metrics,
    }
    if args.case_plots:
        show_event_diagnostics(
            curves,
            splits["test"],
            test_probability,
            validation_threshold,
            str(result["case_id"]),
        )
    return result


def build_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    models = [value.strip() for value in args.models.split(",") if value.strip()]
    distributions = [value.strip() for value in args.distributions.split(",") if value.strip()]
    preprocessings = [value.strip() for value in args.preprocessings.split(",") if value.strip()]
    event_kinds = [value.strip() for value in args.event_kinds.split(",") if value.strip()]
    widths = [int(value) for value in args.widths.split(",") if value.strip()]
    sample_fractions = [float(value) for value in args.sample_fractions.split(",") if value.strip()]
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    cases: list[dict[str, object]] = []

    if args.suite in {"risk_event_probe", "full"}:
        for preprocessing in preprocessings:
            for model in models:
                for event_kind in event_kinds:
                    for seed in seeds:
                        cases.append(
                            {
                                "suite": "risk_event_probe",
                                "model": model,
                                "event_kind": event_kind,
                                "preprocessing": preprocessing,
                                "seed": seed,
                                "hidden": args.hidden,
                            }
                        )

    if args.suite in {"distribution_probe", "full"}:
        for preprocessing in preprocessings:
            for model in models:
                for distribution in distributions:
                    for seed in seeds:
                        cases.append(
                            {
                                "suite": "distribution_probe",
                                "model": model,
                                "distribution": distribution,
                                "preprocessing": preprocessing,
                                "seed": seed,
                                "hidden": args.hidden,
                            }
                        )

    if args.suite in {"model_capacity_probe", "full"}:
        for model in models:
            for width in widths:
                for seed in seeds:
                    cases.append(
                        {
                            "suite": "model_capacity_probe",
                            "model": model,
                            "preprocessing": preprocessings[0],
                            "seed": seed,
                            "hidden": width,
                            "sample_fraction": 1.0,
                        }
                    )

    if args.suite in {"sample_size_probe", "full"}:
        sample_models = models[:1]
        for model in sample_models:
            for width in widths:
                for fraction in sample_fractions:
                    for seed in seeds:
                        cases.append(
                            {
                                "suite": "sample_size_probe",
                                "model": model,
                                "preprocessing": preprocessings[0],
                                "seed": seed,
                                "hidden": width,
                                "sample_fraction": fraction,
                            }
                        )

    if args.suite in {"epoch_probe", "full"}:
        epoch_widths = [int(value) for value in args.epoch_widths.split(",") if value.strip()]
        for model in models:
            for width in epoch_widths:
                for seed in seeds:
                    cases.append(
                        {
                            "suite": "epoch_probe",
                            "model": model,
                            "preprocessing": preprocessings[0],
                            "seed": seed,
                            "hidden": width,
                            "sample_fraction": 1.0,
                        }
                    )

    if not cases:
        raise ValueError(f"알 수 없는 suite 또는 빈 case 구성: {args.suite}")
    return cases[: args.max_cases] if args.max_cases > 0 else cases


def show_risk_event_summary(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    ordered = summary.sort_values(
        ["average_precision", "brier_score", "expected_calibration_error"],
        ascending=[False, True, True],
    ).reset_index(drop=True)
    columns = [
        "model",
        "preprocessing",
        "event_kind",
        "seed",
        "event_rate",
        "average_precision",
        "average_precision_lift",
        "roc_auc",
        "brier_score",
        "climatology_brier_score",
        "brier_skill_score",
        "expected_calibration_error",
        "precision",
        "recall",
        "f1",
        "classification_error",
        "selective_error_at_50pct",
    ]
    print("\n[risk-event leaderboard]")
    print(ordered[columns].head(30).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=140)
    axes[0].scatter(
        ordered["brier_score"],
        ordered["average_precision"],
        c=ordered["expected_calibration_error"],
        cmap="viridis_r",
    )
    axes[0].set_xlabel("Brier score")
    axes[0].set_ylabel("average precision")
    axes[0].set_title("Discrimination versus calibration")

    grouped = (
        ordered.groupby(["event_kind", "model"], as_index=False)
        .agg(
            average_precision=("average_precision", "mean"),
            baseline_event_rate=("event_rate", "mean"),
        )
        .sort_values("average_precision")
    )
    labels = grouped["event_kind"] + " / " + grouped["model"]
    axes[1].barh(labels, grouped["average_precision"])
    for index, baseline in enumerate(grouped["baseline_event_rate"]):
        axes[1].plot([baseline, baseline], [index - 0.4, index + 0.4], color="black")
    axes[1].set_xlabel("average precision")
    axes[1].set_title("Event ranking versus prevalence baseline")

    axes[2].scatter(
        ordered["event_rate"],
        ordered["predicted_event_rate"],
        c=ordered["f1"],
        cmap="magma",
    )
    axes[2].plot([0, 1], [0, 1], color="black", linestyle="--")
    axes[2].set_xlabel("observed event rate")
    axes[2].set_ylabel("predicted event rate")
    axes[2].set_title("Decision frequency calibration")
    for axis in axes:
        axis.grid(alpha=0.2)
    show_figure(fig)


def show_distribution_summary(summary: pd.DataFrame, alpha: float) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    nominal = 1.0 - alpha
    ordered = summary.sort_values(
        ["copy_risk_ratio", "proper_score", "calibrated_interval_width"]
    ).reset_index(drop=True)
    columns = [
        "model",
        "preprocessing",
        "distribution",
        "seed",
        "copy_risk_ratio",
        "direction_accuracy",
        "variance_ratio",
        "proper_score",
        "raw_interval_coverage",
        "calibrated_interval_coverage",
        "calibrated_interval_width",
    ]
    print("\n[distribution leaderboard]")
    print(ordered[columns].head(30).to_string(index=False))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=140)
    axes[0].scatter(
        ordered["calibrated_interval_width"],
        ordered["calibrated_interval_coverage"],
        c=ordered["copy_risk_ratio"],
        cmap="viridis_r",
    )
    axes[0].axhline(nominal, color="black", linestyle="--")
    axes[0].set_xlabel("mean calibrated interval width")
    axes[0].set_ylabel("empirical coverage")
    axes[0].set_title("Coverage versus sharpness")

    axes[1].scatter(
        ordered["variance_ratio"].clip(1e-3, 1e3),
        ordered["copy_risk_ratio"],
        c=ordered["proper_score"],
        cmap="magma",
    )
    axes[1].set_xscale("log")
    axes[1].axvline(1.0, color="black", linestyle="--")
    axes[1].axhline(1.0, color="black", linestyle="--")
    axes[1].set_xlabel("variance ratio")
    axes[1].set_ylabel("MAE / persistence MAE")
    axes[1].set_title("Point quality versus distribution quality")

    grouped = (
        ordered.groupby("distribution", as_index=False)
        .agg(
            coverage=("calibrated_interval_coverage", "mean"),
            width=("calibrated_interval_width", "mean"),
        )
        .sort_values("coverage")
    )
    axes[2].barh(grouped["distribution"], grouped["coverage"])
    axes[2].axvline(nominal, color="black", linestyle="--")
    axes[2].set_xlabel("mean empirical coverage")
    axes[2].set_title("Coverage by distribution head")
    for axis in axes:
        axis.grid(alpha=0.2)
    show_figure(fig)


def show_capacity_summary(summary: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if summary.empty:
        return
    columns = [
        "suite",
        "model",
        "hidden",
        "sample_fraction",
        "seed",
        "param_count",
        "train_samples",
        "parameter_sample_ratio",
        "best_val_loss",
        "copy_risk_ratio",
        "direction_accuracy",
        "variance_ratio",
    ]
    print("\n[capacity leaderboard]")
    print(summary.sort_values(["copy_risk_ratio", "best_val_loss"])[columns].head(40).to_string(index=False))

    aggregated = (
        summary.groupby(["suite", "model", "hidden", "sample_fraction"], as_index=False)
        .agg(
            param_count=("param_count", "mean"),
            train_samples=("train_samples", "mean"),
            train_loss=("final_train_loss", "mean"),
            val_loss=("best_val_loss", "mean"),
            test_error=("copy_risk_ratio", "mean"),
            test_error_std=("copy_risk_ratio", "std"),
        )
        .sort_values(["model", "sample_fraction", "param_count"])
    )
    fig, axes = plt.subplots(1, 3, figsize=(19, 5), dpi=140)
    for (suite, model, fraction), group in aggregated.groupby(["suite", "model", "sample_fraction"]):
        label = f"{suite} / {model} / n={fraction:.2f}"
        axes[0].plot(group["param_count"], group["train_loss"], marker="o", label=label)
        axes[1].plot(group["param_count"], group["val_loss"], marker="o", label=label)
        axes[2].errorbar(
            group["param_count"],
            group["test_error"],
            yerr=group["test_error_std"].fillna(0.0),
            marker="o",
            label=label,
        )
    axes[0].set_title("Capacity versus train error")
    axes[1].set_title("Capacity versus validation error")
    axes[2].set_title("Capacity versus persistence-relative test error")
    axes[2].axhline(1.0, color="black", linestyle="--")
    for axis in axes:
        axis.set_xscale("log")
        axis.set_xlabel("parameter count")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("final training objective")
    axes[1].set_ylabel("best validation objective")
    axes[2].set_ylabel("test MAE / persistence MAE")
    axes[2].legend(fontsize=7)
    show_figure(fig)


def risk_branch_gate(summary: pd.DataFrame) -> tuple[str, str]:
    if summary.empty:
        return "NOT_EVALUATED", "완료된 위험 이벤트 case가 없다."
    candidates = summary.copy()
    candidates["gate_pass"] = (
        (candidates["brier_skill_score"] > 0.0)
        & (candidates["average_precision_lift"] > 1.0)
        & (candidates["selective_error_at_50pct"] < candidates["classification_error"])
    )
    grouped = (
        candidates.groupby(["model", "preprocessing", "event_kind"], as_index=False)
        .agg(passed_seeds=("gate_pass", "sum"), tested_seeds=("seed", "nunique"))
        .sort_values(["passed_seeds", "tested_seeds"], ascending=False)
    )
    qualified = grouped[(grouped["passed_seeds"] >= 2) & (grouped["tested_seeds"] >= 2)]
    if qualified.empty:
        return (
            "NO_GO",
            "두 개 이상 seed에서 확률 기준선·순위·선택적 거부 기준을 함께 통과한 조합이 없다.",
        )
    winner = qualified.iloc[0]
    return (
        "GO",
        f"{winner['model']} / {winner['preprocessing']} / {winner['event_kind']}가 "
        f"{int(winner['passed_seeds'])}개 seed에서 통과했다.",
    )


def build_inline_report(
    args: argparse.Namespace,
    risk_summary: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    capacity_summary: pd.DataFrame,
) -> str:
    risk_status, risk_reason = risk_branch_gate(risk_summary)
    lines = [
        "# 11번 위험 이벤트·분포 예측 실행 요약",
        "",
        "## 연구 질문",
        "",
        "- 10번의 다음 15분 점예측과 달리, 향후 여러 봉 안의 급변·하방 위험 발생 확률은 예측 가능한가?",
        "- 점예측이 틀릴 수 있는 범위를 모델이 직접 학습하면 변동성이 큰 구간에서 과도한 확신을 줄일 수 있는가?",
        "- Gaussian보다 heavy-tail을 허용하는 Student-t 또는 분포 가정이 약한 quantile head가 업비트 수익률에 더 적합한가?",
        "- 모델 폭, 파라미터 수, 학습 표본 수, epoch가 증가할 때 일반화 오차가 단순 U자형이 아니라 두 번 내려가는 Double Descent 형태를 보이는가?",
        "",
        "## 중요한 구분",
        "",
        "- 분포 예측은 전처리가 아니라 target 출력과 loss를 확률분포로 확장하는 학습 방식이다.",
        "- Double Descent는 전처리가 아니라 모델 용량과 일반화 오차의 관계를 진단하는 실험이다.",
        "- Black-Scholes 공식을 코인 수익률에 직접 적용하지 않는다. 미래를 하나의 값이 아니라 분포로 다룬다는 관점만 가져온다.",
        "- 10번과 11번은 결과 의존성이 없는 경쟁 가설이며, 병렬 GPU slot으로 동시에 실행할 수 있다.",
        "- test epoch 곡선은 현상을 진단하기 위해 표시하지만 모델 선택에는 사용하지 않는다. 선택은 validation 결과만 사용한다.",
        "",
        "## 실행 설정",
        "",
        "```json",
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        "```",
        "",
        "## 위험 이벤트 연구선 판정",
        "",
        f"- 상태: `{risk_status}`",
        f"- 근거: {risk_reason}",
        "- 통과 기준: Brier skill > 0, Average Precision lift > 1, 50% selective error가 전체 오류보다 작다는 조건을 두 개 이상 seed에서 동시에 만족한다.",
    ]
    if not risk_summary.empty:
        top = risk_summary.sort_values(
            ["average_precision", "brier_score", "expected_calibration_error"],
            ascending=[False, True, True],
        ).head(15)
        lines.extend(["", "## 위험 이벤트 상위 결과", "", "```text", top.to_string(index=False), "```"])
    if not distribution_summary.empty:
        top = distribution_summary.sort_values(
            ["copy_risk_ratio", "proper_score", "calibrated_interval_width"]
        ).head(15)
        lines.extend(["", "## 분포 예측 상위 결과", "", "```text", top.to_string(index=False), "```"])
    if not capacity_summary.empty:
        top = capacity_summary.sort_values(["copy_risk_ratio", "best_val_loss"]).head(20)
        lines.extend(["", "## 용량 진단 상위 결과", "", "```text", top.to_string(index=False), "```"])
    lines.extend(
        [
            "",
            "## 해석 기준",
            "",
            "- coverage가 목표 90%에 가까워도 interval width가 지나치게 넓으면 실용적인 불확실성 예측이 아니다.",
            "- interval이 좁아도 실제값을 자주 놓치면 과도하게 자신 있는 모델이다.",
            "- PIT histogram이 대체로 균일해야 likelihood가 분포 전체를 잘 보정했다고 해석한다.",
            "- Double Descent는 폭 한두 개에서 성능이 우연히 흔들린 것과 다르다. 여러 seed에서 train error, validation error, test error의 굴곡이 반복되어야 한다.",
            "- Double Descent가 관찰되어도 가장 큰 모델이 자동으로 최선이라는 뜻은 아니다. persistence, 방향성, 분산, 계산비용을 함께 본다.",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="11번 위험 이벤트·분포 예측 진단")
    parser.add_argument("--db", default=None)
    parser.add_argument("--table", default="btc_15m_advance")
    parser.add_argument("--ticker", default=None)
    parser.add_argument("--profile", default="school_4090_15gb")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument(
        "--parallel-slot",
        choices=sorted(obj.PARALLEL_RESOURCE_SLOTS),
        default="risk_secondary",
        help="10번과 11번을 동시에 실행할 때 사용할 자원 분할 슬롯",
    )
    parser.add_argument(
        "--suite",
        choices=[
            "risk_event_probe",
            "distribution_probe",
            "model_capacity_probe",
            "sample_size_probe",
            "epoch_probe",
            "full",
        ],
        default="risk_event_probe",
    )
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--distributions", default=",".join(DEFAULT_DISTRIBUTIONS))
    parser.add_argument("--preprocessings", default=",".join(DEFAULT_PREPROCESSINGS))
    parser.add_argument("--event-kinds", default=",".join(DEFAULT_EVENT_KINDS))
    parser.add_argument("--horizon", type=int, default=16, help="15분봉 기준 미래 위험 평가 길이")
    parser.add_argument("--event-quantile", type=float, default=0.90)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--widths", default=",".join(map(str, DEFAULT_WIDTHS)))
    parser.add_argument("--epoch-widths", default="32,96,256")
    parser.add_argument("--sample-fractions", default=",".join(map(str, DEFAULT_SAMPLE_FRACTIONS)))
    parser.add_argument("--feature-set", choices=sorted(base.FEATURE_SETS), default="wide_stationary")
    parser.add_argument("--normalization", default="window_standard")
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--scheduler", default="cosine")
    parser.add_argument("--gradient-policy", default="clip1")
    parser.add_argument("--capacity-objective", default="balanced_composite")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--hidden", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--capacity-epochs", type=int, default=30)
    parser.add_argument("--objective-warmup-epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-rows", type=int, default=40000)
    parser.add_argument("--max-windows", type=int, default=4096)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--interval-alpha", type=float, default=0.10)
    parser.add_argument("--case-plots", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-failure", action="store_true")
    return parser.parse_known_args(argv)[0]


def main(argv: list[str] | None = None) -> None:
    configure_inline_matplotlib()
    args = parse_args(argv)
    profile = base.build_resource_profile(args.profile, args.device)
    profile, parallel_settings = obj.apply_parallel_resource_slot(profile, args.parallel_slot)
    base.apply_resource_profile(profile)
    environment = base.log_environment(profile)
    environment["parallel_settings"] = parallel_settings
    print("[environment]", json.dumps(environment, ensure_ascii=False, indent=2))

    cases = build_cases(args)
    print(f"[plan] suite={args.suite} cases={len(cases)}")
    print(pd.DataFrame(cases).head(80).to_string(index=False))
    if args.dry_run:
        return

    db_path = base.resolve_db_path(args.db)
    raw = base.load_price_data(db_path, args.table, args.ticker, args.max_rows)
    features = base.make_features(raw)
    print("[statistics]", json.dumps(base.basic_statistics(features), ensure_ascii=False, indent=2))
    obj.display_table(
        "missingness audit",
        obj.build_missingness_audit(raw, features, "향후 4시간 risk-event / distribution label"),
    )

    risk_rows: list[dict[str, object]] = []
    distribution_rows: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    for index, case in enumerate(cases, start=1):
        print(f"\n[case {index}/{len(cases)}] {case}")
        try:
            if case["suite"] == "risk_event_probe":
                risk_rows.append(run_risk_event_case(case, features, profile, args))
            elif case["suite"] == "distribution_probe":
                distribution_rows.append(run_distribution_case(case, features, profile, args))
            else:
                result, _ = run_capacity_case(case, features, profile, args)
                capacity_rows.append(result)
        except (RuntimeError, ValueError) as exc:
            if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if not args.continue_on_failure:
                raise
            print(f"[case failed] {case}: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    risk_summary = pd.DataFrame(risk_rows)
    distribution_summary = pd.DataFrame(distribution_rows)
    capacity_summary = pd.DataFrame(capacity_rows)
    show_risk_event_summary(risk_summary)
    show_distribution_summary(distribution_summary, args.interval_alpha)
    show_capacity_summary(capacity_summary)
    diag.display_markdown(build_inline_report(args, risk_summary, distribution_summary, capacity_summary))
    print("[done] all diagnostics displayed inline; no PNG/CSV/Markdown artifacts were written.")


if __name__ == "__main__":
    main()
