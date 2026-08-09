"""위험 이벤트(분류) 학습·예측·진단.

11번에서 옮겼다. event 모델 학습/예측, 임계값 선택, 분류 지표, calibration,
selective risk, 진단 시각화를 담는다. 보조 helper(`make_tensor_loader`,
`binary_cross_entropy`, `average_precision`, `roc_auc`, `event_preview_table`)도 함께 옮겼다.

11번의 `base.*` indirection은 engine.models로, `obj.display_table`은 engine.display로 연결한다.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, TensorDataset

from . import display, models


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
    optimizer = models.make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = models.make_scheduler(optimizer, args.scheduler, args.epochs, len(train_loader), args.lr)
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
            gradients.append(models.apply_gradient_policy(model, args.gradient_policy, epoch))
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
    display.show_figure(fig)
    display.display_table(
        f"{title} event preview",
        event_preview_table(split, probability, validation_threshold),
    )
