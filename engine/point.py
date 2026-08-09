"""점예측(point) 학습·평가 엔진.

10번에서 `run_case`와 그 transitive 의존 함수들을 그대로 옮겼다:
objective 구성(`loss_components`/`objective_loss`/`OBJECTIVE_WEIGHTS`/`safe_std`),
학습 루프(`train_model`/`make_loader`), 선택·진단 보조
(`validation_selection_score`/`regime_metrics`), candlestick/preview 시각화
(`draw_candles`/`candle_proxy_arrays`/`show_candlestick_comparison`/
`prediction_preview_table`/`show_case_diagnostics`).

10번의 `base.*`/`diag.*` indirection은 engine.* import로 연결한다. 10번 `run_case`가
`base.apply_window_preprocessing = diag.apply_preprocessing_pipeline`로 하던 전역 monkeypatch는
제거했다(fix②): `engine.windows`가 build_windows의 전처리를 full pipeline으로 기본 고정하므로
run_case가 전역 상태를 mutate할 필요가 없다.
"""

from __future__ import annotations

import argparse
import copy
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from . import display, models, resources, windows


def display_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n[{title}]")
    try:
        from IPython.display import display as _display

        _display(frame)
    except Exception:
        print(frame.to_string(index=False))


def draw_candles(
    ax,
    x: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    width: float,
    up_color: str,
    down_color: str,
    alpha: float,
    linewidth: float = 0.9,
    fill: bool = True,
) -> None:
    from matplotlib.patches import Rectangle

    for xi, o, h, l, c in zip(x, open_, high, low, close):
        color = up_color if c >= o else down_color
        ax.vlines(xi, l, h, color=color, linewidth=linewidth, alpha=alpha)
        lower = min(o, c)
        height = max(abs(c - o), 1e-9)
        rect = Rectangle(
            (xi - width / 2.0, lower),
            width,
            height,
            facecolor=color if fill else "none",
            edgecolor=color,
            linewidth=linewidth,
            alpha=alpha,
        )
        ax.add_patch(rect)


def candle_proxy_arrays(split, pred_return: np.ndarray, n: int) -> dict[str, np.ndarray]:
    prev_close = np.asarray(split["prev_close"][-n:], dtype=np.float32)
    target_close = np.asarray(split["target_close"][-n:], dtype=np.float32)
    target_open = np.asarray(split.get("target_open", prev_close)[-n:], dtype=np.float32)
    target_high_source = np.asarray(split.get("target_high", target_close)[-n:], dtype=np.float32)
    target_low_source = np.asarray(split.get("target_low", target_close)[-n:], dtype=np.float32)
    target_high = np.maximum(target_high_source, np.maximum(target_open, target_close))
    target_low = np.minimum(target_low_source, np.minimum(target_open, target_close))
    predicted_close = prev_close * np.exp(np.asarray(pred_return[-n:], dtype=np.float32))
    predicted_open = prev_close
    predicted_high = np.maximum(predicted_open, predicted_close)
    predicted_low = np.minimum(predicted_open, predicted_close)
    return {
        "timestamp": np.asarray(split["timestamp"][-n:]),
        "actual_open": target_open,
        "actual_high": target_high,
        "actual_low": target_low,
        "actual_close": target_close,
        "pred_open": predicted_open,
        "pred_high": predicted_high,
        "pred_low": predicted_low,
        "pred_close": predicted_close,
        "persistence_close": prev_close,
    }


def show_candlestick_comparison(ax, split, pred_return: np.ndarray, title: str, n: int = 60) -> None:
    from matplotlib.lines import Line2D

    n = min(n, len(pred_return))
    frame = candle_proxy_arrays(split, pred_return, n)
    x = np.arange(n, dtype=float)
    draw_candles(
        ax,
        x - 0.18,
        frame["actual_open"],
        frame["actual_high"],
        frame["actual_low"],
        frame["actual_close"],
        0.32,
        up_color="tab:red",
        down_color="tab:blue",
        alpha=0.65,
    )
    draw_candles(
        ax,
        x + 0.18,
        frame["pred_open"],
        frame["pred_high"],
        frame["pred_low"],
        frame["pred_close"],
        0.28,
        up_color="tab:orange",
        down_color="tab:green",
        alpha=0.85,
        fill=False,
    )
    ax.plot(x, frame["persistence_close"], color="black", linewidth=0.9, alpha=0.55)
    ax.set_title(title)
    ax.set_ylabel("KRW")
    ax.legend(
        handles=[
            Line2D([0], [0], color="tab:red", linewidth=6, alpha=0.65, label="actual candle"),
            Line2D([0], [0], color="tab:orange", linewidth=2, label="predicted proxy candle"),
            Line2D([0], [0], color="black", linewidth=1, alpha=0.55, label="persistence close"),
        ],
        fontsize=8,
        loc="best",
    )


def prediction_preview_table(
    split,
    pred_return: np.ndarray,
    n_rows: int = 12,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> pd.DataFrame:
    n_rows = min(n_rows, len(pred_return))
    frame = candle_proxy_arrays(split, pred_return, n_rows)
    actual_return = split["y"][-n_rows:]
    pred_slice = pred_return[-n_rows:]
    table = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "actual_open": np.round(frame["actual_open"], 0),
            "actual_high": np.round(frame["actual_high"], 0),
            "actual_low": np.round(frame["actual_low"], 0),
            "actual_close": np.round(frame["actual_close"], 0),
            "pred_open_proxy": np.round(frame["pred_open"], 0),
            "pred_close_proxy": np.round(frame["pred_close"], 0),
            "persistence_close": np.round(frame["persistence_close"], 0),
            "actual_return": np.round(actual_return, 6),
            "predicted_return": np.round(pred_slice, 6),
        }
    )
    if lower is not None and upper is not None:
        table["return_lower"] = np.round(lower[-n_rows:], 6)
        table["return_upper"] = np.round(upper[-n_rows:], 6)
    return table


def safe_std(values: torch.Tensor) -> torch.Tensor:
    return values.std(unbiased=False).clamp_min(1e-6)


def loss_components(pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
    huber_each = F.smooth_l1_loss(pred, target, beta=0.001, reduction="none")
    huber = huber_each.mean()

    target_sign = torch.sign(target)
    directional = F.softplus(-pred * target_sign * 100.0).mean()

    target_std = safe_std(target.detach())
    pred_std = safe_std(pred)
    variance_ratio = pred_std / target_std
    variance = torch.abs(torch.log(variance_ratio.clamp(0.05, 20.0)))

    pred_centered = pred - pred.mean()
    target_centered = target - target.mean()
    correlation = (pred_centered * target_centered).mean() / (
        safe_std(pred_centered) * safe_std(target_centered)
    )
    correlation_penalty = 1.0 - correlation.clamp(-1.0, 1.0)

    scale = target.detach().abs().median().clamp_min(1e-5)
    tail_weight = 1.0 + 2.0 * torch.clamp(target.detach().abs() / (3.0 * scale), 0.0, 3.0)
    tail = (tail_weight * huber_each).mean()

    volatility = target.detach().abs()
    threshold = torch.quantile(volatility, 0.75)
    high_mask = volatility >= threshold
    low_mask = ~high_mask
    high_loss = huber_each[high_mask].mean() if high_mask.any() else huber
    low_loss = huber_each[low_mask].mean() if low_mask.any() else huber
    regime = 0.5 * (high_loss + low_loss)

    near_zero = torch.exp(-pred.abs() / 0.00025).mean()
    mean_bias = torch.abs(pred.mean() - target.mean()) / target_std
    return {
        "huber": huber,
        "direction": directional,
        "variance": variance,
        "correlation": correlation_penalty,
        "tail": tail,
        "regime": regime,
        "near_zero": near_zero,
        "mean_bias": mean_bias,
    }


OBJECTIVE_WEIGHTS = {
    "huber": {},
    "directional_huber": {"direction": 0.20},
    "variance_huber": {"variance": 0.12},
    "correlation_huber": {"correlation": 0.10},
    "tail_huber": {"tail": 0.30},
    "regime_huber": {"regime": 0.30},
    "anti_collapse_v2": {
        "direction": 0.10,
        "variance": 0.10,
        "near_zero": 0.05,
    },
    "balanced_composite": {
        "direction": 0.08,
        "variance": 0.08,
        "correlation": 0.06,
        "mean_bias": 0.03,
    },
}


def objective_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    objective_name: str,
    auxiliary_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if objective_name not in OBJECTIVE_WEIGHTS:
        raise ValueError(f"지원하지 않는 objective: {objective_name}")
    components = loss_components(pred, target)
    huber = components["huber"]
    total = huber
    for name, weight in OBJECTIVE_WEIGHTS[objective_name].items():
        component = components[name]
        # Auxiliary terms are dimensionless or differently scaled. Match their
        # detached magnitude to Huber so no single term dominates by units alone.
        magnitude_match = huber.detach() / component.detach().abs().clamp_min(1e-6)
        total = total + auxiliary_scale * weight * magnitude_match * component
    return total, components


def make_loader(split, batch_size: int, shuffle: bool, profile):
    return models.make_loader(split, batch_size, shuffle, profile)


def train_model(
    model,
    splits,
    profile,
    args: argparse.Namespace,
    objective_name: str,
    batch_size: int,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    device = torch.device(profile.device)
    model = model.to(device)
    train_loader = make_loader(splits["train"], batch_size, True, profile)
    val_loader = make_loader(splits["val"], batch_size, False, profile)
    optimizer = models.make_optimizer(model, args.optimizer, args.lr, args.weight_decay)
    scheduler = models.make_scheduler(optimizer, args.scheduler, args.epochs, len(train_loader), args.lr)
    best_state = None
    best_val = float("inf")
    patience_left = args.patience
    rows: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        auxiliary_scale = min(1.0, max(0.0, (epoch - 1) / max(1, args.objective_warmup_epochs)))
        model.train()
        train_totals: list[float] = []
        train_gradients: list[float] = []
        component_sums: defaultdict[str, list[float]] = defaultdict(list)
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss, components = objective_loss(pred, yb, objective_name, auxiliary_scale)
            loss.backward()
            train_gradients.append(models.apply_gradient_policy(model, args.gradient_policy, epoch))
            optimizer.step()
            if args.scheduler == "onecycle" and scheduler is not None:
                scheduler.step()
            train_totals.append(float(loss.detach().cpu()))
            for name, value in components.items():
                component_sums[name].append(float(value.detach().cpu()))

        model.eval()
        val_totals: list[float] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                loss, _ = objective_loss(model(xb), yb, objective_name, auxiliary_scale)
                val_totals.append(float(loss.detach().cpu()))

        train_loss = float(np.mean(train_totals))
        val_loss = float(np.mean(val_totals))
        if scheduler is not None and args.scheduler == "plateau":
            scheduler.step(val_loss)
        elif scheduler is not None and args.scheduler not in {"onecycle", "plateau"}:
            scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "grad_norm_mean": float(np.mean(train_gradients)),
            "grad_norm_max": float(np.max(train_gradients)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "auxiliary_scale": auxiliary_scale,
        }
        for name, values in component_sums.items():
            row[f"component_{name}"] = float(np.mean(values))
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


def validation_selection_score(metrics: dict[str, float]) -> float:
    copy_term = math.log1p(max(0.0, metrics["copy_risk_ratio"]))
    variance_term = abs(math.log(max(1e-4, min(1e4, metrics["variance_ratio"]))))
    direction_term = max(0.0, 0.50 - metrics["direction_accuracy"]) * 4.0
    zero_term = metrics["near_zero_return_share"]
    return float(copy_term + 0.35 * variance_term + direction_term + zero_term)


def regime_metrics(split, prediction: np.ndarray) -> dict[str, float]:
    actual = split["y"]
    absolute_return = np.abs(actual)
    threshold = float(np.quantile(absolute_return, 0.75))
    high_mask = absolute_return >= threshold
    low_mask = ~high_mask
    error = np.abs(prediction - actual)
    return {
        "high_vol_return_mae": float(np.mean(error[high_mask])),
        "low_vol_return_mae": float(np.mean(error[low_mask])),
        "high_vol_direction_accuracy": float(
            np.mean((prediction[high_mask] > 0) == (actual[high_mask] > 0))
        ),
        "low_vol_direction_accuracy": float(
            np.mean((prediction[low_mask] > 0) == (actual[low_mask] > 0))
        ),
    }


def run_case(case, features: pd.DataFrame, profile, args: argparse.Namespace):
    models.set_seed(int(case["seed"]))
    feature_columns = resources.FEATURE_SETS[args.feature_set]
    # fix②: 전역 mutation 제거. windows.build_windows가 이미 full pipeline을 기본으로 쓴다.
    data = windows.build_windows(
        features,
        feature_columns,
        args.seq_len,
        str(case["preprocessing"]),
        args.normalization,
        args.max_windows,
        args.stride,
    )
    splits = windows.time_split(data, args.train_ratio, args.val_ratio)
    batch_size = args.batch_size or profile.batch_size or 32

    while True:
        try:
            model = models.make_model(str(case["model"]), args.seq_len, len(feature_columns), args.hidden)
            param_count = sum(parameter.numel() for parameter in model.parameters())
            model, curves = train_model(model, splits, profile, args, str(case["objective"]), batch_size)
            val_pred = resources.predict(model, splits["val"], profile, batch_size)
            test_pred = resources.predict(model, splits["test"], profile, batch_size)
            break
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or batch_size <= 4:
                raise
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            next_batch = max(4, batch_size // 2)
            print(f"[oom-retry] batch_size {batch_size} -> {next_batch}")
            batch_size = next_batch

    val_metrics = resources.evaluate_predictions(splits["val"], val_pred)
    test_metrics = resources.evaluate_predictions(splits["test"], test_pred)
    result = {
        "case_id": (
            f"{case['model']}__{case['preprocessing']}__{case['objective']}__seed{case['seed']}"
        ),
        "model": case["model"],
        "preprocessing": case["preprocessing"],
        "objective": case["objective"],
        "seed": int(case["seed"]),
        "param_count": int(param_count),
        "batch_size": int(batch_size),
        "val_selection_score": validation_selection_score(val_metrics),
        "val_copy_risk_ratio": val_metrics["copy_risk_ratio"],
        "val_variance_ratio": val_metrics["variance_ratio"],
        "val_direction_accuracy": val_metrics["direction_accuracy"],
        **test_metrics,
        **regime_metrics(splits["test"], test_pred),
    }
    show_case_diagnostics(curves, splits["test"], test_pred, result)
    return result, val_pred, test_pred, splits


def show_case_diagnostics(curves: pd.DataFrame, test_split, pred: np.ndarray, result: dict[str, object]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(19, 8), dpi=140)
    axes[0, 0].plot(curves["epoch"], curves["train_loss"], label="train")
    axes[0, 0].plot(curves["epoch"], curves["val_loss"], label="validation")
    axes[0, 0].set_title("Total objective")
    axes[0, 0].legend()

    component_columns = [column for column in curves if column.startswith("component_")]
    for column in component_columns:
        axes[0, 1].plot(curves["epoch"], curves[column], label=column.replace("component_", ""))
    axes[0, 1].set_title("Objective components")
    axes[0, 1].legend(fontsize=8, ncol=2)

    axes[0, 2].plot(curves["epoch"], curves["grad_norm_mean"], label="mean")
    axes[0, 2].plot(curves["epoch"], curves["grad_norm_max"], label="max")
    axes[0, 2].set_title("Gradient norm")
    axes[0, 2].legend()

    n = min(200, len(pred))
    actual_return = test_split["y"][-n:]
    predicted_return = pred[-n:]
    axes[1, 0].plot(actual_return, label="actual return")
    axes[1, 0].plot(predicted_return, label="predicted return")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_title("Return prediction")
    axes[1, 0].legend()

    show_candlestick_comparison(
        axes[1, 1],
        test_split,
        pred,
        "Next-candle comparison",
        n=min(60, len(pred)),
    )

    limit = max(float(np.max(np.abs(actual_return))), float(np.max(np.abs(predicted_return))), 1e-4)
    axes[1, 2].scatter(actual_return, predicted_return, s=13, alpha=0.5)
    axes[1, 2].plot([-limit, limit], [-limit, limit], color="black", linewidth=0.8)
    pearson = float(np.corrcoef(actual_return, predicted_return)[0, 1])
    axes[1, 2].set_xlim(-limit, limit)
    axes[1, 2].set_ylim(-limit, limit)
    axes[1, 2].set_xlabel("actual return")
    axes[1, 2].set_ylabel("predicted return")
    axes[1, 2].set_title(f"Calibration scatter / Pearson={pearson:.3f}")
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(str(result["case_id"]))
    display.show_figure(fig)
    display_table(
        f"{result['case_id']} prediction preview",
        prediction_preview_table(test_split, pred),
    )
