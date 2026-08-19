# %% [markdown]
# # 18f: 지표 유효성 — R²가 무효라면 무엇이 유효한가
#
# ## 왜 (Why)
# 사용자 지적(2026-08-19): "결정계수가 지표로 유효하지 않다고 했는데 그럼 뭐가 유효한지,
# 실제로 여러 지표를 적용해보니 내가 분석하려는 다양한 상황에서 각각 어떤 지표가 유효한지
# 전부 시각화 또는 결과 표로 정리해서 보고서에 넣어라. MSE만 보면 오차의 제곱이라 얼마인지
# 직관이 안 오는데 MAE를 보여주면 직관적이잖아. 다른 성능 평가도 추가하라고 했는데 왜 안 하나."
#
# 앞선 18번은 "레벨축 R²는 무모델과 실모델을 구별하지 못한다"만 보였다. **그러면 무엇을
# 써야 하는가**에 대한 답이 없었다. 이 드라이버는 지표를 상황별로 전수 적용해 그 답을 만든다.
#
# ## 무엇을 (What)
# **판정 방식**: 각 상황에서 "무모델(naive)"과 "실모델"을 둘 다 평가한다. 좋은 지표는 이 둘을
# **분리**해야 한다. 분리하지 못하면(값이 사실상 같으면) 그 상황에서 그 지표는 무효다.
#
#   상황 A 레벨축 가격 예측    : 다음 로그가격을 맞힌다 (무모델 = 직전값 복사)
#   상황 B 차분축 수익률 예측  : 다음 수익률을 맞힌다 (무모델 = 항상 0)
#   상황 C 방향 예측           : 부호를 맞힌다 (무모델 = 항상 상승)
#   상황 D 변동성 예측         : 다음 분산을 맞힌다 (무모델 = 학습구간 상수분산)
#
# 지표 11종: MSE · RMSE · MAE · MAPE · R² · 상관 · MASE · DA · variance_ratio · copy_risk · QLIKE
#
# ## 어떻게 (How)
# 분리도 = |실모델 − 무모델| / (무모델의 척도). 이것이 임계값 미만이면 "구별 못 함(무효)"으로
# 판정한다. 판정은 상태색 + **텍스트 라벨 병기**로 표기한다(색 단독으로 의미를 나르지 않는다).
#
# ## 기대 결과 / 반영 (Expected)
# "상황 × 지표" 유효성 표를 만들어, 앞으로 어떤 실험에서 어떤 지표를 봐야 하는지 고정한다.

# %%
"""18f 지표 유효성 검증.

실행:
    uv run test/models/18f_metric_validity.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.useoffset"] = False
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.color"] = "#e8e8e6"
matplotlib.rcParams["grid.linewidth"] = 0.7
matplotlib.rcParams["axes.edgecolor"] = "#c9c9c6"

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine import preflight  # noqa: E402

C_A, C_B = "#2a78d6", "#eb6834"
S_GOOD, S_WARN, S_CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, INK2 = "#0b0b0b", "#52514e"

TAG = "18f_metric_validity_20260819"
SOURCE_TABLE = "upbit_krw_candle"
BASE_TICKER = "KRW-BTC"
BARS_PER_DAY = 96
SEPARATION_THRESHOLD = 0.02      # 상대 분리도 2% 미만이면 "구별 못 함"

_LINES: list[str] = []


def emit(text: str = "") -> None:
    print(text, flush=True)
    _LINES.append(text)


def num(v, d: int = 4) -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "n/a" if not np.isfinite(f) else f"{f:,.{d}f}"


def _root() -> Path:
    start = Path(__file__).resolve().parent
    for c in [start, *start.parents]:
        if (c / "pyproject.toml").exists() and (c / "engine").is_dir():
            return c
    raise RuntimeError("root 못 찾음")


ROOT = _root()
IMAGES_DIR = ROOT / "test" / "images" / TAG
RESULTS_DIR = ROOT / "test" / "results" / TAG


# %% [markdown]
# ## 지표 정의 — 각각 무엇을 재고, 왜 직관적이거나 아닌가
#
# | 지표 | 정의 | 단위 | 직관 |
# | :--- | :--- | :--- | :--- |
# | MSE | 오차²의 평균 | 원자료 단위의 **제곱** | 낮음 — 제곱이라 크기를 못 읽는다 |
# | RMSE | √MSE | 원자료 단위 | 중간 — 단위는 맞지만 큰 오차에 민감 |
# | MAE | \|오차\|의 평균 | 원자료 단위 | **높음** — "평균 몇 원/몇 % 틀린다" |
# | MAPE | \|오차/실제\|의 평균 | % | 높음 — 단 실제값이 0 근처면 폭발 |
# | R² | 1 − 오차²합/전체분산 | 없음 | 낮음(상황 의존) — 분모가 크면 자동으로 1에 가까워진다 |
# | 상관 | 예측·실제의 선형 동조 | 없음 | 중간 — 크기 무시, 방향성만 |
# | MASE | MAE / 무모델 MAE | 없음 | **높음** — 1 미만이면 무모델을 이겼다 |
# | DA | 부호 일치 비율 | % | **높음** — 0.5가 동전던지기 |
# | variance_ratio | 예측 표준편차 / 실제 표준편차 | 없음 | 높음 — 1보다 작으면 진폭 압축 |
# | copy_risk | MAE / 직전값복사 MAE | 없음 | **높음** — 1 근처면 복사와 다를 게 없다 |
# | QLIKE | log σ² + r²/σ² 의 평균 | 없음 | 분산 예측 전용 — 잡음 대리변수에 강건 |

# %%
def metrics_point(y: np.ndarray, pred: np.ndarray, naive: np.ndarray) -> dict:
    """수준·변화량 예측용 지표 묶음."""
    err = y - pred
    naive_err = y - naive
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    denominator = np.where(np.abs(y) < 1e-12, np.nan, y)
    return {
        "MSE": float(np.mean(err**2)),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "MAE": float(np.mean(np.abs(err))),
        "MAPE(%)": float(np.nanmean(np.abs(err / denominator)) * 100),
        "R^2": 1.0 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else np.nan,
        "상관": float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 1e-15 else np.nan,
        "MASE": float(np.mean(np.abs(err)) / max(np.mean(np.abs(naive_err)), 1e-18)),
        "variance_ratio": float(np.std(pred) / max(np.std(y), 1e-18)),
    }


def metrics_direction(y: np.ndarray, pred: np.ndarray, naive: np.ndarray) -> dict:
    return {
        "DA": float(np.mean((pred > 0) == (y > 0))),
        "DA(큰변동 상위25%)": float(
            np.mean((pred[np.abs(y) >= np.quantile(np.abs(y), 0.75)] > 0)
                    == (y[np.abs(y) >= np.quantile(np.abs(y), 0.75)] > 0))
        ),
    }


def metrics_variance(realized: np.ndarray, forecast: np.ndarray, naive: np.ndarray) -> dict:
    mask = np.isfinite(forecast) & (forecast > 0) & np.isfinite(naive) & (naive > 0)
    r, f, n = realized[mask], forecast[mask], naive[mask]
    return {
        "QLIKE": float(np.mean(np.log(f) + r / f)),
        "MSE": float(np.mean((r - f) ** 2)),
        "MAE": float(np.mean(np.abs(r - f))),
        "R^2": 1.0 - float(np.sum((r - f) ** 2)) / float(np.sum((r - r.mean()) ** 2)),
        "상관(√예측 vs |r|)": float(np.corrcoef(np.sqrt(f), np.sqrt(r))[0, 1]),
        "MASE": float(np.mean(np.abs(r - f)) / max(np.mean(np.abs(r - n)), 1e-18)),
    }


# %%
def build_situations(frame: pd.DataFrame) -> dict:
    lp = frame["log_close"].to_numpy(float)
    returns = np.diff(lp)
    n = len(lp)
    t = np.arange(1, n - 1)
    split = int(len(t) * 0.7)
    train, test = slice(0, split), slice(split, len(t))

    y_level = lp[t + 1]
    y_return = lp[t + 1] - lp[t]
    naive_level = lp[t]
    naive_return = np.zeros_like(y_return)

    slope, intercept = np.polyfit(returns[t - 1][train], y_return[train], 1)
    pred_return = intercept + slope * returns[t - 1]
    pred_level = lp[t] + pred_return

    # 상황 D: 변동성 — 롤링분산(실모델 대리) vs 상수분산(무모델)
    squared = pd.Series(returns).pow(2)
    rolling = squared.rolling(BARS_PER_DAY).mean().shift(1).to_numpy()
    realized = y_return**2
    const = np.full(len(y_return), float(np.var(returns[:split])))

    return {
        "A. 레벨축 가격 예측": {
            "kind": "point",
            "y": y_level[test], "pred": pred_level[test], "naive": naive_level[test],
            "naive_name": "직전값 복사", "model_name": "수익률 AR(1) → 레벨 환산",
            "unit": "log 가격",
        },
        "B. 차분축 수익률 예측": {
            "kind": "point",
            "y": y_return[test], "pred": pred_return[test], "naive": naive_return[test],
            "naive_name": "항상 0", "model_name": "수익률 AR(1)",
            "unit": "로그수익률",
        },
        "C. 방향 예측": {
            "kind": "direction",
            "y": y_return[test], "pred": pred_return[test],
            "naive": np.full(len(y_return[test]), 1e-9),
            "naive_name": "항상 상승", "model_name": "수익률 AR(1) 부호",
            "unit": "부호",
        },
        "D. 변동성 예측": {
            "kind": "variance",
            "y": realized[test], "pred": rolling[t][test], "naive": const[test],
            "naive_name": "상수분산", "model_name": f"롤링분산({BARS_PER_DAY}봉)",
            "unit": "분산",
        },
    }


def evaluate(situations: dict) -> pd.DataFrame:
    records = []
    for name, spec in situations.items():
        if spec["kind"] == "point":
            model = metrics_point(spec["y"], spec["pred"], spec["naive"])
            naive = metrics_point(spec["y"], spec["naive"], spec["naive"])
            model.update(metrics_direction(spec["y"], spec["pred"], spec["naive"]))
            naive.update(metrics_direction(spec["y"], spec["naive"] * 0 + 1e-9, spec["naive"]))
            model["copy_risk"] = model["MAE"] / max(naive["MAE"], 1e-18)
            naive["copy_risk"] = 1.0
        elif spec["kind"] == "direction":
            model = metrics_direction(spec["y"], spec["pred"], spec["naive"])
            naive = metrics_direction(spec["y"], spec["naive"], spec["naive"])
        else:
            model = metrics_variance(spec["y"], spec["pred"], spec["naive"])
            naive = metrics_variance(spec["y"], spec["naive"], spec["naive"])
        for metric in model:
            m_val, n_val = model[metric], naive.get(metric, np.nan)
            scale = max(abs(n_val), abs(m_val), 1e-18)
            separation = abs(m_val - n_val) / scale if np.isfinite(n_val) else np.nan
            records.append({
                "상황": name, "지표": metric,
                "무모델": n_val, "실모델": m_val,
                "분리도": separation,
                "판정": ("무효 — 구별 못 함" if np.isfinite(separation) and separation < SEPARATION_THRESHOLD
                        else "주의 — 약한 분리" if np.isfinite(separation) and separation < 0.10
                        else "유효 — 분리됨"),
            })
    return pd.DataFrame(records)


# %%
def report(table: pd.DataFrame, situations: dict) -> None:
    emit("# 18f 지표 유효성 원시 수치 (자동 생성)")
    emit()
    emit(f"- 판정 규칙: 분리도 = |실모델 − 무모델| / max(|실모델|, |무모델|). "
         f"**{SEPARATION_THRESHOLD:.0%} 미만이면 무효**(그 상황에서 이 지표는 실력과 무모델을 구별하지 못한다), "
         "10% 미만은 주의.")
    emit()
    emit("**주의 — 분리도는 지표 값 자체가 아니다.** 예를 들어 R²는 정의상 1(100%)을 넘을 수 없지만, ")
    emit("아래 표의 '분리도' 열은 100%를 넘는 경우가 있다(예: 112%). 이는 R²가 112%라는 뜻이 아니라, ")
    emit("무모델·실모델 R² 두 값이 **둘 다 0 근처이고 부호가 다를 때** `|차이|/분모`가 1을 넘을 수 있어서다 ")
    emit("(예: 무모델 −0.00006, 실모델 +0.00046 → 분리도 112%. R² 실제 값은 항상 −∞~1 사이에 있다). ")
    emit("**분리도는 '구별 가능성'을 재는 별도 지표이고, 원래 지표의 스케일과는 무관하다.**")
    emit()
    for name, spec in situations.items():
        emit(f"## {name}")
        emit()
        emit(f"- 맞히는 대상: {spec['unit']} · 무모델: {spec['naive_name']} · 실모델: {spec['model_name']}")
        emit(f"- 평가 표본: {len(spec['y']):,}행")
        emit()
        emit("| 지표 | 무모델 | 실모델 | 분리도 | 판정 |")
        emit("| :--- | ---: | ---: | ---: | :--- |")
        for _, row in table[table["상황"] == name].iterrows():
            digits = 6 if abs(row["실모델"]) < 0.01 else 4
            emit(f"| {row['지표']} | {num(row['무모델'], digits)} | {num(row['실모델'], digits)} | "
                 f"{num(row['분리도'] * 100, 1)}% | {row['판정']} |")
        emit()

    emit("## 종합 — 상황별로 무엇을 써야 하는가")
    emit()
    pivot = table.pivot_table(index="지표", columns="상황", values="분리도", aggfunc="first")
    emit("| 지표 | " + " | ".join(pivot.columns) + " |")
    emit("| :--- | " + " | ".join(["---:"] * len(pivot.columns)) + " |")
    for metric, row in pivot.iterrows():
        cells = []
        for value in row:
            if not np.isfinite(value):
                cells.append("—")
            elif value < SEPARATION_THRESHOLD:
                cells.append(f"✗ {value*100:.1f}%")
            elif value < 0.10:
                cells.append(f"△ {value*100:.1f}%")
            else:
                cells.append(f"✓ {value*100:.1f}%")
        emit(f"| {metric} | " + " | ".join(cells) + " |")
    emit()
    emit("✓ 유효(분리 10%+) · △ 주의(2~10%) · ✗ 무효(2% 미만). 기호는 색과 별개로 판정을 나른다.")
    emit()


def figure(table: pd.DataFrame) -> None:
    pivot = table.pivot_table(index="지표", columns="상황", values="분리도", aggfunc="first")
    order = ["MSE", "RMSE", "MAE", "MAPE(%)", "R^2", "상관", "상관(√예측 vs |r|)",
             "MASE", "copy_risk", "variance_ratio", "DA", "DA(큰변동 상위25%)", "QLIKE"]
    pivot = pivot.reindex([m for m in order if m in pivot.index])

    fig, axes = plt.subplots(1, 2, figsize=(19, 7.2), gridspec_kw={"width_ratios": [1.15, 1]})

    ax = axes[0]
    display = pivot.fillna(-1).to_numpy() * 100
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([c.split(". ")[1] if ". " in c else c for c in pivot.columns],
                       rotation=18, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.set_xlim(-0.5, len(pivot.columns) - 0.5)
    ax.set_ylim(len(pivot.index) - 0.5, -0.5)
    ax.grid(False)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = display[i, j]
            if value < 0:
                color, mark = "#f0efec", "—"
            elif value < SEPARATION_THRESHOLD * 100:
                color, mark = S_CRIT, f"✗ {value:.1f}%"
            elif value < 10:
                color, mark = S_WARN, f"△ {value:.1f}%"
            else:
                color, mark = S_GOOD, f"✓ {value:.0f}%"
            ax.add_patch(plt.Rectangle((j - 0.46, i - 0.44), 0.92, 0.88,
                                       facecolor=color, edgecolor="white", lw=2, alpha=0.9))
            ax.text(j, i, mark, ha="center", va="center", fontsize=9.5, color="white"
                    if value >= 0 else INK2, weight="bold")
    ax.set_title("상황 × 지표 유효성 — 셀 값은 '실모델과 무모델의 분리도'\n"
                 "✓ 유효(10%+) · △ 주의(2~10%) · ✗ 무효(2% 미만) · — 해당 없음", fontsize=12)

    ax = axes[1]
    # 축을 바꾸면 같은 지표의 분리력이 어떻게 달라지는가 — 지표 선택보다 축 선택이 먼저다
    shared = ["MSE", "RMSE", "MAE", "MASE", "copy_risk", "R^2", "variance_ratio"]
    a_key, b_key = "A. 레벨축 가격 예측", "B. 차분축 수익률 예측"
    a_values, b_values, labels = [], [], []
    for metric in shared:
        row_a = table[(table["상황"] == a_key) & (table["지표"] == metric)]
        row_b = table[(table["상황"] == b_key) & (table["지표"] == metric)]
        if row_a.empty or row_b.empty:
            continue
        labels.append(metric)
        a_values.append(float(row_a["분리도"].iloc[0]) * 100)
        b_values.append(float(row_b["분리도"].iloc[0]) * 100)
    x = np.arange(len(labels))
    width = 0.38
    ax.bar(x - width / 2, a_values, width, label="레벨축에서 재면", color=C_A)
    ax.bar(x + width / 2, b_values, width, label="차분축에서 재면", color=C_B)
    ax.axhline(SEPARATION_THRESHOLD * 100, color=S_CRIT, ls="--", lw=1.3)
    ax.text(-0.42, SEPARATION_THRESHOLD * 100 * 1.5, "이 선 아래 = 무효(2% 미만)",
            fontsize=9.5, color=S_CRIT, ha="left", va="bottom")
    for xi, (a_value, b_value) in enumerate(zip(a_values, b_values)):
        for offset, value in [(-width / 2, a_value), (width / 2, b_value)]:
            ax.text(xi + offset, max(value, 0.05) * 1.15, f"{value:.1f}", ha="center",
                    va="bottom", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.5, rotation=20, ha="right")
    ax.set_ylabel("분리도 (%, 로그축)")
    ax.set_title("같은 지표라도 '어느 축에서 재는가'로 분리력이 갈린다\n"
                 "레벨축은 전 지표가 무효 · 차분축에서는 R²·variance_ratio가 살아난다", fontsize=12)
    ax.legend(fontsize=10, frameon=False, loc="upper center", ncol=2)
    ax.grid(True, axis="y")

    fig.suptitle("18f. R²가 유효하지 않다면 무엇이 유효한가 — 상황마다 다르다", fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(IMAGES_DIR / "metric_validity_matrix.png", dpi=115)
    plt.close(fig)


# %%
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="18f 지표 유효성")
    parser.add_argument("--ticker", default=BASE_TICKER)
    args = parser.parse_args(argv)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preflight.ensure_data(table=SOURCE_TABLE, tickers=[args.ticker])
    con = duckdb.connect(str(ROOT / "data" / "upbit_data.db"), read_only=True)
    try:
        frame = con.execute(
            f"select timestamp, close from {SOURCE_TABLE} where ticker = ? order by timestamp",
            [args.ticker],
        ).df()
    finally:
        con.close()
    frame["log_close"] = np.log(frame["close"].astype(float))

    situations = build_situations(frame)
    table = evaluate(situations)
    report(table, situations)
    figure(table)

    table.to_csv(RESULTS_DIR / "metric_validity.csv", index=False, encoding="utf-8")
    (RESULTS_DIR / "metric_validity_raw.md").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[18f] 저장: {RESULTS_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
