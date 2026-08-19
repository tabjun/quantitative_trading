# %% [markdown]
# # 19c: B 갈래 모델 선택을 전종목으로 확인 — BTC 하나로 정한 게 아니다
#
# ## 왜 (Why)
# 사용자 지적(2026-08-19): "모든 종목에 대한 분석을 다 원하는데, `preflight.ensure_data`
# 예시가 KRW-BTC만 있어서 다시 BTC 단일종목 문제로 돌아간 거 아닌가?"
#
# 확인해보니 실제로 절반은 맞는 지적이었다. 18d(전처리 사다리)·18e(설명용 그림)·18f(지표
# 유효성)는 **개념 설명용**이라 BTC 하나로 충분하다(269종목에 같은 메커니즘 그림을 269번
# 그려도 새 정보가 없다). 그런데 **19번(B 갈래 모델 선택 — GARCH-t vs HAR-RV)은 BTC 하나로만
# 결론 냈다.** "4시간이면 GARCH-t, 16시간이면 HAR-RV"가 BTC 특정 현상인지 구조적인지 확인이
# 안 됐다 — 정확히 지적받은 지점이다.
#
# ## 무엇을 (What)
# 19번의 삼원 격자(모델×주기제거×horizon)를 **유동성 상위 20종목**(16번 t3·18번 D6와 같은
# 표본)에 반복한다. 결과는 "종목마다 최우수 모델"의 분포로 집계한다 — 개별 QLIKE 수치가
# 아니라 **승자의 일관성**이 이 실험의 산출물이다.
#
# ## 어떻게 (How)
# 19번의 `run_condition`을 그대로 재사용한다(재구현하면 두 실험의 수치가 갈린다). 주기 제거
# 조건만 쓴다(19번에서 18/18셀 전부 이겼으므로 이미 결정됐다). 비용 통제를 위해 GARCH 시작점을
# 19번보다 줄인다.
#
# ## 기대 결과 / 반영 (Expected)
# "target horizon별 최우수 모델"이 BTC 특정이 아니라 몇 종목에서 재현되는지 확정한다.

# %%
"""19c 모델 선택 전종목 확인.

실행:
    uv run test/models/19c_model_selection_crosssection.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.color"] = "#e8e8e6"
matplotlib.rcParams["axes.edgecolor"] = "#c9c9c6"

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine import preflight  # noqa: E402

C_A, C_B, C_C = "#2a78d6", "#eb6834", "#1baf7a"
S_GOOD, S_WARN, S_CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, INK2 = "#0b0b0b", "#52514e"
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

TAG = "19c_model_selection_crosssection_20260819"
SOURCE_TABLE = "upbit_krw_candle"
TOP_N = 20
HORIZONS = [1, 16, 64]

RESULTS_DIR = None
IMAGES_DIR = None


def _root() -> Path:
    start = Path(__file__).resolve().parent
    for c in [start, *start.parents]:
        if (c / "pyproject.toml").exists() and (c / "engine").is_dir():
            return c
    raise RuntimeError("root 못 찾음")


ROOT = _root()
IMAGES_DIR = ROOT / "test" / "images" / TAG
RESULTS_DIR = ROOT / "test" / "results" / TAG

_spec19 = importlib.util.spec_from_file_location(
    "d19", ROOT / "test" / "models" / "19_volatility_model_selection.py")
d19 = importlib.util.module_from_spec(_spec19)
sys.modules["d19"] = d19
_spec19.loader.exec_module(d19)

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


def top_liquid_tickers(con, n: int) -> list[str]:
    return [r[0] for r in con.execute(
        f"select ticker from {SOURCE_TABLE} group by ticker "
        f"having count(*) > 95000 order by sum(value) desc limit {n}").fetchall()]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="19c B갈래 모델선택 전종목 확인")
    parser.add_argument("--n-tickers", type=int, default=TOP_N)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args(argv)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(ROOT / "data" / "upbit_data.db"), read_only=True)
    try:
        tickers = top_liquid_tickers(con, args.n_tickers)
        emit(f"# 19c 모델 선택 전종목({len(tickers)}) 확인 원시 수치")
        emit()
        emit(f"- 대상: 유동성 상위 {len(tickers)}종목(16번 t3·18번 D6와 같은 표본): {', '.join(tickers)}")
        emit("- 조건: 주기 성분 제거(19번에서 18/18셀 전부 우세이므로 이 조건만 사용)")
        emit("- 19번의 `run_condition`을 그대로 재사용 — 수치 일관성 보장")
        emit()

        preflight.ensure_data(table=SOURCE_TABLE, tickers=tickers)

        winners = []
        for i, ticker in enumerate(tickers, 1):
            print(f"[19c] ({i}/{len(tickers)}) {ticker}", flush=True)
            frame = con.execute(
                f"select timestamp, close from {SOURCE_TABLE} where ticker = ? order by timestamp",
                [ticker]).df()
            if args.max_rows:
                frame = frame.tail(args.max_rows).reset_index(drop=True)
            frame["ret"] = np.log(frame["close"].astype(float)).diff()
            if len(frame) < 20_000:
                print(f"  [19c] 표본 부족({len(frame)}행) — 건너뜀", flush=True)
                continue
            table, _ = d19.run_condition(frame, deseasonalize=True)
            for horizon in HORIZONS:
                subset = table[table["horizon"] == horizon].dropna(subset=["QLIKE"])
                if subset.empty:
                    continue
                best = subset.sort_values("QLIKE").iloc[0]
                winners.append({"ticker": ticker, "horizon": horizon, "최우수": best["모델"],
                                "QLIKE": best["QLIKE"], "행수": len(frame)})
    finally:
        con.close()

    result = pd.DataFrame(winners)
    result.to_csv(RESULTS_DIR / "crosssection_winners.csv", index=False, encoding="utf-8")

    emit("## 종목 × horizon 최우수 모델")
    emit()
    emit("| 종목 | h=1(15분) | h=16(4시간) | h=64(16시간) |")
    emit("| :--- | :--- | :--- | :--- |")
    pivot_model = result.pivot(index="ticker", columns="horizon", values="최우수")
    for ticker in tickers:
        if ticker not in pivot_model.index:
            emit(f"| {ticker} | (표본 부족) | | |")
            continue
        row = pivot_model.loc[ticker]
        emit(f"| {ticker} | {row.get(1, '—')} | {row.get(16, '—')} | {row.get(64, '—')} |")
    emit()

    emit("## horizon별 승자 분포 — 이것이 실제 산출물이다")
    emit()
    counts = {}
    for horizon in HORIZONS:
        subset = result[result["horizon"] == horizon]
        counts[horizon] = subset["최우수"].value_counts()
        emit(f"### h={horizon} ({horizon*15}분{'='+str(horizon*15//60)+'시간' if horizon>=4 else ''})")
        emit()
        emit(f"- 유효 종목 수: {len(subset)}/{len(tickers)}")
        for model, count in counts[horizon].items():
            emit(f"  - **{model}**: {count}종목 ({100*count/len(subset):.0f}%)")
        emit()

    dominant = {}
    emit("## 판정")
    emit()
    for horizon in HORIZONS:
        counted = counts[horizon]
        if not len(counted):
            continue
        total = len(result[result["horizon"] == horizon])
        top_count = int(counted.iloc[0])
        tied = counted[counted == top_count].index.tolist()
        dominant[horizon] = (tied[0], top_count, total)   # 대표값(그림 범례용) — 판정문은 tied 전체를 본다
        agreement = top_count / total
        if len(tied) > 1:
            emit(f"- h={horizon}: **동률** — {' / '.join(tied)}가 각 {top_count}/{total}종목"
                 f"({agreement*100:.0f}%)으로 공동 최다 → **단일 승자를 정할 수 없다, 두 모델을 함께 고려**")
        else:
            label = "구조적" if agreement >= 0.6 else ("소수 우세" if agreement < 0.4 else "혼재")
            emit(f"- h={horizon}: **{tied[0]}**가 {top_count}/{total}종목({agreement*100:.0f}%)에서 "
                 f"최다 → **{label}**")
    emit()
    btc_check = pivot_model.loc["KRW-BTC"] if "KRW-BTC" in pivot_model.index else None
    if btc_check is not None:
        matches = sum(1 for h in HORIZONS if h in btc_check.index and dominant.get(h, (None,))[0] == btc_check.get(h))
        emit(f"- BTC 단독 결과(19번)와 전종목 다수결의 일치: **{matches}/{len(HORIZONS)}개 horizon**")
        if matches < len(HORIZONS):
            emit("  → BTC 단일종목 결론이 전종목 대표성을 완전히 갖지는 않는다. horizon별 판정표(위)를 따른다.")
        else:
            emit("  → BTC 단일종목 결론이 전종목에서도 그대로 유지된다.")
    emit()

    fig, ax = plt.subplots(figsize=(11, 6.5))
    x = np.arange(len(HORIZONS))
    all_models = sorted({m for h in HORIZONS for m in counts[h].index})
    bottom = np.zeros(len(HORIZONS))
    for color, model in zip(CATEGORICAL, all_models):
        heights = [int(counts[h].get(model, 0)) for h in HORIZONS]
        ax.bar(x, heights, bottom=bottom, label=model, color=color, width=0.55,
               edgecolor="white", linewidth=1.5)
        for xi, (h_val, b_val) in enumerate(zip(heights, bottom)):
            if h_val > 0:
                ax.text(xi, b_val + h_val / 2, f"{h_val}", ha="center", va="center",
                        fontsize=10, color="white", weight="bold")
        bottom += np.array(heights)
    ax.set_xticks(x)
    ax.set_xticklabels([f"h={h}\n({h*15}분)" for h in HORIZONS])
    ax.set_ylabel(f"최우수 모델인 종목 수 (전체 {len(tickers)}종목)")
    ax.set_title(f"19c. B 갈래 모델 선택 — 유동성 상위 {len(tickers)}종목에서 재현되는가\n"
                 "BTC 하나가 아니라 각 종목에서 이긴 모델을 집계", fontsize=13)
    ax.legend(fontsize=9.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "crosssection_winners.png", dpi=115)
    plt.close(fig)

    (RESULTS_DIR / "crosssection_raw.md").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[19c] 저장: {RESULTS_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
