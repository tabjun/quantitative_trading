# %% [markdown]
# # 18c: t8 다중 시드 잡음 정량화 — 종목 간 순위를 믿을 수 있는가
#
# 헤드리스 `.py` 드라이버(`#%%` 셀 + 파일 내 마크다운, AGENTS.md 2.3).
#
# ## 왜 (Why)
# 2026-08-14 재실행에서 t8의 겹치는 4종목이 **같은 구성·같은 시드(42)·거의 같은 데이터**인데도
# trend_corr가 +0.0005 ~ +0.0355 움직였다. 10종목 평균이 +0.0167이라 **잡음이 신호보다 크다**는
# 의심이 생겼다. t8의 원래 판독 기준은 "여러 종목에서 trend_corr가 함께 양수면 신호가 구조적,
# 한 종목만이면 우연·과적합 의심"인데, 부호가 잡음으로 뒤집힐 수 있으면 이 기준은 무의미하다.
#
# ## 무엇을 (What)
# `run_t8_multiseed.sh`가 만든 5개 실행 결과를 읽어 두 가지를 분리한다.
#   (1) **순수 비결정성**: seed 42 vs 42repeat — 시드가 같으므로 차이는 GPU 비결정성 + 데이터 정렬
#   (2) **시드 민감도**: seed 42/7/123/2026 — 시드가 바뀔 때의 변동
# 그 위에서 "종목 간 차이가 잡음보다 큰가"를 신호대잡음비로 판정한다.
#
# ## 어떻게 (How)
# 종목별로 시드 간 평균·표준편차를 구하고, 표준편차와 종목 간 표준편차를 비교한다.
# 부호 안정성(4개 시드에서 부호가 일치하는 종목 수)도 함께 센다 — 이것이 t8 판독 기준의
# 유효성을 직접 판정한다.
#
# ## 기대 결과 / 반영 (Expected)
# 잡음 규모를 수치로 확정해, t8 결과를 어디까지 해석할 수 있는지 경계를 정한다.

# %%
"""18c t8 시드 잡음 분석.

실행:
    uv run test/models/18c_t8_seed_noise.py
전제:
    bash test/scripts/run_t8_multiseed.sh   # t8_seed{42,42repeat,7,123,2026}.csv 생성
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["axes.formatter.useoffset"] = False

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "engine").is_dir():
            return candidate
    raise RuntimeError("quantitative_trading 디렉터리를 찾지 못했다.")


try:
    _START = Path(__file__).resolve().parent
except NameError:
    _START = Path.cwd()
ROOT = _project_root(_START)

TAG = "18b_rerun_verification_20260814"
RESULTS_DIR = ROOT / "test" / "results" / TAG
IMAGES_DIR = ROOT / "test" / "images" / TAG
RAW_PATH = RESULTS_DIR / "t8_seed_noise_raw.md"

SEED_LABELS = ["42", "42repeat", "7", "123", "2026"]
DISTINCT_SEEDS = ["42", "7", "123", "2026"]      # 42repeat 제외(시드 중복)
METRIC = "trend_corr"

_LINES: list[str] = []


def emit(text: str = "") -> None:
    print(text, flush=True)
    _LINES.append(text)


def num(value, digits: int = 4) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "n/a" if not np.isfinite(v) else f"{v:,.{digits}f}"


def load_runs() -> dict[str, pd.DataFrame]:
    runs = {}
    for label in SEED_LABELS:
        path = RESULTS_DIR / f"t8_seed{label}.csv"
        if path.exists():
            runs[label] = pd.read_csv(path).set_index("ticker")
        else:
            print(f"[warn] 없음: {path}")
    return runs


# %%
def main() -> None:
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if len(runs) < 3:
        raise SystemExit("실행 결과가 3개 미만이다. run_t8_multiseed.sh를 먼저 돌려라.")

    emit("# t8 다중 시드 잡음 정량화 (자동 생성)")
    emit()
    emit(f"- 대상 지표: `{METRIC}` · 실행 {len(runs)}개: {', '.join(runs)}")
    configs = {(df['model'].iloc[0], df['objective'].iloc[0]) for df in runs.values()}
    emit(f"- 구성 일치 확인: {configs} (하나여야 한다)")
    emit()

    matrix = pd.DataFrame({label: df[METRIC] for label, df in runs.items()})

    # ── (1) 순수 비결정성: 같은 시드 두 번
    emit("## 1. 순수 비결정성 (seed 42 vs 42repeat — 시드가 같다)")
    emit()
    if "42" in matrix and "42repeat" in matrix:
        delta = (matrix["42repeat"] - matrix["42"]).abs()
        emit("| 종목 | seed42 | seed42(반복) | 절대차 |")
        emit("| :--- | ---: | ---: | ---: |")
        for ticker in matrix.index:
            emit(f"| {ticker} | {num(matrix.loc[ticker, '42'])} | "
                 f"{num(matrix.loc[ticker, '42repeat'])} | {num(delta[ticker])} |")
        emit()
        emit(f"- 절대차 평균 **{num(delta.mean())}**, 중앙값 {num(delta.median())}, 최대 **{num(delta.max())}**")
        if delta.max() < 1e-9:
            emit("- 완전히 동일 → GPU 비결정성 없음. 이 실행 경로는 재현 가능하다.")
            emit("  (따라서 앞서 관측한 4종목 이동은 시드가 아니라 **데이터 차이**에서 온 것이다.)")
        else:
            emit("- 0이 아니다 → **같은 시드로도 결과가 재현되지 않는다**(GPU 비결정성).")
            emit("  `torch.backends.cudnn.deterministic = True`, `benchmark = False`로 줄일 수 있다.")
        emit()

    # ── (2) 시드 민감도
    emit("## 2. 시드 민감도 (서로 다른 시드 4개)")
    emit()
    distinct = matrix[[c for c in DISTINCT_SEEDS if c in matrix]]
    stats = pd.DataFrame({
        "평균": distinct.mean(axis=1),
        "표준편차": distinct.std(axis=1, ddof=1),
        "최소": distinct.min(axis=1),
        "최대": distinct.max(axis=1),
    })
    stats["범위"] = stats["최대"] - stats["최소"]
    stats["부호일치"] = distinct.apply(lambda r: bool(np.all(r > 0) or np.all(r < 0)), axis=1)
    stats = stats.sort_values("평균", ascending=False)

    emit("| 종목 | 평균 | 표준편차 | 최소 | 최대 | 범위 | 4시드 부호 일치 |")
    emit("| :--- | ---: | ---: | ---: | ---: | ---: | :--- |")
    for ticker, row in stats.iterrows():
        emit(f"| {ticker} | {num(row['평균'])} | {num(row['표준편차'])} | {num(row['최소'])} | "
             f"{num(row['최대'])} | {num(row['범위'])} | {'예' if row['부호일치'] else '**아니오**'} |")
    emit()

    within = float(stats["표준편차"].mean())          # 종목 내 시드 변동(잡음)
    between = float(stats["평균"].std(ddof=1))        # 종목 간 변동(신호 후보)
    sign_stable = int(stats["부호일치"].sum())
    emit(f"- **종목 내 시드 표준편차 평균(잡음) = {num(within)}**")
    emit(f"- **종목 간 평균의 표준편차(신호 후보) = {num(between)}**")
    emit(f"- 신호대잡음비 = {num(between / within if within > 0 else np.inf, 2)}")
    emit(f"- 4개 시드에서 부호가 일치한 종목: **{sign_stable}/{len(stats)}**")
    emit()

    emit("## 3. 판정")
    emit()
    if between / max(within, 1e-12) < 1.0:
        emit("- **종목 간 차이가 시드 잡음보다 작다.** 종목별 trend_corr 값이나 순위를 해석할 수 없다.")
    elif between / max(within, 1e-12) < 2.0:
        emit("- 종목 간 차이가 잡음과 비슷한 규모다. 순위 해석은 위험하고, 큰 격차만 조심스럽게 볼 수 있다.")
    else:
        emit("- 종목 간 차이가 잡음보다 뚜렷하다. 순위 해석이 가능하다.")
    if sign_stable < len(stats):
        emit(f"- **{len(stats) - sign_stable}개 종목은 시드에 따라 부호가 뒤집힌다.** 따라서 t8의 원래 판독 기준")
        emit("  (\"여러 종목이 함께 양수면 신호가 구조적\")은 **성립하지 않는다** — 양수 개수는 시드가 결정한다.")
    else:
        emit("- 모든 종목의 부호가 시드에 안정적이다. 부호 기반 판독은 유효하다.")
    emit()
    emit("- 어느 쪽이든 **평균 수준(|trend_corr| < 0.05)이 실질적으로 0에 가깝다**는 결론은 변하지 않는다.")
    emit("  이 분석이 무효화하는 것은 '신호 없음'이 아니라 '종목 간 비교·순위'다.")
    emit()

    # ── 그림
    fig, axes = plt.subplots(1, 2, figsize=(18, 6.5))
    order = stats.index.tolist()
    x = np.arange(len(order))
    axes[0].errorbar(x, stats["평균"], yerr=stats["표준편차"], fmt="o", capsize=5,
                     color="#1f77b4", ecolor="#d62728", elinewidth=2, ms=7)
    axes[0].axhline(0, color="black", lw=1)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(order, rotation=90, fontsize=9)
    axes[0].set_ylabel("trend_corr")
    axes[0].set_title("종목별 trend_corr — 시드 4개 평균 ± 표준편차\n오차막대가 0을 걸치면 부호를 말할 수 없다")

    for label in [c for c in DISTINCT_SEEDS if c in matrix]:
        axes[1].plot(x, matrix.loc[order, label], marker="o", ms=4, lw=1.2, alpha=0.85, label=f"seed {label}")
    axes[1].axhline(0, color="black", lw=1)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(order, rotation=90, fontsize=9)
    axes[1].set_title("시드별 궤적 — 선이 서로 엇갈리면 순위가 잡음")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "t8_seed_noise.png", dpi=120)
    plt.close(fig)

    stats.to_csv(RESULTS_DIR / "t8_seed_noise_summary.csv", encoding="utf-8")
    RAW_PATH.write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[18c] 저장: {RAW_PATH}")
    print(f"[18c] 그림: {IMAGES_DIR / 't8_seed_noise.png'}")


if __name__ == "__main__":
    main()
