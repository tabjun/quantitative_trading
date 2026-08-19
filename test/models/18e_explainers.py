# %% [markdown]
# # 18e: 설명용 시각화 — 용어와 개념을 대조로 보여준다
#
# ## 왜 (Why)
# 사용자 지적(2026-08-19): "설명이 들어가는 것도 대조를 통해 이해하기 쉽게 시각화를 붙여라.
# 단순히 분석 결과에만 시각화를 넣지 말고. '레벨이란?', 반감기, |수익률|이 정규화인지 무엇인지,
# horizon을 바꾸면 결과가 바뀐다는 게 무슨 말인지, 장기기억이 변동성에 있다는 게 왜 중요한지,
# 가로 bar 그래프가 뭘 의미하는지 — 지금은 결과만 있고 왜가 없다."
#
# 앞선 드라이버(18/18b/18c/18d)는 **분석 결과**를 그렸다. 이 드라이버는 **개념 자체**를 그린다.
# 각 그림은 "A와 B를 나란히 놓고 무엇이 다른가"의 형태를 갖는다.
#
# ## 무엇을 (What)
#   E1 레벨이란 무엇인가          — 같은 구간을 레벨/차분으로 나란히, 무엇을 읽을 수 있는지 대조
#   E2 로그를 왜 씌우나            — 같은 상승률이 레벨에선 다른 크기로, 로그에선 같은 크기로
#   E3 |수익률|은 정규화인가       — 절댓값이 하는 일: 부호 제거. 정규화(z-score)와 나란히 대조
#   E4 반감기 메커니즘             — 충격 후 감쇠 곡선, α+β가 "얼마나 남는가"임을 단계로 표시
#   E5 horizon이란·왜 결과가 바뀌나 — h=1/16/64가 무엇을 맞히는지 도식 + 예측 가능성 감쇠
#   E6 장기기억이 변동성에 있으면 왜 중요한가 — 지수감쇠 vs 하이퍼볼릭의 예측 지평 차이
#
# ## 어떻게 (How)
# 색은 역할로 배정한다(dataviz 지침): 2범주 대조는 범주형 슬롯1·2(파랑 #2a78d6 / 주황 #eb6834,
# 검증 통과), 순서가 있는 단계는 단일 파랑 서열 램프(#86b6ef→#104281, --ordinal 통과),
# 판정은 상태색(good/warning/critical) + **반드시 텍스트 라벨 병기**(상태색은 색 단독으로
# 의미를 나르지 않는다). 이중 축은 쓰지 않는다. 격자·축은 흐리게, 값 라벨은 선택적으로만.
#
# ## 기대 결과 / 반영 (Expected)
# 보고서의 설명 문단마다 대조 그림을 붙여, 용어를 모르는 상태에서도 흐름으로 이해되게 만든다.

# %%
"""18e 설명용 시각화.

실행:
    uv run test/models/18e_explainers.py
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
matplotlib.rcParams["axes.formatter.limits"] = (-9, 12)
# 격자·축은 recessive (dataviz: 마크는 얇게, 크롬은 물러나게)
matplotlib.rcParams["axes.grid"] = True
matplotlib.rcParams["grid.color"] = "#e8e8e6"
matplotlib.rcParams["grid.linewidth"] = 0.7
matplotlib.rcParams["axes.edgecolor"] = "#c9c9c6"
matplotlib.rcParams["axes.labelcolor"] = "#52514e"
matplotlib.rcParams["xtick.color"] = "#52514e"
matplotlib.rcParams["ytick.color"] = "#52514e"
matplotlib.rcParams["axes.titlecolor"] = "#0b0b0b"

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from statsmodels.tsa.stattools import acf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine import preflight  # noqa: E402  (경로 삽입 후 import)

# ── 색 역할 (dataviz 검증 완료)
C_A = "#2a78d6"        # 범주형 슬롯1 — 레벨/원본 쪽
C_B = "#eb6834"        # 범주형 슬롯2 — 차분/변환 쪽
C_C = "#1baf7a"        # 범주형 슬롯3 — 제3 계열
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]   # 서열형 5단
S_GOOD, S_WARN, S_CRIT = "#0ca30c", "#fab219", "#d03b3b"          # 상태 (라벨 병기 필수)
INK, INK2 = "#0b0b0b", "#52514e"

TAG = "18e_explainers_20260819"
SOURCE_TABLE = "upbit_krw_candle"
BASE_TICKER = "KRW-BTC"
BARS_PER_DAY = 96


def _root() -> Path:
    start = Path(__file__).resolve().parent
    for c in [start, *start.parents]:
        if (c / "pyproject.toml").exists() and (c / "engine").is_dir():
            return c
    raise RuntimeError("root 못 찾음")


ROOT = _root()
IMAGES_DIR = ROOT / "test" / "images" / TAG


def load(ticker: str) -> pd.DataFrame:
    preflight.ensure_data(table=SOURCE_TABLE, tickers=[ticker])
    con = duckdb.connect(str(ROOT / "data" / "upbit_data.db"), read_only=True)
    try:
        frame = con.execute(
            f"select timestamp, close from {SOURCE_TABLE} where ticker = ? order by timestamp",
            [ticker],
        ).df()
    finally:
        con.close()
    frame["log_close"] = np.log(frame["close"].astype(float))
    frame["ret"] = frame["log_close"].diff()
    return frame


def krw(ax) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))


def note(ax, text: str, color: str = INK2, y: float = 0.02) -> None:
    ax.text(0.01, y, text, transform=ax.transAxes, fontsize=9.5, color=color,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb", ec="#e0e0dd", lw=0.8))


# %% [markdown]
# ## E1 — "레벨"이란 무엇인가
# 같은 300봉 구간을 두 축으로 나란히 그린다. 위는 레벨(지금 얼마), 아래는 차분(얼마 변했나).
# 같은 데이터인데 **읽을 수 있는 질문이 다르다**는 것이 핵심이다.

# %%
def fig_e1_level(frame: pd.DataFrame) -> None:
    seg = frame.iloc[-300:].reset_index(drop=True)
    x = np.arange(len(seg))
    fig, axes = plt.subplots(2, 1, figsize=(13, 7.6), sharex=True,
                             gridspec_kw={"height_ratios": [1.25, 1]})

    axes[0].plot(x, seg["close"], lw=1.8, color=C_A)
    axes[0].set_title("① 레벨(level) — 종가 그 자체.  답할 수 있는 질문: “지금 얼마인가”", fontsize=12)
    axes[0].set_ylabel("종가 (원)")
    krw(axes[0])
    i, j = 150, 200
    for k, mark in [(i, "A"), (j, "B")]:
        axes[0].scatter([k], [seg["close"][k]], s=70, color=C_A, zorder=5,
                        edgecolor="white", linewidth=1.5)
        axes[0].annotate(f"{mark}: {seg['close'][k]:,.0f}원", (k, seg["close"][k]),
                         textcoords="offset points", xytext=(8, 12), fontsize=10, color=INK)
    note(axes[0], "레벨은 '수준'이다. 점 하나만 봐도 그 시점의 가격을 알 수 있다.", y=0.05)

    axes[1].bar(x, seg["ret"], color=C_B, width=1.0)
    axes[1].axhline(0, color=INK2, lw=0.9)
    axes[1].set_title("③ 차분(difference) — 직전 대비 변화율.  답할 수 있는 질문: “얼마 변했나”", fontsize=12)
    axes[1].set_ylabel("로그수익률")
    axes[1].set_xlabel("봉 (15분 단위, 최근 300봉)")
    change = seg["close"][j] / seg["close"][i] - 1
    axes[1].annotate("", xy=(j, 0), xytext=(i, 0),
                     arrowprops=dict(arrowstyle="<->", color=INK, lw=1.6))
    axes[1].text((i + j) / 2, axes[1].get_ylim()[1] * 0.55,
                 f"A→B 누적 {change * 100:+.2f}%\n(차분을 더해야 알 수 있다)",
                 ha="center", fontsize=10, color=INK)
    note(axes[1],
         "차분은 '변화'다. 점 하나만 봐서는 가격이 얼마인지 알 수 없다 — 수준 정보를 버렸기 때문.",
         y=0.05)

    fig.suptitle("E1. 레벨 vs 차분 — 같은 데이터, 다른 질문", fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(IMAGES_DIR / "e1_level_vs_diff.png", dpi=115)
    plt.close(fig)


# %% [markdown]
# ## E2 — 로그를 왜 씌우나
# 로그 변환은 차분이 아니다. "같은 비율의 변동을 같은 크기로 보이게" 만드는 단위 변경이다.
# 낮은 가격대와 높은 가격대에서 각각 +10%를 주고, 레벨과 로그에서 그 크기를 비교한다.

# %%
def fig_e2_log(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.4))

    low, high = 40_000_000, 160_000_000
    labels = ["4천만원에서\n+10%", "1.6억원에서\n+10%"]
    level_delta = [low * 0.10, high * 0.10]
    log_delta = [np.log(low * 1.10) - np.log(low), np.log(high * 1.10) - np.log(high)]

    bars = axes[0].bar(labels, level_delta, color=[RAMP[1], RAMP[3]], width=0.55)
    axes[0].set_title("레벨에서 본 '+10%'\n→ 크기가 4배 다르다", fontsize=11.5)
    axes[0].set_ylabel("변화량 (원)")
    krw(axes[0])
    for bar, value in zip(bars, level_delta):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value, f"{value:,.0f}원",
                     ha="center", va="bottom", fontsize=10, color=INK)

    bars = axes[1].bar(labels, log_delta, color=[RAMP[1], RAMP[3]], width=0.55)
    axes[1].set_title("로그에서 본 '+10%'\n→ 크기가 정확히 같다", fontsize=11.5)
    axes[1].set_ylabel("변화량 (log)")
    axes[1].set_ylim(0, max(log_delta) * 1.45)
    for bar, value in zip(bars, log_delta):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value, f"{value:.4f}",
                     ha="center", va="bottom", fontsize=10, color=INK)

    seg = frame.iloc[-6000:]
    ax2 = axes[2]
    ax2.plot(np.arange(len(seg)), seg["close"] / seg["close"].iloc[0],
             lw=1.6, color=C_A, label="레벨 (첫 값=1로 맞춤)")
    ax2.plot(np.arange(len(seg)), np.exp(seg["log_close"] - seg["log_close"].iloc[0]),
             lw=1.6, ls="--", color=C_B, label="로그 차이를 되돌린 값")
    ax2.set_title("두 경로가 완전히 겹친다\n→ 로그는 정보를 버리지 않는다(단위만 바꿈)", fontsize=11.5)
    ax2.set_xlabel("봉")
    ax2.legend(fontsize=9.5, frameon=False)

    fig.suptitle("E2. 로그 변환은 차분이 아니다 — 곱셈적 변동을 덧셈적으로 바꾸는 '단위 변경'", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(IMAGES_DIR / "e2_why_log.png", dpi=115)
    plt.close(fig)


# %% [markdown]
# ## E3 — |수익률|은 정규화인가? (아니다)
# 사용자 질문에 직접 답한다. 절댓값은 **부호를 버리는 연산**이고, 정규화(z-score)는
# **중심과 척도를 맞추는 연산**이다. 셋을 나란히 놓고 무엇이 달라지는지 본다.

# %%
def fig_e3_abs(frame: pd.DataFrame) -> None:
    seg = frame.iloc[-400:].reset_index(drop=True)
    ret = seg["ret"].fillna(0.0)
    x = np.arange(len(seg))
    z = (ret - ret.mean()) / ret.std()

    fig, axes = plt.subplots(3, 2, figsize=(16, 10),
                             gridspec_kw={"width_ratios": [1.7, 1]})

    axes[0][0].bar(x, ret, color=C_B, width=1.0)
    axes[0][0].axhline(0, color=INK2, lw=0.9)
    axes[0][0].set_title("③ 수익률 — 부호(+/−)가 있다", fontsize=11.5)
    axes[0][1].hist(ret, bins=60, color=C_B)
    axes[0][1].set_title("분포: 0을 중심으로 좌우 대칭", fontsize=11.5)

    axes[1][0].bar(x, ret.abs(), color=C_C, width=1.0)
    axes[1][0].set_title("④ |수익률| = 절댓값 — 부호를 버렸다 (음수가 위로 접힘)", fontsize=11.5)
    axes[1][1].hist(ret.abs(), bins=60, color=C_C)
    axes[1][1].set_title("분포: 0에서 시작해 한쪽으로만 뻗는다", fontsize=11.5)

    axes[2][0].bar(x, z, color=RAMP[3], width=1.0)
    axes[2][0].axhline(0, color=INK2, lw=0.9)
    axes[2][0].set_title("(대조) z-score 정규화 — 부호는 그대로, 척도만 바뀐다", fontsize=11.5)
    axes[2][1].hist(z, bins=60, color=RAMP[3])
    axes[2][1].set_title("분포: 모양 그대로, 축 단위만 바뀜", fontsize=11.5)

    for row in axes:
        row[0].set_xlim(0, len(seg))
    axes[2][0].set_xlabel("봉 (최근 400봉)")

    note(axes[1][0],
         "절댓값은 정규화가 아니다. 정규화는 '같은 정보를 다른 눈금으로', 절댓값은 '정보 하나(부호)를 버림'.",
         color=INK, y=0.72)
    note(axes[2][0],
         "그래서 |수익률|로는 오를지 내릴지 답할 수 없고, z-score로는 답할 수 있다.",
         color=INK, y=0.72)

    fig.suptitle("E3. |수익률|은 정규화가 아니다 — 부호를 버리는 연산이다\n"
                 "왜 버리나: 방향에는 신호가 없고 크기에는 있으니(17번 S3), 크기만 남기면 신호가 진해진다",
                 fontsize=13.5, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.945])
    fig.savefig(IMAGES_DIR / "e3_abs_is_not_normalization.png", dpi=115)
    plt.close(fig)


# %% [markdown]
# ## E4 — 반감기 메커니즘
# "α+β는 직전 봉의 초과 변동성이 다음 봉에 얼마나 남는가"를 계단으로 그린다.
# 그리고 실제 BTC 데이터에서 큰 충격 뒤 변동성이 가라앉는 구간을 겹쳐 놓는다.

# %%
def fig_e4_halflife(frame: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    # (1) 남는 비율의 계단 — α+β=0.98 사례
    persistence = 0.98
    steps = np.arange(0, 60)
    remain = persistence ** steps
    axes[0].step(steps, remain, where="post", lw=2, color=C_A)
    axes[0].fill_between(steps, remain, step="post", alpha=0.12, color=C_A)
    half = np.log(0.5) / np.log(persistence)
    axes[0].axhline(0.5, color=S_CRIT, ls="--", lw=1.5)
    axes[0].axvline(half, color=S_CRIT, ls="--", lw=1.5)
    axes[0].scatter([half], [0.5], s=90, color=S_CRIT, zorder=6, edgecolor="white", lw=1.5)
    axes[0].annotate(f"반감기 = {half:.1f}봉 = {half*0.25:.1f}시간\n(절반으로 줄어드는 지점)",
                     (half, 0.5), textcoords="offset points", xytext=(14, 30),
                     fontsize=10.5, color=INK,
                     arrowprops=dict(arrowstyle="->", color=S_CRIT, lw=1.2))
    chain = " → ".join(f"{k}봉 {remain[k]*100:.0f}%" for k in range(1, 4))
    axes[0].text(0.30, 0.90,
                 f"매 봉마다 ×{persistence}\n{chain}",
                 transform=axes[0].transAxes, fontsize=10, color=INK, va="top",
                 bbox=dict(boxstyle="round,pad=0.35", fc="#fcfcfb", ec="#e0e0dd", lw=0.8))
    axes[0].set_title(f"α+β = {persistence} 일 때\n매 봉 {persistence*100:.0f}%가 남는다", fontsize=11.5)
    axes[0].set_xlabel("충격 이후 경과 봉"); axes[0].set_ylabel("남아 있는 초과 변동성 비율")
    axes[0].set_ylim(0, 1.05)

    # (2) α+β를 바꾸면 — 서열 램프
    for color, p in zip(RAMP, [0.95, 0.98, 0.99, 0.999, 0.9999]):
        h = np.log(0.5) / np.log(p)
        axes[1].plot(steps, p ** steps, lw=2, color=color,
                     label=f"α+β={p} → 반감기 {h*0.25:,.0f}시간")
    axes[1].axhline(0.5, color=S_CRIT, ls="--", lw=1.3)
    axes[1].set_title("α+β가 1에 가까워지면\n충격이 거의 사라지지 않는다", fontsize=11.5)
    axes[1].set_xlabel("충격 이후 경과 봉")
    axes[1].legend(fontsize=9, loc="lower left", frameon=True, framealpha=0.95,
                   edgecolor="#e0e0dd")
    axes[1].set_ylim(0, 1.05)

    # (3) 실제 데이터: 최대 충격 이후 실현변동성 감쇠
    ret = frame["ret"].fillna(0.0)
    realized = ret.abs().rolling(BARS_PER_DAY).mean()
    shock = int(ret.abs().idxmax())
    lo, hi = max(shock - BARS_PER_DAY, 0), min(shock + BARS_PER_DAY * 12, len(frame) - 1)
    window = realized.iloc[lo:hi].to_numpy()
    axes[2].plot(np.arange(len(window)), window, lw=1.8, color=C_B)
    axes[2].axvline(shock - lo, color=S_CRIT, lw=1.5, ls="--")
    axes[2].annotate("← 최대 충격 발생",
                     (shock - lo, np.nanmax(window) * 0.95),
                     textcoords="offset points", xytext=(12, 0), fontsize=10.5, color=INK)
    axes[2].set_title("실제 BTC: 큰 충격 뒤 변동성이 가라앉는다\n(이 감쇠 속도를 재는 것이 반감기)", fontsize=11.5)
    axes[2].set_xlabel("봉 (충격 전 1일 ~ 충격 후 12일)")
    axes[2].set_ylabel("평균 |수익률| (1일 이동)")

    fig.suptitle("E4. 반감기란 — “부풀어 오른 변동성이 절반으로 줄어드는 시간”. "
                 "α+β는 “직전 봉의 초과 변동성이 다음 봉에 남는 비율”", fontsize=13.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(IMAGES_DIR / "e4_halflife_mechanism.png", dpi=115)
    plt.close(fig)


# %% [markdown]
# ## E5 — horizon이란 무엇이고 왜 결과가 바뀌나
# h는 "몇 봉 뒤를 맞히려 하는가"다. h가 커지면 (1) 맞혀야 하는 대상이 멀어지고
# (2) 그 대상이 여러 봉의 평균이 되어 성질이 달라진다. 그래서 모델 순위가 바뀔 수 있다.

# %%
def fig_e5_horizon(frame: pd.DataFrame) -> None:
    ret = frame["ret"].dropna()
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))

    # (1) 도식: h가 무엇을 가리키나
    seg = ret.iloc[-120:].to_numpy()
    x = np.arange(len(seg))
    axes[0].bar(x, np.abs(seg), color="#d8d8d5", width=1.0)
    now = 40
    axes[0].axvline(now, color=INK, lw=1.6)
    axes[0].text(now, np.abs(seg).max() * 1.02, "현재(t)", ha="center", fontsize=10.5, color=INK)
    for color, h, dy in zip([RAMP[1], RAMP[2], RAMP[4]], [1, 16, 64], [0.86, 0.66, 0.46]):
        axes[0].annotate("", xy=(now + h, np.abs(seg).max() * dy),
                         xytext=(now, np.abs(seg).max() * dy),
                         arrowprops=dict(arrowstyle="->", color=color, lw=2.4))
        axes[0].text(now + h / 2, np.abs(seg).max() * (dy + 0.03),
                     f"h={h} ({h*15}분)", ha="center", fontsize=10, color=color)
    axes[0].set_title("horizon(h) = 몇 봉 뒤를 맞히려 하는가", fontsize=11.5)
    axes[0].set_xlabel("봉"); axes[0].set_ylabel("|수익률|")

    # (2) h가 커지면 대상이 평균이 되어 매끄러워진다
    for color, h in zip([RAMP[1], RAMP[2], RAMP[4]], [1, 16, 64]):
        target = ret.pow(2).rolling(h).mean().dropna()
        sample = target.iloc[-3000:].to_numpy()
        axes[1].plot(np.arange(len(sample)), sample / sample.mean(), lw=1.1, color=color,
                     label=f"h={h} 평균분산 (변동계수 {sample.std()/sample.mean():.2f})")
    axes[1].set_title("h가 커지면 맞혀야 할 대상이 매끄러워진다\n→ 같은 모델도 난이도가 달라진다", fontsize=11.5)
    axes[1].set_xlabel("봉"); axes[1].set_ylabel("평균분산 (각자의 평균=1로 맞춤)")
    axes[1].legend(fontsize=9, frameon=False)

    # (3) 예측 가능성(자기상관)이 h에 따라 감쇠
    bars = acf(ret.abs().tail(20_000), nlags=200, fft=True)[1:]
    axes[2].plot(np.arange(1, len(bars) + 1), bars, lw=2, color=C_C)
    for color, h in zip([RAMP[1], RAMP[2], RAMP[4]], [1, 16, 64]):
        axes[2].axvline(h, color=color, ls="--", lw=1.5)
        axes[2].annotate(f"h={h}\nACF {bars[h-1]:.3f}", (h, bars[h - 1]),
                         textcoords="offset points", xytext=(8, 14), fontsize=9.5, color=color)
    axes[2].set_xscale("log")
    axes[2].set_title("h가 멀어질수록 남아 있는 신호가 줄어든다\n(|수익률| 자기상관)", fontsize=11.5)
    axes[2].set_xlabel("시차 = h (봉, 로그축)"); axes[2].set_ylabel("자기상관")

    fig.suptitle("E5. “horizon을 바꾸면 결과가 바뀐다”의 뜻 — h는 맞혀야 할 대상 자체를 바꾼다. "
                 "대상이 바뀌면 어느 모델이 유리한지도 바뀐다", fontsize=13.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(IMAGES_DIR / "e5_horizon_meaning.png", dpi=115)
    plt.close(fig)


# %% [markdown]
# ## E6 — 장기기억이 변동성에 있으면 왜 중요한가
# "장기기억"은 **자기상관이 얼마나 멀리까지 남는가**에 대한 성질이다. 지수적으로 꺼지면
# 짧은 지평만 예측 가능하고, 하이퍼볼릭(멱함수)으로 꺼지면 먼 지평까지 예측 가능하다.
# 그래서 "어느 채널에 장기기억이 있는가"가 곧 "어느 채널을 얼마나 먼 미래까지 예측할 수 있는가"다.

# %%
def fig_e6_longmemory(frame: pd.DataFrame) -> None:
    ret = frame["ret"].dropna().tail(20_000)
    lags = np.arange(1, 201)
    acf_ret = acf(ret, nlags=200, fft=True)[1:]
    acf_abs = acf(ret.abs(), nlags=200, fft=True)[1:]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))

    # (1) 두 채널의 감쇠를 나란히 (선형축)
    axes[0].plot(lags, acf_ret, lw=2, color=C_B, label="③ 수익률(방향 채널)")
    axes[0].plot(lags, acf_abs, lw=2, color=C_C, label="④ |수익률|(크기 채널)")
    axes[0].axhline(0, color=INK2, lw=0.9)
    axes[0].set_title("같은 데이터, 두 채널의 자기상관\n방향은 즉시 0 / 크기는 길게 남는다", fontsize=11.5)
    axes[0].set_xlabel("시차 (봉)"); axes[0].set_ylabel("자기상관")
    axes[0].legend(fontsize=9.5, frameon=False)

    # (2) 감쇠 형태 — 로그로그에서 직선이면 하이퍼볼릭(장기기억)
    axes[1].loglog(lags, np.maximum(acf_abs, 1e-4), lw=2, color=C_C, label="실제 |수익률| ACF")
    axes[1].loglog(lags, acf_abs[0] * 0.97 ** lags, lw=1.6, ls="--", color=C_B,
                   label="지수 감쇠 (기억 짧음)")
    axes[1].loglog(lags, acf_abs[0] * lags ** -0.2, lw=1.6, ls="-.", color=RAMP[4],
                   label="하이퍼볼릭 감쇠 (장기기억)")
    axes[1].set_title("감쇠 '형태'를 본다\n실제 곡선이 하이퍼볼릭 직선을 따라간다", fontsize=11.5)
    axes[1].set_xlabel("시차 (봉, 로그축)"); axes[1].set_ylabel("자기상관 (로그축)")
    axes[1].legend(fontsize=9, frameon=False)

    # (3) 그래서 무엇이 달라지나 — 예측 가능 지평
    threshold = 0.05
    exp_curve = acf_abs[0] * 0.97 ** lags
    hyp_curve = acf_abs[0] * lags ** -0.2
    exp_h = lags[exp_curve < threshold][0] if (exp_curve < threshold).any() else lags[-1]
    hyp_h = lags[hyp_curve < threshold][0] if (hyp_curve < threshold).any() else lags[-1]
    names = ["지수 감쇠\n(기억 짧음)", "하이퍼볼릭\n(장기기억)"]
    values = [exp_h * 0.25, hyp_h * 0.25]
    bars = axes[2].bar(names, values, color=[C_B, RAMP[4]], width=0.5)
    for bar, value in zip(bars, values):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value,
                     f"{value:,.0f}시간" if value < 24 * 30 else "200봉 내\n안 꺼진다",
                     ha="center", va="bottom", fontsize=11, color=INK)
    axes[2].set_title(f"자기상관이 {threshold}까지 떨어지는 시점\n= 대략 여기까지 예측을 시도할 수 있다", fontsize=11.5)
    axes[2].set_ylabel("예측 가능 지평 (시간)")
    axes[2].set_ylim(0, max(values) * 1.22)

    fig.suptitle("E6. 장기기억이 왜 중요한가 — “기억이 얼마나 멀리 남는가”가 “얼마나 먼 미래를 예측할 수 있는가”를 정한다",
                 fontsize=13.5, y=0.99)
    fig.text(0.685, 0.015,
             "장기기억이 있으면 먼 미래까지 예측 여지가 남는다. 그것이 '가격이 아니라 변동성에\n"
             "장기기억이 있다'가 중요한 이유 — 먼 지평 예측을 걸 수 있는 대상이 변동성뿐이라는 뜻이다.",
             fontsize=10, color=INK, ha="center", va="bottom",
             bbox=dict(boxstyle="round,pad=0.4", fc="#fcfcfb", ec="#e0e0dd", lw=0.8))
    fig.tight_layout(rect=[0, 0.11, 1, 0.93])
    fig.savefig(IMAGES_DIR / "e6_why_long_memory.png", dpi=115)
    plt.close(fig)


# %%
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="18e 설명용 시각화")
    parser.add_argument("--ticker", default=BASE_TICKER)
    args = parser.parse_args(argv)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    frame = load(args.ticker)
    print(f"[18e] 데이터 {len(frame):,}행", flush=True)
    for name, fn in [
        ("E1 레벨", fig_e1_level), ("E2 로그", fig_e2_log), ("E3 절댓값", fig_e3_abs),
        ("E4 반감기", fig_e4_halflife), ("E5 horizon", fig_e5_horizon),
        ("E6 장기기억", fig_e6_longmemory),
    ]:
        fn(frame)
        print(f"  [완료] {name}", flush=True)
    print(f"[18e] 그림 저장: {IMAGES_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
