# %% [markdown]
# # 27b번: EDA·결과 점검·계열 단위 검정(27번의 근거)
#
# 27번(신규 알고리즘 확대) 결과를 해석하기 전에, 데이터가 어떤 모습인지와 적용한 처리가 의도대로 작동했는지를 거시에서 미시로
# 내려가며 확인하고, 연구 질문(순차 대 병렬 어텐션 대 파운데이션)에 직접 답하는 계열 단위 검정을 한다. CPU만 쓴다.
#
# | 절 | 내용 | 필요한 입력 |
# | :--- | :--- | :--- |
# | A. 거시 EDA | 가격·수익률 선 그래프, 일별 RV, 월별 정지 비율, 블록 로그 RV 자기상관 | 연장 DB |
# | B. 적용 변경 점검 | 행 기준 창의 시간 왜곡, 정지 확률의 신뢰도, MS-GARCH 수정 전후 | 연장 DB, 정지 확률 캐시, 26c 결과 |
# | C. 결과 점검 | 모델별 예측 품질 이상치, 26c 공유 모델 값 대조, 예측-실제 겹쳐 그리기 | 27번 결합 결과(없으면 "결과 대기") |
# | D. 계열 단위 검정 | 계열 평균 손실의 쌍별 DM, 계열별 MCS 포함 수 | 27번 결합 결과와 26b(27) 결과 |
#
# 계열 단위 검정에서 계열 대표를 평가 성적으로 고르면 선택 편향이 생긴다. 그래서 **계열 구성원 손실의 단순 평균**(선택 없음)을
# 비교한다. 이것은 "그 계열의 전형적인 모델"을 비교하는 것이며, "그 계열의 최선 모델" 비교는 26b의 모델 단위 MCS가 맡는다.

# %%
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "test" / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

TAG = "27b_eda_verification_20261007"
STEM = "27b_eda_verification"
RES = ROOT / "test" / "results" / TAG
IMG = ROOT / "test" / "images" / TAG
R27 = ROOT / "test" / "results" / "27_model_expansion_20261006"
R26C = ROOT / "test" / "results" / "26c_recent_twopart_20261005"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


X27 = _load("m27_for27b", ROOT / "test" / "models" / "27_model_expansion_test.py")
M = X27.M
# 고정 색 순서(dataviz 기준 팔레트). 색은 계열·구간을 따라가고 순위를 따라가지 않는다.
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#8a6b2e", "#6b6b66"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e5e5e0"
FAMS = ["통계", "하이브리드", "트리", "딥러닝", "커널", "어텐션", "합성곱", "상태공간", "파운데이션"]
FAM_COLOR = dict(zip(FAMS, PAL))
H_COLOR = dict(zip(M.HORIZONS_H, PAL))
DELISTED = {"KRW-AQT", "KRW-AERGO"}
REP_MODELS = ["GARCH-t", "LightGBM", "GRU", "PatchTST", "TimesFM"]     # 결과를 보기 전에 정한 계열별 예시 모델(선택 편향 방지)
REP_TICKERS = ["KRW-BTC", "KRW-DOGE", "KRW-BOUNTY"]
_LINES: list[str] = []


def emit(t: str = "") -> None:
    print(t, flush=True)
    _LINES.append(t)


def style(ax, title: str = "") -> None:
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10, loc="left", color=INK)


def save(fig, name: str) -> str:
    IMG.mkdir(parents=True, exist_ok=True)
    p = IMG / f"{STEM}_{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return os.path.relpath(p, RES)


def tidy_time_log(ax, fs: float = 7) -> None:
    """작은 칸 여러 개의 시간 축(연도만)과 로그 축(일반 숫자)을 겹치지 않게 정리한다."""
    import matplotlib.dates as mdates
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(labelsize=fs)


# %% [markdown]
# ## 종목별 요약 계산(병렬)

# %%
def ticker_stats(tk: str) -> dict:
    close = M.load_close(tk)
    D = M.build_data(tk)
    g, r = D["grid"], D["r"]
    ok = np.isfinite(r)
    day = pd.DatetimeIndex(g).floor("D")
    s = pd.DataFrame({"day": day, "r2": np.where(ok, r ** 2, np.nan), "z": np.where(ok, (r == 0).astype(float), np.nan)})
    daily_rv = np.sqrt(s.groupby("day")["r2"].sum(min_count=1))
    month = pd.DatetimeIndex(g).to_period("M").astype(str)
    zero_m = pd.Series(s["z"].to_numpy()).groupby(month).mean()
    ts = close.index
    span = (ts[4:] - ts[:-4]) / pd.Timedelta("1min")                      # 행 4개(옛 '1시간' 창)의 실제 시간 길이
    te_mask = ts[:-4] >= M.SPLIT
    acf = {}
    for H in M.HORIZONS_H:
        B = X27.block_data(D, H)
        y = pd.Series(B["y"])
        y = y - y.mean()
        acf[H] = [float(y.autocorr(k)) for k in range(1, 61)]
    tr = g < M.SPLIT
    return dict(tk=tk, close=close.resample("D").last(), daily_rv=daily_rv, zero_m=zero_m, acf=acf,
                r15=pd.Series(r, index=g) if tk in ("KRW-BTC", "KRW-DOGE") else None,
                span_over60_all=float(np.mean(span > 60)), span_over60_te=float(np.mean(span[te_mask] > 60)),
                span_p99_te=float(np.quantile(span[te_mask], 0.99)),
                zero_tr=float(np.nanmean(s["z"][tr])), zero_te=float(np.nanmean(s["z"][~tr])),
                rv_tr=float(np.nanmedian(daily_rv[daily_rv.index < M.SPLIT])),
                rv_te=float(np.nanmedian(daily_rv[daily_rv.index >= M.SPLIT])),
                last=str(close.index.max()))


# %% [markdown]
# ## A. 거시 EDA

# %%
def acf_median(st: list[dict]) -> dict:
    """예측 구간별 블록 로그 RV 자기상관(시차 1~60)의 종목 중앙값."""
    return {H: np.nanmedian(np.array([x["acf"][H] for x in st]), axis=0) for H in M.HORIZONS_H}


def section_a0(st: list[dict]) -> None:
    """A0 한눈에 보기: 20종목 가격·변동성을 한 축에 겹쳐 그리고, 변동성의 자기상관을 붙인다."""
    am = acf_median(st)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1.15, 0.9]})
    norm, rv30 = {}, {}
    for x in st:
        c = x["close"].dropna()
        norm[x["tk"]] = c / c.iloc[0]
        v = x["daily_rv"].dropna() * 100
        rv30[x["tk"]] = v.rolling(30, min_periods=10).median()
    for ax, dd, col, ttl in ((axes[0], norm, PAL[0], "(가) 20종목 일별 종가(첫날 = 1, 로그 축)\n가격 수준이 한곳으로 돌아오지 않는다"),
                             (axes[1], rv30, PAL[2], "(나) 20종목 일별 변동성(30일 이동 중앙값, %, 로그 축)\n높거나 낮은 상태가 몇 달씩 이어지다 바뀐다")):
        for tk, sr in dd.items():
            ax.plot(sr.index, sr.values, color=col, lw=0.8, alpha=0.45)
        med = pd.DataFrame(dd).median(axis=1)
        ax.plot(med.index, med.values, color=INK, lw=2.2, label="20종목 중앙값")
        ax.set_yscale("log")
        ax.axvline(M.SPLIT, color=MUTED, lw=1.2, ls="--")
        y1 = ax.get_ylim()[1]
        ax.text(M.SPLIT, y1, "학습 ← | → 평가 ", ha="center", va="bottom", fontsize=12, color=MUTED)
        style(ax)
        ax.set_title(ttl, fontsize=13.5, loc="left", color=INK, pad=22)
        tidy_time_log(ax, fs=12)
        ax.legend(frameon=False, fontsize=12, loc="lower left" if ax is axes[0] else "upper right")
    ax = axes[2]
    for H in M.HORIZONS_H:
        ax.plot(range(1, 61), am[H], color=H_COLOR[H], lw=2.2, label=M.hlabel(H))
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel("몇 구간 전 변동성인가(시차)", fontsize=12.5)
    ax.set_ylabel("다음 변동성과의 상관(종목 중앙값)", fontsize=12.5)
    ax.tick_params(labelsize=12)
    ax.legend(frameon=False, fontsize=12, title="예측 구간", title_fontsize=12, ncol=1)
    style(ax)
    ax.set_title("(다) 변동성의 자기상관(로그 실현변동성)\n과거 변동성이 다음 변동성을 설명한다", fontsize=13.5, loc="left", color=INK, pad=22)
    fig.tight_layout(w_pad=3)  # 제목은 보고서·포스터 캡션이 맡는다
    emit("### A0. 한눈에 보기: 비정상이지만 시간 구조가 있는 시계열")
    emit()
    emit(f"![한눈에 보기]({save(fig, 'a0_overview')})")
    emit()
    fin = pd.Series({tk.replace("KRW-", ""): float(sr.iloc[-1]) for tk, sr in norm.items()})
    swing = pd.Series({tk.replace("KRW-", ""): float(sr.max() / sr.min()) for tk, sr in rv30.items()})
    emit("**읽는 법**: (가)·(나)는 20종목을 한 축에 겹쳐 그린 것이고(옅은 선 = 종목, 검정 = 그날의 20종목 중앙값), 점선은 학습/평가 분할"
         f"({M.SPLIT.date()})이다. (다)는 예측 구간 길이로 묶은 블록의 로그 실현변동성이 몇 블록 전 값과 얼마나 상관되는지(종목마다 구한 뒤 중앙값)다.")
    emit()
    emit(f"- **(가) 가격**: 3년 뒤 가격은 첫날의 {fin.min():.2f}배({fin.idxmin()})~{fin.max():.1f}배({fin.idxmax()})로 퍼지고, "
         f"{int((fin > 1).sum())}종목이 오르고 {int((fin <= 1).sum())}종목이 내렸다. 가격 수준은 돌아올 고정점이 없다.")
    emit(f"- **(나) 변동성**: 종목마다 30일 이동 중앙값의 최대÷최소는 중앙 {swing.median():.1f}배(범위 {swing.min():.1f}~{swing.max():.1f}배)다. "
         "변동성은 한 수준에 머물지 않고 높은 상태와 낮은 상태가 몇 달씩 이어진다.")
    emit(f"- **(다) 시간 구조**: 직전 블록과의 상관은 {' · '.join(f'{M.hlabel(H)} {am[H][0]:.2f}' for H in M.HORIZONS_H)}이고, "
         f"60블록 뒤에도 {min(am[H][59] for H in M.HORIZONS_H):.2f}~{max(am[H][59] for H in M.HORIZONS_H):.2f}로 0이 되지 않는다. "
         "가격은 예측할 고정점이 없지만, 변동성은 과거가 다음을 설명하는 시계열이다.")
    ns = ROOT / "test" / "results" / "27d_regime_seqpar_20261008" / "27d_regime_seqpar_nonstationarity.csv"
    if ns.exists():
        d = pd.read_csv(ns)
        n_tk = d["종목"].nunique()
        price_ur = int(d[d["계열"] == "로그가격"]["판정"].str.startswith("비정상").sum())
        arch = int((d[d["계열"] == "로그수익률"]["ARCH_LM_p"] < 0.05).sum())
        mix = " · ".join(f"{M.hlabel(H)} {int(d[d['계열'] == f'로그RV {M.hlabel(H)}']['판정'].str.startswith('엇갈림').sum())}/{n_tk}"
                         for H in M.HORIZONS_H)
        emit(f"- **검정(27d 2-3절)**: 로그가격은 {price_ur}/{n_tk}종목이 단위근 비정상(ADF·KPSS), 로그수익률은 ARCH-LM이 {arch}/{n_tk}종목에서 "
             f"기각(분산이 시간에 따라 변함), 로그 RV는 ADF·KPSS가 함께 기각한 '엇갈림'(단위근은 아니지만 정상성도 기각, 장기기억·수준 이동)이 {mix}종목이다.")
    else:
        emit("- **검정(27d 2-3절)**: 해당 없음(27d 비정상성 결과가 아직 없음).")
    emit("- **연구 방향과의 연결**: 변동성의 수준이 국면처럼 바뀌므로 전체 기간 평균 하나로 모델을 비교하면 국면마다 다른 승부가 섞인다. 그래서 "
         "직전 변동성 국면으로 나눠 다시 비교한다(27d). 또 과거가 다음을 설명하는 시간 구조가 있으므로, 과거를 어떤 방식으로 읽는가"
         "(차례로 읽는 순차 대 한꺼번에 보는 병렬)가 비교의 축이 된다.")
    emit()


def section_a(st: list[dict]) -> None:
    emit("## A. 거시 EDA: 데이터 자체가 어떤 모습인가")
    emit()
    section_a0(st)
    # A1 가격
    fig, axes = plt.subplots(4, 5, figsize=(16, 10), sharex=True)
    for ax, x in zip(axes.ravel(), st, strict=True):
        c = x["close"].dropna()
        ax.plot(c.index, c / c.iloc[0], color=PAL[0], lw=1)
        ax.set_yscale("log")
        ax.axvline(M.SPLIT, color=MUTED, lw=1, ls="--")
        if x["tk"] in DELISTED:
            ax.axvline(pd.Timestamp(x["last"]), color=PAL[1], lw=1.2)
        style(ax, x["tk"].replace("KRW-", "") + (" (상장폐지, 주황선 = 데이터 끝)" if x["tk"] in DELISTED else ""))
        tidy_time_log(ax)
    fig.tight_layout()  # 제목은 보고서·포스터 캡션이 맡는다
    emit(f"![가격]({save(fig, 'a1_price')})")
    emit()
    emit(f"**그림**: 종목별 일별 종가(첫날 = 1, 로그 축). 점선 = 학습/평가 분할({M.SPLIT.date()}).")
    emit()
    rows = [(x["tk"].replace("KRW-", ""), float(x["close"].dropna().iloc[-1] / x["close"].dropna().iloc[0])) for x in st]
    up = sum(v > 1 for _, v in rows)
    emit(f"**읽는 법과 해석**: 각 칸은 종목 하나의 3년 가격 경로다. 3년 동안 오른 종목 {up}개, 내린 종목 {20 - up}개이고, "
         f"가장 많이 오른 종목은 {max(rows, key=lambda z: z[1])[0]}({max(v for _, v in rows):.1f}배), 가장 많이 내린 종목은 "
         f"{min(rows, key=lambda z: z[1])[0]}({min(v for _, v in rows):.2f}배)다. 가격은 방향 없이 떠돌며 종목마다 경로가 전혀 달라 "
         "가격 수준 자체는 예측 대상이 될 수 없다(비정상). 그래서 이 연구는 가격이 아니라 변동성을 예측한다. AQT·AERGO는 2026-07-18에 "
         "데이터가 끝난다(상장폐지, 평가 기간이 약 2.5개월 짧음).")
    emit()
    # A2 수익률(BTC·DOGE)과 일별 RV
    fig, axes = plt.subplots(2, 1, figsize=(14, 5.5), sharex=True)
    for ax, x in zip(axes, [x for x in st if x["r15"] is not None], strict=True):
        r = x["r15"]
        ax.plot(r.index, r.values * 100, color=PAL[0], lw=0.3)
        ax.axvline(M.SPLIT, color=MUTED, lw=1, ls="--")
        style(ax, f"{x['tk'].replace('KRW-', '')} 15분 로그수익률(%)")
    emit(f"![15분 수익률]({save(fig, 'a2_returns')})")
    emit()
    emit("**읽는 법과 해석**: 15분 수익률은 0 주변에서 오르내리지만 큰 움직임이 몰려서 나타난다(변동성 군집). 조용한 기간과 요동치는 "
         "기간이 번갈아 오는 이 구조가 변동성을 예측할 수 있는 근거다. 수익률의 부호는 예측할 수 없지만, 크기는 앞뒤로 이어진다.")
    emit()
    fig, axes = plt.subplots(4, 5, figsize=(16, 10), sharex=True)
    for ax, x in zip(axes.ravel(), st, strict=True):
        v = x["daily_rv"].dropna() * 100
        ax.plot(v.index, v.values, color=PAL[2], lw=0.6)
        ax.plot(v.index, v.rolling(30, min_periods=10).median().values, color=INK, lw=1.2)
        ax.set_yscale("log")
        ax.axvline(M.SPLIT, color=MUTED, lw=1, ls="--")
        style(ax, x["tk"].replace("KRW-", ""))
        tidy_time_log(ax)
    fig.tight_layout()  # 제목은 보고서·포스터 캡션이 맡는다
    emit(f"![일별 RV]({save(fig, 'a2_daily_rv')})")
    emit()
    emit("**그림**: 종목별 일별 실현변동성(%, 로그 축). 초록 = 일별, 검정 = 30일 이동 중앙값, 점선 = 학습/평가 분할.")
    emit()
    ratio = pd.Series({x["tk"].replace("KRW-", ""): x["rv_te"] / x["rv_tr"] for x in st})
    emit("| 종목 | 일별 RV 중앙값(학습) | 일별 RV 중앙값(평가) | 평가/학습 |")
    emit("| :--- | ---: | ---: | ---: |")
    for x in st:
        emit(f"| {x['tk'].replace('KRW-', '')} | {x['rv_tr']:.3%} | {x['rv_te']:.3%} | {x['rv_te'] / x['rv_tr']:.2f} |")
    emit()
    emit(f"**읽는 법과 해석**: 변동성 수준은 몇 달 단위로 크게 바뀐다(이동 중앙값의 오르내림). 평가 기간의 일별 변동성은 학습 기간 대비 "
         f"종목 중앙 {ratio.median():.2f}배이고, 범위는 {ratio.min():.2f}배({ratio.idxmin()})~{ratio.max():.2f}배({ratio.idxmax()})다. "
         "학습 때와 수준이 다른 구간을 예측해야 하므로, 모델이 최근 수준에 얼마나 빨리 적응하는지와 보정(내부검증 배율)이 결과에 영향을 준다.")
    emit()
    # A3 월별 정지 비율
    zm = pd.DataFrame({x["tk"].replace("KRW-", ""): x["zero_m"] for x in st}).T
    zm = zm[sorted(zm.columns)]
    fig, ax = plt.subplots(figsize=(15, 6.5))
    im = ax.imshow(zm.values * 100, aspect="auto", cmap="Blues", vmin=0, vmax=np.nanpercentile(zm.values * 100, 98))
    ax.set_yticks(range(len(zm)), zm.index, fontsize=8)
    ax.set_xticks(range(0, zm.shape[1], 3), zm.columns[::3], rotation=45, fontsize=8)
    sp = list(zm.columns).index(str(M.SPLIT.to_period("M")))
    ax.axvline(sp - 0.5, color=PAL[1], lw=2)
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("15분 수익률이 정확히 0인 봉의 비율(%)")
    emit(f"![정지 비율]({save(fig, 'a3_zero_share')})")
    emit()
    emit("**그림**: 월별 가격 정지(15분 수익률 0) 비율. 주황선 오른쪽이 평가 기간.")
    emit()
    zr = pd.Series({x["tk"].replace("KRW-", ""): (x["zero_te"], x["zero_tr"]) for x in st})
    more = sum(a > b for a, b in zr.values)
    emit(f"**읽는 법과 해석**: 진할수록 그 달에 가격이 한 칸도 안 움직인 15분 봉이 많다. 20종목 중 {more}종목에서 평가 기간의 정지 비율이 "
         f"학습 기간보다 높다(종목 중앙 학습 {np.median([b for a, b in zr.values]):.1%} → 평가 {np.median([a for a, b in zr.values]):.1%}). "
         "평가 기간에 시장 유동성이 줄었다는 뜻이며, 정지를 학습 평균으로 고정하지 않고 최근 정지 이력으로 따로 예측하는 두 부분 모형의 근거다. "
         "BTC·ETH·SOL처럼 정지가 거의 없는 종목은 이 처리의 영향을 받지 않는다.")
    cols = list(zm.columns)
    seg = {"초기(2023-10~2024-01)": [c for c in cols if c <= "2024-01"],
           "중기(2024-02~2025-07)": [c for c in cols if "2024-02" <= c <= "2025-07"],
           "분할 직전(2025-08~2025-10)": [c for c in cols if "2025-08" <= c <= "2025-10"],
           "평가(2025-11~)": [c for c in cols if c >= "2025-11"]}
    med = {k: float(np.nanmedian(zm[v].values)) for k, v in seg.items() if v}
    emit("시기별 정지 비율(종목·월 중앙값): " + ", ".join(f"{k} {v:.1%}" for k, v in med.items()) + ". 정지 비율은 평가 기간에 갑자기 생긴 것이 "
         "아니라 수집 초기에 높았다가 중기에 낮아지고 2025년 8월부터 다시 오르는 U자 모양이다. 학습 구간 안에도 정지가 많던 시기가 있어 분류기가 "
         "그 패턴을 배울 수 있었지만, 평가 기간의 수준은 학습 구간 전체 평균보다 높다(B2절의 과소예측과 연결된다).")
    emit()
    # A4 ACF
    fig, ax = plt.subplots(figsize=(10, 4.5))
    tab = []
    for H in M.HORIZONS_H:
        a = np.nanmedian(np.array([x["acf"][H] for x in st]), axis=0)
        ax.plot(range(1, 61), a, color=H_COLOR[H], lw=2, label=M.hlabel(H))
        below = int(np.argmax(a < 0.2)) + 1 if np.any(a < 0.2) else None
        tab.append((H, a[0], a[9], a[59], below))
    ax.axhline(0.2, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("시차(블록 수)")
    ax.set_ylabel("자기상관(종목 중앙값)")
    ax.legend(frameon=False, fontsize=9)
    style(ax)
    emit(f"![자기상관]({save(fig, 'a4_acf')})")
    emit()
    emit("**그림**: 블록 로그 RV의 자기상관(종목 중앙값). 예측 구간별로 '과거 변동성이 다음 변동성을 얼마나 설명하나'를 본다.")
    emit()
    emit("| 예측 구간 | 시차 1 | 시차 10 | 시차 60 | 0.2 아래로 처음 떨어지는 시차(블록) |")
    emit("| :--- | ---: | ---: | ---: | ---: |")
    for H, a1, a10, a60, b in tab:
        emit(f"| {M.hlabel(H)} | {a1:.3f} | {a10:.3f} | {a60:.3f} | {b if b else '60 안에서 안 떨어짐'} |")
    emit()
    emit(f"**읽는 법과 해석**: 직전 블록과의 자기상관은 15분 {tab[0][1]:.2f}에서 12시간 {tab[-1][1]:.2f}로 예측 구간이 길수록 높다. 짧은 구간의 "
         "RV는 정지·호가 단위로 생기는 잡음이 커서 바로 앞 값과도 덜 비슷하고, 긴 구간은 잡음이 합쳐지며 평균으로 줄어 지속성이 또렷해진다. "
         "자기상관이 수십 블록에 걸쳐 천천히 줄어드는 것(장기기억)은 긴 이력을 보는 모델(GARCH 재귀, 순환망, 긴 문맥의 파운데이션 모델)이 "
         "유리할 수 있다는 근거이고, 짧은 구간의 낮은 자기상관은 모델끼리 차이를 내기 어려운 이유다.")
    emit()


# %% [markdown]
# ## B. 적용 변경 점검

# %%
def section_b(st: list[dict]) -> None:
    emit("## B. 적용한 처리가 의도대로 작동했는가")
    emit()
    emit("### B1. 시간 격자 복원: 옛 '행 4개 = 1시간' 창의 실제 시간 길이")
    emit()
    emit("| 종목 | 60분 넘는 창 비율(전체) | 60분 넘는 창 비율(평가) | 평가 창 길이 99% 분위(분) |")
    emit("| :--- | ---: | ---: | ---: |")
    for x in sorted(st, key=lambda z: -z["span_over60_te"]):
        emit(f"| {x['tk'].replace('KRW-', '')} | {x['span_over60_all']:.1%} | {x['span_over60_te']:.1%} | {x['span_p99_te']:.0f} |")
    emit()
    worst = max(st, key=lambda z: z["span_over60_te"])
    emit(f"**읽는 법과 해석**: 업비트는 체결이 없는 15분에는 캔들을 만들지 않으므로, 행 4개를 1시간으로 보면 실제로는 더 긴 시간이 된다. "
         f"평가 기간에 60분을 넘는 창이 가장 많은 종목은 {worst['tk'].replace('KRW-', '')}({worst['span_over60_te']:.1%})이고, BTC 등 유동성 "
         "높은 종목은 0%에 가깝다. 24·25번은 이 왜곡을 안고 있었고, 26번부터 15분 시간 격자로 복원해 모든 창이 정확히 같은 시간 길이가 됐다. "
         "저유동 종목의 결과가 24·25번과 달라진 원인 중 하나다.")
    emit()
    emit("### B2. 정지 확률의 신뢰도(예측한 정지 확률 대 실제 정지 비율)")
    emit()
    z = np.load(R26C / "26c_recent_twopart_test_predictions.npz")
    zp = np.load(X27.PI_NPZ)
    rows = []
    for H in (15, 30, 60):
        P, A = [], []
        for x in st:
            k = f"{x['tk']}|{H}|π_평가"
            if k not in zp.files:
                raise KeyError(f"정지 확률 묶음에 {k}가 없다: {X27.PI_NPZ.name}. 27번을 --family pi로 다시 돌려 묶음을 만들라")
            pt = zp[k]
            a = z[f"{x['tk']}|{H}|실제"].astype(float)
            if len(pt) != len(a):
                raise RuntimeError(f"정지 확률과 실제값의 길이가 다르다: {k} {len(pt)} 대 {len(a)}")
            P.append(pt); A.append((a == 0).astype(float))
        P, A = np.concatenate(P), np.concatenate(A)
        bins = np.array([0, .02, .05, .1, .2, .3, .4, .5, .7, 1.0])
        b = np.digitize(P, bins[1:-1])
        for k in range(len(bins) - 1):
            sel = b == k
            if sel.sum() >= 30:
                rows.append({"H": H, "구간": f"{bins[k]:.2f}~{bins[k + 1]:.2f}", "π평균": P[sel].mean(), "실제정지": A[sel].mean(), "n": int(sel.sum())})
        rows.append({"H": H, "구간": "전체", "π평균": P.mean(), "실제정지": A.mean(), "n": len(P),
                     "Brier": float(np.mean((P - A) ** 2)), "Brier_기저": float(np.mean((A.mean() - A) ** 2))})
    rel = pd.DataFrame(rows)
    rel.to_csv(RES / f"{STEM}_pi_reliability.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot([0, 0.8], [0, 0.8], color=MUTED, lw=1, ls="--", label="완벽한 신뢰도")
    for H in (15, 30, 60):
        g = rel[(rel["H"] == H) & (rel["구간"] != "전체")]
        ax.plot(g["π평균"], g["실제정지"], color=H_COLOR[H], marker="o", ms=6, lw=2, label=M.hlabel(H), markeredgecolor="white")
    ax.set_xlabel("예측한 정지 확률 π(구간 평균)")
    ax.set_ylabel("실제 정지 비율")
    ax.legend(frameon=False, fontsize=9)
    style(ax, "정지 확률 신뢰도(평가 구간, 20종목 합산)")
    emit(f"![신뢰도]({save(fig, 'b2_pi_reliability')})")
    emit()
    emit("| 예측 구간 | π 평균 | 실제 정지 비율 | Brier(분류기) | Brier(상수, 평가 평균 사용) |")
    emit("| :--- | ---: | ---: | ---: | ---: |")
    for _, x in rel[rel["구간"] == "전체"].iterrows():
        emit(f"| {M.hlabel(int(x['H']))} | {x['π평균']:.3f} | {x['실제정지']:.3f} | {x['Brier']:.4f} | {x['Brier_기저']:.4f} |")
    emit("| 4시간·12시간 | 해당 없음 | 해당 없음 | 해당 없음 | 해당 없음(정지 표본이 거의 없어 분류기 미적합, π=0) |")
    emit()
    emit("**읽는 법과 해석**: 점이 대각선 위에 있으면 \"π=0.3이라고 예측한 시점의 30%에서 실제로 정지\"라는 뜻으로, 확률이 믿을 만하다. "
         "대각선보다 아래면 정지를 과대예측, 위면 과소예측이다. 상수 기준 Brier는 평가 구간의 실제 평균 정지율을 미리 안다고 가정한 값이라 "
         "분류기에 불리한 비교인데도, 분류기가 이보다 작으면 시점마다 정지 가능성을 가려낸다는 뜻이다.")
    tot = rel[rel["구간"] == "전체"]
    under = [f"{M.hlabel(int(x['H']))} π {x['π평균']:.3f} 대 실제 {x['실제정지']:.3f}" for _, x in tot.iterrows()]
    above = all((g_["실제정지"] >= g_["π평균"] - 0.01).mean() >= 0.8 for _, g_ in rel[rel["구간"] != "전체"].groupby("H"))
    emit(f"**이번 결과**: {', '.join(under)}로 분류기가 정지를 **평균적으로 낮게 예측**한다"
         + ("(신뢰도 그림의 점이 거의 모두 대각선 위)." if above else ".") + " 분류기는 학습 구간으로 맞춰졌는데 평가 기간의 정지가 그보다 많기 "
         "때문이다(A3절). 그래도 Brier는 분류기 쪽이 작아서, 시점별로 정지 가능성의 높낮이는 가려낸다(순서는 맞고 수준이 낮다). 결과적으로 두 부분 "
         "모형은 정지 시점의 손실을 줄이지만 그 효과는 완전하지 않다. 수준까지 맞추려면 내부검증 구간의 정지율로 π를 재보정하는 방법이 있으나, "
         "이번 회차는 26c와 같은 조건을 유지하려고 적용하지 않았다.")
    emit()
    emit("### B3. MS-GARCH 제약 모수화 전후")
    emit()
    d = pd.read_csv(R26C / "26c_recent_twopart_ms_refit_diagnostics.csv")
    g = d.groupby("H")[["c_원본", "c_제약", "QLIKE_원본", "QLIKE_제약"]].agg(["min", "median", "max"]).round(3)
    emit("| 예측 구간 | 보정계수 원본(최소~최대) | 보정계수 제약(최소~최대) | QLIKE 원본(중앙) | QLIKE 제약(중앙) |")
    emit("| :--- | :--- | :--- | ---: | ---: |")
    for H, x in g.iterrows():
        emit(f"| {M.hlabel(int(H))} | {x[('c_원본', 'min')]:.3f}~{x[('c_원본', 'max')]:.3f} | {x[('c_제약', 'min')]:.3f}~{x[('c_제약', 'max')]:.3f} | "
             f"{x[('QLIKE_원본', 'median')]:.3f} | {x[('QLIKE_제약', 'median')]:.3f} |")
    emit()
    nb = int((d.groupby("종목")["전체_경계접촉"].first()).sum())
    emit(f"**읽는 법과 해석**: 원본 모수화에서는 일부 종목의 보정계수가 0.04까지 떨어졌다(내부학습 적합이 경계에 붙어 분산을 수십 배 과대예측). "
         f"제약 모수화 후에는 모든 칸이 {d['c_제약'].min():.2f}~{d['c_제약'].max():.2f}로 정상 범위이고 20종목 모두 수렴했다. 다만 {nb}종목은 "
         "자유도 등이 허용 범위의 경계에 닿아 있어(꼬리가 아주 두꺼운 저유동 종목), 그 종목의 MS-GARCH는 꼬리 추정이 한계에 걸린 상태라는 단서가 붙는다.")
    emit()


# %% [markdown]
# ## C. 결과 점검

# %%
def seed_losses(store: dict, models: list[str]) -> dict:
    """(종목, H) → 모델 → 시각별 QLIKE(시드 평균). 시드 집합은 27번과 같다."""
    out = {}
    seeds = [s for s in X27.SEEDS_USED if s != 0]
    miss = [s for s in seeds if not (R27 / f"27_model_expansion_seed{s}_test_predictions.npz").exists()]
    if miss:
        raise RuntimeError(f"[시드 완전성 게이트: 27b] 시드 {miss}의 예측 파일이 없다(RUN27_SEEDS={X27.SEEDS_USED})")
    seed_preds = {s: M._npz_to_store(R27 / f"27_model_expansion_seed{s}_test_predictions.npz") for s in seeds}
    M.check_seed_preds(store, {s: {k: v["preds"] for k, v in sp.items()} for s, sp in seed_preds.items()},
                       X27.STOCHASTIC_ALL, "27b")
    for key, S in store.items():
        a = S["act"].astype(float)
        d = {}
        for nm in models:
            if nm not in S["preds"]:
                continue
            ls = [M.qlike_vec(a, S["preds"][nm].astype(float))]
            for s in seeds:
                p = seed_preds[s].get(key, {}).get("preds", {}).get(nm)
                if p is not None:
                    ls.append(M.qlike_vec(a, p.astype(float)))
            if nm in X27.STOCHASTIC_ALL and len(ls) != 1 + len(seeds):
                raise RuntimeError(f"{key} {nm}: 시드 {len(ls)}개만 있다(기대 {1 + len(seeds)}개)")
            d[nm] = np.mean(ls, axis=0)
        out[key] = d
    return out


def section_c(rd: pd.DataFrame | None, store: dict | None) -> None:
    emit("## C. 결과 점검")
    emit()
    if rd is None:
        emit("**결과 대기**: 27번 결합 결과(`27_model_expansion_model_comparison.csv`)가 아직 없다. 신경망 시드 0이 끝나 예비 결과가 생기면 "
             "이 절이 채워진다.")
        emit()
        return
    models = [m_ for m_ in M.ALL_MODELS if m_ in set(rd["모델"]) and m_ != "naive"]
    emit("### C1. 모델별 예측 품질 이상치")
    emit()
    rows = []
    for (tk, H), S in store.items():
        a = S["act"].astype(float)
        pos = a > 0
        for nm in models:
            if nm not in S["preds"]:
                continue
            p = S["preds"][nm].astype(float)
            rows.append({"종목": tk, "H": H, "모델": nm, "로그상관": float(np.corrcoef(np.log(a[pos]), np.log(p[pos]))[0, 1])})
    q = pd.DataFrame(rows).merge(rd[["종목", "H", "모델", "MZ_R2", "보정계수", "트리밍비율"]], on=["종목", "H", "모델"], how="left")
    q.to_csv(RES / f"{STEM}_model_quality.csv", index=False)
    s = q.groupby(["모델", "H"]).agg(상관=("로그상관", "median"), 상관최소=("로그상관", "min"), MZ=("MZ_R2", "median"),
                                    c=("보정계수", "median"), 트리밍=("트리밍비율", "max")).reset_index()
    flag = s[(s["상관"] < 0.1) | (s["트리밍"] > 0.05)]
    emit("| 모델 | " + " | ".join(f"{M.hlabel(H)} 상관(최소)" for H in M.HORIZONS_H) + " |")
    emit("| :--- | " + " | ".join(["---:"] * len(M.HORIZONS_H)) + " |")
    for nm in models:
        cells = []
        for H in M.HORIZONS_H:
            x = s[(s["모델"] == nm) & (s["H"] == H)]
            if not len(x):
                cells.append("제외(TTM 지원 해상도 밖)" if (nm == "TTM" and H in X27.TTM_UNSUPPORTED_H) else "해당 없음(산출물 없음)")
            else:
                cells.append(f"{x['상관'].iloc[0]:.3f} ({x['상관최소'].iloc[0]:.3f})")
        emit(f"| {nm} | " + " | ".join(cells) + " |")
    emit()
    # 종목 단위 판정: 중앙값으로 먼저 묶으면 한 종목의 품질 저하가 가려진다(2026-10-10 Codex 리뷰 지적).
    rowflag = q[(q["로그상관"] < 0.1) | (q["트리밍비율"] > 0.05)].sort_values("로그상관")
    rowtxt = ", ".join(f"{r_.종목.replace('KRW-', '')} {r_.모델} {M.hlabel(int(r_.H))}({r_.로그상관:.3f})"
                       for r_ in rowflag.itertuples()) or "해당 없음(모든 종목·구간·모델 칸이 기준 안)"
    emit("**읽는 법과 해석**: 셀은 예측과 실제의 로그 상관(RV>0 시점)의 종목 중앙값과 괄호 안 최솟값이다. 판정 기준은 상관 0.1 미만 또는 "
         "필터로 바뀐 예측 5% 초과이고, 두 단위로 따로 본다. "
         f"(1) 모델·구간 단위(종목 중앙값): {', '.join(f'{r_.모델} {M.hlabel(int(r_.H))}' for r_ in flag.itertuples()) or '해당 없음(중앙값이 모두 기준 안)'}. "
         f"(2) 종목 단위(종목 × 구간 × 모델 {len(q):,}칸 각각): {len(rowflag)}칸 — {rowtxt}. "
         "종목 단위 이상치는 그 종목에서 해당 모델의 예측이 실제와 거의 같이 움직이지 않았다는 뜻이고, 그 손실은 이미 QLIKE 평균에 들어 있다. "
         "이상치 칸의 모델이 어느 구간·국면에서든 1위인지는 27d 결과로 따로 확인한다.")
    emit()
    emit("### C2. 26c 공유 모델 값 대조")
    emit()
    o = pd.read_csv(R26C / "26c_recent_twopart_model_comparison.csv")
    mm = rd.merge(o[["종목", "H", "모델", "QLIKE"]], on=["종목", "H", "모델"], suffixes=("", "_26c"))
    diff = float((mm["QLIKE"] - mm["QLIKE_26c"]).abs().max()) if len(mm) else np.nan
    emit(f"26c의 {mm['모델'].nunique()}개 모델 {len(mm)}칸을 결합 결과와 대조했다. QLIKE 최대 차이 {diff:.2e}. "
         + ("**같다**(결합 과정에서 기존 결과가 바뀌지 않았다)." if diff < 1e-9 else "**다르다 — 결합 과정 확인 필요**."))
    emit()
    emit("### C3. 예측-실제 겹쳐 그리기(결과를 보기 전에 정한 예시 모델)")
    emit()
    for H, days in ((60, 21), (720, 330)):
        fig, axes = plt.subplots(len(REP_TICKERS), 1, figsize=(14, 3.0 * len(REP_TICKERS)), sharex=False,
                                 gridspec_kw={"hspace": 0.45})   # 칸 제목이 위 칸 x축 눈금과 겹치지 않게
        for ax, tk in zip(axes, REP_TICKERS, strict=True):
            S = store.get((tk, H))
            if S is None:
                ax.text(0.5, 0.5, "해당 없음(결과 없음)", transform=ax.transAxes, ha="center")
                continue
            T = pd.DatetimeIndex(np.asarray(S["T"]).astype("datetime64[ns]"))
            sel = T >= T.max() - pd.Timedelta(days=days)
            ax.plot(T[sel], S["act"][sel] * 100, color="#9a9a94", lw=0.8, label="실제 RV")
            for i, nm in enumerate(REP_MODELS):
                if nm in S["preds"]:
                    ax.plot(T[sel], S["preds"][nm][sel] * 100, lw=1.3, color=PAL[i], label=nm)
            ax.set_yscale("log")
            style(ax, f"{tk.replace('KRW-', '')} · {M.hlabel(H)} 예측(%, 로그 축)")
        axes[0].legend(frameon=False, fontsize=8, ncol=6, loc="upper left")
        emit(f"![예측-실제 {M.hlabel(H)}]({save(fig, f'c3_pred_actual_{H}')})")
        emit()
    emit("**읽는 법과 해석**: 회색이 실제 RV, 색선이 각 계열 예시 모델의 예측이다(1시간은 평가 마지막 3주, 12시간은 평가 전체). 예측선이 실제의 "
         "오르내림을 따라가면 \"폭 변화 추적력\"이 있고, 평균 높이가 실제와 맞으면 \"평균 폭 정확도\"가 있다. 모든 모델이 급등 직후에야 따라 오르는 "
         "지연을 보이면 그것은 변동성 예측의 본질적 한계(충격은 예측 불가, 그 뒤의 지속만 예측 가능)다. 회색선이 그림 아래로 잘려 내려가는 곳은 RV=0(가격 정지) 시점이다. 로그 축에는 0을 그릴 수 없어 잘린다. 1시간 DOGE·BOUNTY에서 이런 시점이 잦은 것은 A절의 정지 비율과 맞는다.")
    emit()


# %% [markdown]
# ## D. 계열 단위 검정

# %%
def section_d(rd: pd.DataFrame | None, store: dict | None) -> None:
    emit("## D. 계열 단위 검정: 순차 대 병렬 어텐션 대 파운데이션")
    emit()
    if rd is None:
        emit("**결과 대기**: 27번 결합 결과가 없어 이 절은 비어 있다.")
        emit()
        return
    os.environ.setdefault("RUN26B_SRC", "27")
    B26 = _load("b26_for27b", ROOT / "test" / "models" / "26b_robustness_significance_test.py")
    models = [m_ for m_ in M.ALL_MODELS if m_ in set(rd["모델"]) and m_ != "naive"]
    L = seed_losses(store, models)
    emit("**검정 설계**")
    emit()
    emit("| 항목 | 내용 |")
    emit("| :--- | :--- |")
    emit("| 비교 단위 | 계열(통계, 하이브리드, 트리, 순환 딥러닝, 커널, 어텐션, 합성곱, 상태공간, 파운데이션) |")
    emit("| 계열 손실 | 시각 t마다, 그 계열에 속한 모든 모델의 QLIKE(무작위 모델은 시드 평균)를 종목·모델에 걸쳐 단순 평균. 대표 모델을 고르지 않으므로 선택 편향이 없다 |")
    emit("| 비교 집단(코드) | `d_t = 계열A손실_t − 계열B손실_t`, 같은 시각의 대응 표본 |")
    emit("| 귀무가설 H0 | E[d_t]=0, 두 계열의 전형적인 모델의 기대 손실이 같다 |")
    emit("| 대립가설 H1 | E[d_t]≠0 |")
    emit("| 통계량 | HAC t(Newey-West, lag=⌊4(n/100)^(2/9)⌋), 구간마다 계열 쌍 전체에 Holm 보정, 유의수준 0.05 |")
    emit("| 기각의 의미 | 한 계열의 전형적인 모델이 다른 계열보다 평균적으로 낫다(부호가 음수면 A가 낫다). 계열의 최선 모델끼리의 비교가 아니다 |")
    emit()
    rows = []
    for H in M.HORIZONS_H:
        fam_series = {}
        for fam in FAMS:
            mem = [m_ for m_ in models if M.FAMILY.get(m_) == fam]
            cols = {}
            for (tk, h), d in L.items():
                if h != H:
                    continue
                T = pd.DatetimeIndex(np.asarray(store[(tk, h)]["T"]).astype("datetime64[ns]"))
                arr = [d[m_] for m_ in mem if m_ in d]
                if arr:
                    cols[tk] = pd.Series(np.mean(arr, axis=0), index=T)
            if cols:
                fam_series[fam] = pd.DataFrame(cols).sort_index().mean(axis=1)
        fs = pd.DataFrame(fam_series).dropna()
        fams = list(fs.columns)
        for i in range(len(fams)):
            for j in range(i + 1, len(fams)):
                mu, se, p = B26.hac_mean_test((fs[fams[i]] - fs[fams[j]]).to_numpy())
                rows.append({"H": H, "A": fams[i], "B": fams[j], "평균차(A-B)": mu, "표준오차": se, "p": p, "시각수": len(fs)})
        for fam in fams:
            rows.append({"H": H, "A": fam, "B": "(평균손실)", "평균차(A-B)": float(fs[fam].mean()), "표준오차": np.nan, "p": np.nan, "시각수": len(fs)})
    t = pd.DataFrame(rows)
    pair = t[t["B"] != "(평균손실)"].copy()
    pair["p_holm"] = np.nan
    for H, g in pair.groupby("H"):
        pair.loc[g.index, "p_holm"] = B26.holm(g["p"].to_numpy())
    pair.to_csv(RES / f"{STEM}_family_dm.csv", index=False)
    lvl = t[t["B"] == "(평균손실)"].pivot_table(index="A", columns="H", values="평균차(A-B)")
    emit("### D1. 계열 평균 손실(낮을수록 좋음, 구간마다 최선 계열 대비 차)")
    emit()
    emit("| 계열 | 모델 수 | " + " | ".join(M.hlabel(H) for H in M.HORIZONS_H) + " |")
    emit("| :--- | ---: | " + " | ".join(["---:"] * len(M.HORIZONS_H)) + " |")
    for fam in FAMS:
        n = len([m_ for m_ in models if M.FAMILY.get(m_) == fam])
        if fam not in lvl.index:
            emit(f"| {fam} | {n} | " + " | ".join(["해당 없음(구성원 없음)"] * len(M.HORIZONS_H)) + " |")
            continue
        emit(f"| {fam} | {n} | " + " | ".join(f"{lvl.loc[fam, H] - lvl[H].min():+.4f}" for H in M.HORIZONS_H) + " |")
    emit()
    n_fm = {H: len([m_ for m_ in models if M.FAMILY.get(m_) == "파운데이션" and any(m_ in d for (tk, h), d in L.items() if h == H)])
            for H in M.HORIZONS_H}
    emit("**읽는 법과 해석**: 셀은 그 계열 구성원 손실의 단순 평균에서 그 구간 최선 계열의 값을 뺀 차이다(0이 최선, 클수록 나쁨). "
         "구간별 최선 계열은 " + ", ".join(f"{M.hlabel(H)} {lvl[H].idxmin()}" for H in M.HORIZONS_H) + "이다. "
         "모델 수 칸은 전체 구간 기준이며, 파운데이션 계열은 TTM 제외로 구간별 구성원 수가 "
         + ", ".join(f"{M.hlabel(H)} {n_fm[H]}" for H in M.HORIZONS_H) + "개다. 계열 평균은 약한 구성원에 끌려 내려가므로, "
         "계열의 최선 모델이 경쟁력이 있는지는 D3(MCS)와 함께 본다.")
    emit()
    emit("### D2. 계열 쌍 DM 검정(Holm 보정)")
    emit()
    for H in M.HORIZONS_H:
        g = pair[pair["H"] == H]
        best = lvl[H].idxmin()
        emit(f"**{M.hlabel(H)}** (평가 시각 {int(g['시각수'].iloc[0]):,}, 최선 계열 {best})")
        emit()
        emit("| 계열 | 최선 계열 대비 평균차 | Holm p | 판정 |")
        emit("| :--- | ---: | ---: | :--- |")
        for fam in lvl[H].sort_values().index:
            if fam == best:
                emit(f"| {fam} | 0(기준) | - | 최선 |")
                continue
            r = g[((g["A"] == fam) & (g["B"] == best)) | ((g["A"] == best) & (g["B"] == fam))].iloc[0]
            mu = r["평균차(A-B)"] * (1 if r["A"] == fam else -1)
            ok = r["p_holm"] < 0.05
            emit(f"| {fam} | {mu:+.4f} | {r['p_holm']:.3g} | {'H0 기각: 최선 계열보다 유의하게 나쁨' if ok else 'H0 기각 못함: 최선 계열과 구분되지 않음'} |")
        emit()
        tie = [f_ for f_ in lvl[H].sort_values().index if f_ != best and not (
            g[((g["A"] == f_) & (g["B"] == best)) | ((g["A"] == best) & (g["B"] == f_))]["p_holm"].iloc[0] < 0.05)]
        emit(f"**읽는 법과 해석**: 평균차는 (그 계열 − {best}) 손실의 시각 평균이고, 양수면 그 계열이 나쁘다. "
             f"Holm 보정 후 {best}와 구분되지 않는 계열은 {', '.join(tie) or '없음'}이다. 나머지 계열은 H0(기대 손실 같음)가 "
             f"기각되어, 그 계열의 전형적인 모델은 {best}의 전형적인 모델보다 평균적으로 손실이 크다.")
        emit()
    emit("### D2b. 연구 질문 쌍: 순차(순환 딥러닝) 대 병렬 어텐션 대 파운데이션")
    emit()
    emit("비교 집단과 가설은 D2와 같다(H0: 두 계열의 전형적인 모델의 기대 손실이 같다, H1: 다르다). 평균차는 (앞 계열 − 뒤 계열)이고 "
         "음수면 앞 계열이 낫다. p는 D2와 같은 구간별 전 계열 쌍 Holm 보정값이다.")
    emit()
    qs = [("딥러닝", "어텐션"), ("딥러닝", "파운데이션"), ("어텐션", "파운데이션")]
    emit("| 쌍 | " + " | ".join(M.hlabel(H) for H in M.HORIZONS_H) + " |")
    emit("| :--- | " + " | ".join([":---"] * len(M.HORIZONS_H)) + " |")
    verdict = {}
    for a_, b_ in qs:
        cells = []
        for H in M.HORIZONS_H:
            g = pair[pair["H"] == H]
            r = g[((g["A"] == a_) & (g["B"] == b_)) | ((g["A"] == b_) & (g["B"] == a_))]
            if r.empty:
                cells.append("해당 없음(계열 구성원 없음)")
                continue
            r = r.iloc[0]
            mu = r["평균차(A-B)"] * (1 if r["A"] == a_ else -1)
            win = (a_ if mu < 0 else b_) if r["p_holm"] < 0.05 else "구분 안 됨"
            verdict[(a_, b_, H)] = win
            cells.append(f"{mu:+.4f} (p={r['p_holm']:.2g}, {win})")
        emit(f"| {a_} − {b_} | " + " | ".join(cells) + " |")
    emit()
    emit("**읽는 법과 해석**: 괄호 안 마지막 항목은 Holm 보정 유의수준 0.05에서 손실이 유의하게 작은 계열이고, 유의하지 않으면 "
         "'구분 안 됨'이다. " + " ".join(
             f"{a_} 대 {b_}: " + ", ".join(f"{M.hlabel(H)} {verdict.get((a_, b_, H), '해당 없음')}" for H in M.HORIZONS_H) + "."
             for a_, b_ in qs)
         + " 순환 딥러닝은 입력이 15분봉 96개이고 어텐션·파운데이션은 H분 블록 이력이므로, 이 차이에는 입력 표현의 차이가 섞여 있다.")
    emit()
    emit("### D3. 계열별 MCS 포함 수(모델 단위 MCS, 26b 결과)")
    emit()
    f = B26.RES / f"{B26.STEM}_mcs.csv"           # 26b 모듈이 실제로 쓰는 경로(폴더 날짜가 26b 쪽에 고정돼 있다)
    if not f.exists():
        emit("**결과 대기**: 26b(27) MCS 결과가 아직 없다.")
        emit()
        return
    mc = pd.read_csv(f)
    mc["계열"] = mc["모델"].map(M.FAMILY)
    emit("| 계열 | " + " | ".join(M.hlabel(H) for H in M.HORIZONS_H) + " |")
    emit("| :--- | " + " | ".join([":---"] * len(M.HORIZONS_H)) + " |")
    for fam in FAMS:
        cells = []
        for H in M.HORIZONS_H:
            g = mc[(mc["H"] == H) & (mc["계열"] == fam)]
            cells.append("해당 없음" if not len(g) else f"{int(g['MCS포함'].sum())}/{len(g)}")
        emit(f"| {fam} | " + " | ".join(cells) + " |")
    emit()
    emit("**읽는 법과 해석**: 셀은 그 계열 모델 중 최선 모델 집합(MCS)에 남은 수/계열 모델 수다. D2는 \"계열의 전형적인 모델\", D3는 \"계열의 "
         "최선 모델\"을 본다. 두 판정이 엇갈리면(예: 계열 평균은 나쁘지만 MCS에 한 모델이 남음) 그 계열은 모델 선택에 민감하다는 뜻이다. "
         "MCS는 26b 3-1절 결과로 시드 0 예측 기준이다(시드를 바꾼 포함 횟수는 26b 3-3절). D1·D2는 시드 평균 손실 기준이다.")
    emit()


# %% [markdown]
# ## 실행

# %%
def main(argv=None) -> None:
    from report_header import study_universe
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    RES.mkdir(parents=True, exist_ok=True)
    tickers, _ = study_universe()
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        st = list(pool.map(ticker_stats, tickers))
    emit("# 27b번: EDA·결과 점검·계열 단위 검정")
    emit()
    emit(f"데이터 창 {M.DATA_START} ~ {M.DATA_END}, 분할 {M.SPLIT}, 20종목. 27번 결합 결과 "
         + ("있음." if (R27 / "27_model_expansion_model_comparison.csv").exists() else "**아직 없음(C·D절은 결과 대기)**."))
    emit()
    section_a(st)
    section_b(st)
    rd = store = None
    if (R27 / "27_model_expansion_model_comparison.csv").exists():
        rd, store, _, _ = M.load_saved("27_model_expansion") if M.RES == R27 else _load_27_saved()
    section_c(rd, store)
    section_d(rd, store)
    (RES / f"{STEM}_report.md").write_text("\n".join(_LINES), encoding="utf-8")
    print(f"[27b 완료] {RES / (STEM + '_report.md')}", flush=True)


def _load_27_saved():
    M.RES, M.STEM = R27, "27_model_expansion"
    return M.load_saved("27_model_expansion")


if __name__ == "__main__":
    main()
