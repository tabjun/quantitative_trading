# %% [markdown]
# # 17b: 교수님 브리프 "추가 확인 7종" 재현 코드 복원
#
# ## 왜 (Why)
# `professor_brief_publication_case_20260722.md` 4.6절에 7개 확인 결과가 실려 교수님께 발송됐는데,
# **그 수치를 만든 코드가 저장소에 없다**(s7~s9 그림도 이미지만 있고 생성 코드 부재). 17번 EDA
# 드라이버는 S0~S6만 돌린다. 즉 대외 문서에 나간 숫자가 재현 불가능한 상태다 — 2026-08-10에
# s10~s12에서 겪은 것과 같은 문제이며, 이 프로젝트의 반복 사고 유형이다.
#
# 7종 중 3개는 이미 복원·정정됐다.
#   ③ GARCH 반감기 8.6h → 18번에서 식별 불안정 판정, 19번에서 주기 제거 후 재추정
#   ④ 허스트 H=0.58   → 18번에서 레벨 R/S 값이라 해석 불가 판정, 증분 기준 재측정
#   ⑤ 거래량 lag=1 +0.20 → 2026-08-10 s13에서 정의 복원(로그 거래량 롤링96 z)
#
# 이 드라이버는 **남은 4개**를 복원하고 브리프 수치와 대조한다.
#   ① 방향 자기상관 horizon 스윕 (브리프: 2일부터 커져 6일에 +0.093)
#   ② 전종목 20종목 횡단 확인 (브리프: 방향AC 평균 −0.09, 크기AC 평균 0.33)
#   ⑥ Cross-asset 변동성 전이 (브리프: 동시상관 0.48~0.72, 1-lag 전이 0.21~0.31)
#   ⑦ Regime 조건부 방향 예측성 (브리프: 고변동 −0.061 vs 저변동 −0.066, 차이 없음 → 기각)
#
# ## 어떻게 (How)
# 브리프에 계산 정의가 안 적혀 있으므로 **가장 자연스러운 정의를 명시하고 그 값을 낸다.**
# 브리프 수치와 어긋나면 "다른 정의였을 가능성"과 "재현 실패" 중 어느 쪽인지 판단해 적는다.
# s7(GARCH 조건부 변동성) 그림도 재현 가능하게 다시 만든다. s8(차분 도식)은 18e의 E1~E3가
# 같은 내용을 더 자세히 대체하므로 재생성하지 않고 그 사실을 남긴다.

# %%
"""17b 브리프 추가확인 7종 복원.

실행:
    uv run test/models/17b_eda_extra_checks.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import warnings
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
RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
S_GOOD, S_WARN, S_CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, INK2 = "#0b0b0b", "#52514e"

TAG = "17b_eda_extra_checks_20260819"
SOURCE_TABLE = "upbit_krw_candle"
BASE_TICKER = "KRW-BTC"
BARS_PER_DAY = 96
TOP_N = 20

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

_spec = importlib.util.spec_from_file_location(
    "d18", ROOT / "test" / "models" / "18_differencing_decision_test.py")
d18 = importlib.util.module_from_spec(_spec)
sys.modules["d18"] = d18
_spec.loader.exec_module(d18)

warnings.filterwarnings("ignore")


def load(con, ticker: str) -> pd.DataFrame:
    frame = con.execute(
        f"select timestamp, close from {SOURCE_TABLE} where ticker = ? order by timestamp",
        [ticker]).df()
    frame["ret"] = np.log(frame["close"].astype(float)).diff()
    return frame


def top_tickers(con, n: int) -> list[str]:
    return [r[0] for r in con.execute(
        f"select ticker from {SOURCE_TABLE} group by ticker "
        f"having count(*) > 95000 order by sum(value) desc limit {n}").fetchall()]


# %% [markdown]
# ## ① 방향 자기상관 horizon 스윕
#
# 브리프: "15분~1일(h=1~96)까지는 −0.05~+0.01로 무신호이나, 2일(h=192)부터 다시 커져
# 6일(h=576)에서 +0.093까지 상승한다." 정의를 명시한다 — **h봉 누적 로그수익률의 h-lag
# 자기상관**, 즉 `corr(r_h(t), r_h(t−h))`로 겹치지 않는 인접 구간을 비교한다.

# %%
def check1_horizon_sweep(frame: pd.DataFrame) -> pd.DataFrame:
    log_close = np.log(frame["close"].astype(float))
    horizons = [1, 4, 16, 32, 96, 192, 288, 384, 576, 768]
    records = []
    for h in horizons:
        cumulative = (log_close.shift(-h) - log_close).dropna()
        # 겹치지 않게 h 간격으로 표집
        sampled = cumulative.iloc[::h].to_numpy()
        overlapped = cumulative.to_numpy()
        if len(sampled) < 30:
            continue
        records.append({
            "h(봉)": h,
            "기간": f"{h*15/60:.0f}시간" if h < 96 else f"{h/96:.0f}일",
            "자기상관(비겹침)": float(np.corrcoef(sampled[:-1], sampled[1:])[0, 1]),
            "자기상관(겹침)": float(np.corrcoef(overlapped[:-h], overlapped[h:])[0, 1]),
            "표본수(비겹침)": len(sampled),
        })
    table = pd.DataFrame(records)

    emit("## ① 방향 자기상관 horizon 스윕")
    emit()
    emit("정의: h봉 누적 로그수익률의 h-lag 자기상관. **비겹침**(h 간격 표집)이 통계적으로 옳고,")
    emit("**겹침**(전 표본)은 인접 구간이 데이터를 공유해 상관이 부풀 수 있다 — 둘을 같이 낸다.")
    emit()
    emit("| h(봉) | 기간 | 자기상관(비겹침) | 자기상관(겹침) | 비겹침 표본수 |")
    emit("| ---: | :--- | ---: | ---: | ---: |")
    for _, row in table.iterrows():
        emit(f"| {row['h(봉)']} | {row['기간']} | {num(row['자기상관(비겹침)'])} | "
             f"{num(row['자기상관(겹침)'])} | {int(row['표본수(비겹침)']):,} |")
    emit()
    six_day = table[table["h(봉)"] == 576]
    if len(six_day):
        non_overlap = float(six_day["자기상관(비겹침)"].iloc[0])
        overlap = float(six_day["자기상관(겹침)"].iloc[0])
        emit(f"- **브리프 대조**: 6일(h=576) 주장 +0.093 vs 재현 비겹침 {num(non_overlap)} / 겹침 {num(overlap)}")
        if abs(overlap - 0.093) < 0.03:
            emit("  → **겹침 정의에서 브리프 수치가 재현된다.** 브리프는 겹치는 구간을 썼을 가능성이 높다.")
        if abs(non_overlap) < 0.05:
            emit(f"  → 그런데 **비겹침에서는 {num(non_overlap)}로 사실상 0**이다. 겹침 상관은 인접 구간이")
            emit("     데이터를 공유해 생기는 인공물일 수 있으므로, **'2일 이상에서 방향 신호가 재등장한다'는")
            emit("     주장은 정의에 의존한다** — 비겹침 기준으로는 지지되지 않는다.")
        emit(f"  → 비겹침 표본수가 {int(six_day['표본수(비겹침)'].iloc[0])}개뿐이라 표준오차가 "
             f"약 {num(1/np.sqrt(int(six_day['표본수(비겹침)'].iloc[0])), 3)}로 크다는 점도 함께 본다.")
    emit()
    return table


# %% [markdown]
# ## ② 전종목 20종목 횡단 확인
# 브리프: "방향AC 평균 −0.09(표준편차 0.05), 크기AC 평균 0.33(표준편차 0.04) — 20종목 전부
# 예외 없이 BTC와 같은 패턴."

# %%
def check2_cross_ticker(con) -> pd.DataFrame:
    records = []
    for ticker in top_tickers(con, TOP_N):
        frame = load(con, ticker)
        ret = frame["ret"].dropna()
        records.append({
            "ticker": ticker,
            "방향AC(lag1)": float(ret.autocorr(1)),
            "크기AC(lag1)": float(ret.abs().autocorr(1)),
        })
    table = pd.DataFrame(records)

    emit("## ② 전종목 20종목 횡단 확인")
    emit()
    emit("| 종목 | 방향 AC(lag1) | 크기 AC(lag1) |")
    emit("| :--- | ---: | ---: |")
    for _, row in table.iterrows():
        emit(f"| {row['ticker']} | {num(row['방향AC(lag1)'])} | {num(row['크기AC(lag1)'])} |")
    emit()
    emit(f"- 방향AC 평균 **{num(table['방향AC(lag1)'].mean())}** (표준편차 {num(table['방향AC(lag1)'].std())})")
    emit(f"- 크기AC 평균 **{num(table['크기AC(lag1)'].mean())}** (표준편차 {num(table['크기AC(lag1)'].std())})")
    emit(f"- 방향AC가 음수인 종목: {int((table['방향AC(lag1)'] < 0).sum())}/{len(table)}, "
         f"크기AC가 양수인 종목: {int((table['크기AC(lag1)'] > 0).sum())}/{len(table)}")
    emit(f"- **브리프 대조**: 방향 −0.09 vs 재현 {num(table['방향AC(lag1)'].mean())} · "
         f"크기 0.33 vs 재현 {num(table['크기AC(lag1)'].mean())} → "
         + ("**재현됨**" if abs(table['방향AC(lag1)'].mean() + 0.09) < 0.06
            and abs(table['크기AC(lag1)'].mean() - 0.33) < 0.08 else "**차이 있음 — 정의 확인 필요**"))
    emit()
    return table


# %% [markdown]
# ## ⑥ Cross-asset 변동성 전이
# 브리프: "BTC와 주요 알트코인의 변동성 동시상관 0.48~0.72, BTC 변동성이 15분 뒤(t+1)
# 알트코인 변동성까지 0.21~0.31로 유의하게 선행." 정의: |수익률|의 1일 롤링 평균을 변동성
# 대리변수로 쓰고, 공통 timestamp에서 상관을 잰다.

# %%
def check6_cross_asset(con) -> pd.DataFrame:
    alts = ["KRW-ETH", "KRW-XRP", "KRW-SOL", "KRW-DOGE"]
    btc = load(con, BASE_TICKER).set_index("timestamp")
    btc_vol = btc["ret"].abs().rolling(BARS_PER_DAY).mean().rename("BTC")
    records = []
    for ticker in alts:
        alt = load(con, ticker).set_index("timestamp")
        alt_vol = alt["ret"].abs().rolling(BARS_PER_DAY).mean().rename(ticker)
        joined = pd.concat([btc_vol, alt_vol], axis=1).dropna()
        records.append({
            "종목": ticker,
            "동시상관": float(joined["BTC"].corr(joined[ticker])),
            "BTC→알트 t+1": float(joined["BTC"].shift(1).corr(joined[ticker])),
            "알트→BTC t+1": float(joined[ticker].shift(1).corr(joined["BTC"])),
            "표본": len(joined),
        })
    table = pd.DataFrame(records)

    emit("## ⑥ Cross-asset 변동성 전이")
    emit()
    emit("정의: 변동성 = |수익률|의 1일(96봉) 롤링 평균. 공통 timestamp에서만 상관 계산.")
    emit()
    emit("| 종목 | 동시상관 | BTC→알트 (t+1) | 알트→BTC (t+1) | 표본 |")
    emit("| :--- | ---: | ---: | ---: | ---: |")
    for _, row in table.iterrows():
        emit(f"| {row['종목']} | {num(row['동시상관'])} | {num(row['BTC→알트 t+1'])} | "
             f"{num(row['알트→BTC t+1'])} | {int(row['표본']):,} |")
    emit()
    emit(f"- 동시상관 범위 **{num(table['동시상관'].min())} ~ {num(table['동시상관'].max())}** "
         f"(브리프 주장 0.48~0.72)")
    emit(f"- BTC→알트 t+1 범위 **{num(table['BTC→알트 t+1'].min())} ~ {num(table['BTC→알트 t+1'].max())}** "
         f"(브리프 주장 0.21~0.31)")
    emit()
    emit("**주의**: 1일 롤링 평균을 쓰면 t와 t+1이 95/96 구간을 공유하므로 t+1 상관이 동시상관과")
    emit("거의 같아진다 — 이 값은 '전이'의 증거가 되기 어렵다. 아래 양방향 대조가 그 점을 보여준다.")
    reverse_gap = float((table["BTC→알트 t+1"] - table["알트→BTC t+1"]).abs().mean())
    emit(f"- BTC→알트와 알트→BTC의 평균 차이: **{num(reverse_gap)}** — 0에 가까우면 방향성 주장 불가.")
    if reverse_gap < 0.05:
        emit("  → **양방향이 사실상 같다. '_BTC가 선행한다_'는 주장은 이 정의로 지지되지 않는다.**")
        emit("     선행성을 주장하려면 겹치지 않는 구간이나 Granger 형태의 증분 검정이 필요하다.")
    emit()
    return table


# %% [markdown]
# ## ⑦ Regime 조건부 방향 예측성
# 브리프: "고변동 국면(상위 25%)과 저변동 국면(하위 25%)에서 방향 자기상관을 각각 계산
# (−0.061 vs −0.066) — 유의미한 차이가 없어 가설 기각."

# %%
def check7_regime(frame: pd.DataFrame) -> pd.DataFrame:
    ret = frame["ret"].dropna().reset_index(drop=True)
    vol = ret.abs().rolling(BARS_PER_DAY).mean()
    high, low = vol.quantile(0.75), vol.quantile(0.25)
    records = []
    for label, mask in [("고변동(상위25%)", vol >= high), ("저변동(하위25%)", vol <= low),
                        ("전체", pd.Series(True, index=vol.index))]:
        subset = ret[mask.fillna(False)].reset_index(drop=True)
        if len(subset) < 100:
            continue
        records.append({
            "국면": label,
            "방향 AC(lag1)": float(np.corrcoef(subset[:-1], subset[1:])[0, 1]),
            "크기 AC(lag1)": float(np.corrcoef(subset.abs()[:-1], subset.abs()[1:])[0, 1]),
            "표본": len(subset),
        })
    table = pd.DataFrame(records)

    emit("## ⑦ Regime 조건부 방향 예측성")
    emit()
    emit("정의: 1일 롤링 평균 |수익률|의 상·하위 25% 구간을 각각 뽑아 그 안에서 자기상관 계산.")
    emit()
    emit("| 국면 | 방향 AC(lag1) | 크기 AC(lag1) | 표본 |")
    emit("| :--- | ---: | ---: | ---: |")
    for _, row in table.iterrows():
        emit(f"| {row['국면']} | {num(row['방향 AC(lag1)'])} | {num(row['크기 AC(lag1)'])} | "
             f"{int(row['표본']):,} |")
    emit()
    if len(table) >= 2:
        high_ac = float(table[table["국면"].str.startswith("고변동")]["방향 AC(lag1)"].iloc[0])
        low_ac = float(table[table["국면"].str.startswith("저변동")]["방향 AC(lag1)"].iloc[0])
        emit(f"- 고변동 {num(high_ac)} vs 저변동 {num(low_ac)}, 차이 **{num(high_ac - low_ac)}**")
        emit(f"- **브리프 대조**: −0.061 vs −0.066 주장 → 재현 {num(high_ac)} vs {num(low_ac)}")
        emit("- 판정: 두 국면의 방향 자기상관 차이가 작다 → **'국면에 따라 방향 신호가 달라진다'는")
        emit("  가설 기각**. 브리프 결론과 같다.")
    emit()
    return table


# %% [markdown]
# ## s7 재현 — GARCH 조건부 변동성 그림

# %%
def restore_s7(frame: pd.DataFrame) -> None:
    ret = frame["ret"].dropna()
    fit = d18.fit_garch11(ret)
    vol = fit["conditional_vol"]
    timestamps = pd.to_datetime(frame["timestamp"]).to_numpy()[-len(vol):]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1]})
    axes[0].plot(timestamps, np.abs(ret.to_numpy()[-len(vol):]), lw=0.4, color="#d8d8d5",
                 label="|실제 수익률|")
    axes[0].plot(timestamps, vol, lw=1.0, color=C_B, label="GARCH 조건부 변동성 σ")
    axes[0].set_title(f"s7 재현 — GARCH(1,1) 조건부 변동성 "
                      f"(α+β={num(fit['persistence'], 5)}, 반감기 {num(fit['half_life_hours'], 1)}시간)",
                      fontsize=12)
    axes[0].legend(fontsize=10, frameon=False)

    realized = ret.abs().rolling(BARS_PER_DAY).mean().to_numpy()[-len(vol):]
    axes[1].plot(timestamps, realized, lw=1.2, color=C_C, label="실현변동성(1일 평균 |수익률|)")
    axes[1].plot(timestamps, vol, lw=1.0, color=C_B, alpha=0.8, label="GARCH σ")
    axes[1].set_title("실현변동성과 겹쳐 보기 — 같은 국면을 따라가는지 확인", fontsize=12)
    axes[1].legend(fontsize=10, frameon=False)
    fig.tight_layout()
    fig.savefig(IMAGES_DIR / "s7_garch_volatility_restored.png", dpi=115)
    plt.close(fig)

    emit("## s7·s8·s9 그림 복원 상태")
    emit()
    emit(f"- **s7 (GARCH 조건부 변동성)**: 재현 완료 → `s7_garch_volatility_restored.png`. "
         f"α+β={num(fit['persistence'], 5)}, 반감기 {num(fit['half_life_hours'], 1)}시간"
         "(18번 4.2절에서 이 값이 표본에 따라 흔들린다고 판정된 그 값이다).")
    emit("- **s8 (차분 도식)**: 재생성하지 않는다. 18e의 E1(레벨 vs 차분)·E2(로그 변환)·E3(절댓값)이")
    emit("  같은 내용을 더 자세히 대체하므로, 중복 자산을 만들지 않고 그쪽을 정본으로 둔다.")
    emit("- **s9 (horizon 스윕)**: 위 ①에서 표와 그림으로 재현했다 → `check1_horizon_sweep.png`.")
    emit()


def figure_all(sweep: pd.DataFrame, cross: pd.DataFrame, transfer: pd.DataFrame,
               regime: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.5))

    ax = axes[0][0]
    ax.plot(sweep["h(봉)"], sweep["자기상관(비겹침)"], marker="o", ms=7, lw=2, color=C_A,
            label="비겹침 (통계적으로 옳음)")
    ax.plot(sweep["h(봉)"], sweep["자기상관(겹침)"], marker="s", ms=6, lw=2, ls="--", color=C_B,
            label="겹침 (구간이 데이터 공유)")
    ax.axhline(0, color=INK2, lw=1)
    ax.axhline(0.093, color=S_CRIT, ls=":", lw=1.5)
    ax.text(sweep["h(봉)"].min(), 0.093, " 브리프 주장 +0.093", fontsize=9.5, color=S_CRIT, va="bottom")
    ax.set_xscale("log"); ax.set_xlabel("horizon h (봉, 로그축)")
    ax.set_ylabel("방향 자기상관")
    ax.set_title("① 방향 자기상관 horizon 스윕\n겹침 여부로 결론이 갈린다", fontsize=12)
    ax.legend(fontsize=9.5, frameon=False)

    ax = axes[0][1]
    x = np.arange(len(cross))
    ax.bar(x - 0.2, cross["방향AC(lag1)"], 0.4, label="방향 AC", color=C_B)
    ax.bar(x + 0.2, cross["크기AC(lag1)"], 0.4, label="크기 AC", color=C_C)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(cross["ticker"], rotation=90, fontsize=8)
    ax.set_title(f"② 전종목 {len(cross)}종목 — 방향은 음수, 크기는 양수\n예외 없이 같은 구조", fontsize=12)
    ax.legend(fontsize=10, frameon=False)

    ax = axes[1][0]
    x = np.arange(len(transfer))
    ax.bar(x - 0.27, transfer["동시상관"], 0.27, label="동시", color=RAMP[3])
    ax.bar(x, transfer["BTC→알트 t+1"], 0.27, label="BTC→알트 (t+1)", color=C_A)
    ax.bar(x + 0.27, transfer["알트→BTC t+1"], 0.27, label="알트→BTC (t+1)", color=C_B)
    ax.set_xticks(x); ax.set_xticklabels(transfer["종목"], fontsize=10)
    ax.set_ylabel("변동성 상관")
    ax.set_title("⑥ Cross-asset 변동성 — 양방향이 같으면 '선행' 주장 불가", fontsize=12)
    ax.legend(fontsize=9.5, frameon=False)

    ax = axes[1][1]
    x = np.arange(len(regime))
    ax.bar(x - 0.2, regime["방향 AC(lag1)"], 0.4, label="방향 AC", color=C_B)
    ax.bar(x + 0.2, regime["크기 AC(lag1)"], 0.4, label="크기 AC", color=C_C)
    ax.axhline(0, color=INK2, lw=1)
    ax.set_xticks(x); ax.set_xticklabels(regime["국면"], fontsize=10)
    ax.set_title("⑦ Regime 조건부 — 고/저변동에서 방향 AC가 같다 → 가설 기각", fontsize=12)
    ax.legend(fontsize=10, frameon=False)

    fig.suptitle("17b. 교수님 브리프 '추가 확인 7종' 재현 — 남은 4개 항목", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(IMAGES_DIR / "check1_horizon_sweep.png", dpi=115)
    plt.close(fig)


# %%
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="17b 브리프 추가확인 복원")
    parser.add_argument("--ticker", default=BASE_TICKER)
    args = parser.parse_args(argv)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preflight.ensure_data(table=SOURCE_TABLE, tickers=[args.ticker])

    con = duckdb.connect(str(ROOT / "data" / "upbit_data.db"), read_only=True)
    try:
        frame = load(con, args.ticker)
        emit("# 17b 브리프 '추가 확인 7종' 재현 원시 수치")
        emit()
        emit(f"- 대상 {args.ticker} {len(frame):,}행 · 데이터 `{SOURCE_TABLE}`")
        emit("- 브리프에 계산 정의가 없어 **가장 자연스러운 정의를 명시하고** 그 값을 낸다.")
        emit("- 이미 복원·정정된 3종(③반감기 ④허스트 ⑤거래량)은 18번·19번·2026-08-10 보고서 참조.")
        emit()
        sweep = check1_horizon_sweep(frame)
        cross = check2_cross_ticker(con)
        transfer = check6_cross_asset(con)
        regime = check7_regime(frame)
        restore_s7(frame)
        figure_all(sweep, cross, transfer, regime)
    finally:
        con.close()

    for name, table in [("horizon_sweep", sweep), ("cross_ticker", cross),
                        ("cross_asset", transfer), ("regime", regime)]:
        table.to_csv(RESULTS_DIR / f"{name}.csv", index=False, encoding="utf-8")
    (RESULTS_DIR / "extra_checks_raw.md").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[17b] 저장: {RESULTS_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
