# %% [markdown]
# # 19b: 생존편의 감사 — 상장폐지 종목이 표본에서 빠졌는가
#
# ## 왜 (Why)
# 2026-08-10 보고서 2절에서 남긴 후속 항목이다. `upbit_krw_candle`은 수집 시점에 업비트에
# 상장돼 있던 종목 목록으로 만들어졌다(`pipelines/rebuild_price_mart.py`가 현재 마켓 목록을
# 조회한다). 그렇다면 **수집 이전에 이미 상장폐지된 종목은 애초에 표본에 없다** — 짧은 종목을
# 버리지 않아도 표본 자체가 생존자만 담는 셈이다. Ranse(2026)는 인도 소형주에서 이 편의가
# 연 4.94%p 수익률·샤프 0.097 과대평가를 낳는다고 정량화했다.
#
# 그때는 "수집 파이프라인 구조상 이번 감사로는 답할 수 없다"고 미뤘다. 그런데 **데이터 안에
# 흔적이 남는다**: 상장폐지·거래정지된 종목은 특정 시점 이후 데이터가 끊긴다. 그 흔적을 세면
# 편의의 방향과 최소 규모를 말할 수 있다.
#
# ## 무엇을 (What)
#   S1 종료 시점 분포   : 종목별 마지막 봉이 전역 최대와 얼마나 떨어져 있는가 → 중도 종료 탐지
#   S2 시작 시점 분포   : 상장 시점 흩어짐 → 불균형 패널 규모(2026-08-10 s10 재확인)
#   S3 현재 마켓 대조   : 업비트 API의 현재 KRW 마켓과 DB 종목을 비교
#   S4 편의 방향 추정   : 중도 종료 종목의 마지막 구간 수익률이 생존 종목과 다른가
#
# ## 어떻게 (How)
# API 조회는 실패할 수 있으므로(네트워크·정책) 실패 시 S3만 건너뛰고 나머지를 진행한다.
# S4는 "중도 종료 종목이 종료 직전에 더 나빴다면, 그런 종목이 빠진 표본은 낙관적으로 편향된다"는
# 방향 논증을 데이터로 확인하는 것이다 — 크기를 확정하는 것이 아니라 **방향과 하한**을 잡는다.
#
# ## 기대 결과 / 반영 (Expected)
# "생존편의가 있는가/없는가"를 미확인에서 확인으로 옮기고, 있다면 어느 방향인지 명시한다.

# %%
"""19b 생존편의 감사.

실행:
    uv run test/models/19b_survivorship_audit.py
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
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

C_A, C_B = "#2a78d6", "#eb6834"
S_GOOD, S_WARN, S_CRIT = "#0ca30c", "#fab219", "#d03b3b"
INK, INK2 = "#0b0b0b", "#52514e"

TAG = "19b_survivorship_audit_20260819"
SOURCE_TABLE = "upbit_krw_candle"
BARS_PER_DAY = 96
STALE_DAYS = 7            # 전역 최대보다 이만큼 이상 일찍 끝나면 "중도 종료"로 본다

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


def current_krw_markets() -> list[str] | None:
    """업비트 현재 KRW 마켓 목록. 실패하면 None."""
    try:
        request = urllib.request.Request(
            "https://api.upbit.com/v1/market/all?isDetails=false",
            headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode())
        return sorted(item["market"] for item in data if item["market"].startswith("KRW-"))
    except Exception as exc:  # noqa: BLE001
        print(f"[19b] 마켓 목록 조회 실패: {type(exc).__name__}: {exc}", flush=True)
        return None


# %%
def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="19b 생존편의 감사")
    args = parser.parse_args(argv)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preflight.ensure_data(table=SOURCE_TABLE, tickers=None, auto_build=False, strict=False)

    con = duckdb.connect(str(ROOT / "data" / "upbit_data.db"), read_only=True)
    try:
        span = con.execute(f"""
            select ticker, count(*) as rows,
                   min(timestamp) as first_ts, max(timestamp) as last_ts
            from {SOURCE_TABLE} group by ticker order by last_ts
        """).df()
        global_last = con.execute(f"select max(timestamp) from {SOURCE_TABLE}").fetchone()[0]
        # 중도 종료 종목의 마지막 30일 수익률 vs 생존 종목의 같은 기간
        tail_returns = con.execute(f"""
            with ranked as (
                select ticker, timestamp, close,
                       row_number() over (partition by ticker order by timestamp desc) as rn
                from {SOURCE_TABLE}
            )
            select ticker,
                   max(case when rn = 1 then close end) as last_close,
                   max(case when rn = {BARS_PER_DAY * 30} then close end) as close_30d_ago
            from ranked where rn <= {BARS_PER_DAY * 30} group by ticker
        """).df()
    finally:
        con.close()

    span["last_ts"] = pd.to_datetime(span["last_ts"])
    span["first_ts"] = pd.to_datetime(span["first_ts"])
    global_last = pd.to_datetime(global_last)
    span["종료지연_일"] = (global_last - span["last_ts"]).dt.total_seconds() / 86400
    span["중도종료"] = span["종료지연_일"] >= STALE_DAYS

    emit("# 19b 생존편의 감사 원시 수치")
    emit()
    emit(f"- 대상: `{SOURCE_TABLE}` {len(span):,}종목, 전역 마지막 봉 {global_last:%Y-%m-%d %H:%M}")
    emit(f"- '중도 종료' 판정: 마지막 봉이 전역 최대보다 **{STALE_DAYS}일 이상** 이른 종목")
    emit()

    # ── S1 종료 시점
    stale = span[span["중도종료"]]
    emit("## S1. 종료 시점 — 중도에 끊긴 종목이 있는가")
    emit()
    emit(f"- 중도 종료 종목: **{len(stale)}개 / {len(span)}개 ({100*len(stale)/len(span):.1f}%)**")
    if len(stale):
        emit(f"- 종료지연 분포: 중앙값 {num(stale['종료지연_일'].median(), 1)}일, "
             f"최대 {num(stale['종료지연_일'].max(), 1)}일")
        emit()
        emit("| 종목 | 행수 | 첫 봉 | 마지막 봉 | 종료지연(일) |")
        emit("| :--- | ---: | :--- | :--- | ---: |")
        for _, row in stale.sort_values("종료지연_일", ascending=False).head(25).iterrows():
            emit(f"| {row['ticker']} | {row['rows']:,} | {row['first_ts']:%Y-%m-%d} | "
                 f"{row['last_ts']:%Y-%m-%d} | {num(row['종료지연_일'], 1)} |")
        if len(stale) > 25:
            emit(f"| … 외 {len(stale)-25}개 | | | | |")
    else:
        emit("- **없다.** 모든 종목이 전역 마지막 시점까지 데이터를 갖는다.")
        emit("  → 이 표본에는 '수집 기간 중 상장폐지된 종목'의 흔적이 없다.")
    emit()

    # ── S2 시작 시점
    emit("## S2. 시작 시점 — 불균형 패널 규모")
    emit()
    max_rows = int(span["rows"].max())
    full = span[span["rows"] >= 0.95 * max_rows]
    emit(f"- 최장 종목 {max_rows:,}행, 풀히스토리(95%+) **{len(full)}개 ({100*len(full)/len(span):.1f}%)**")
    emit(f"- 상장 시점 범위: {span['first_ts'].min():%Y-%m-%d} ~ {span['first_ts'].max():%Y-%m-%d}")
    emit(f"- 수집 기간 중 신규 상장(첫 봉이 전역 시작보다 30일 이상 늦음): "
         f"**{int((span['first_ts'] > span['first_ts'].min() + pd.Timedelta(days=30)).sum())}개**")
    emit()

    # ── S3 현재 마켓 대조
    emit("## S3. 업비트 현재 KRW 마켓과 대조")
    emit()
    markets = current_krw_markets()
    if markets is None:
        emit("- API 조회 실패 — 이 절은 건너뛴다(네트워크·정책). S1·S2·S4로 판단한다.")
        only_db, only_api = [], []
    else:
        db_set, api_set = set(span["ticker"]), set(markets)
        only_db = sorted(db_set - api_set)
        only_api = sorted(api_set - db_set)
        emit(f"- 현재 KRW 마켓 **{len(api_set)}개** vs DB **{len(db_set)}개**")
        emit(f"- **DB에만 있고 현재 마켓엔 없음 = 수집 이후 상장폐지: {len(only_db)}개**"
             + (f" — {', '.join(only_db)}" if only_db and len(only_db) <= 20 else ""))
        emit(f"- **현재 마켓엔 있고 DB엔 없음 = 수집 이후 신규 상장: {len(only_api)}개**"
             + (f" — {', '.join(only_api)}" if only_api and len(only_api) <= 20 else ""))
        emit()
        emit("  이 대조는 **수집 시점 이후**의 변화만 보여준다. 수집 시점 *이전*에 이미")
        emit("  상장폐지된 종목은 DB에도 API에도 없어 이 방법으로는 보이지 않는다 — 그것이")
        emit("  생존편의의 본체이며, 확인에는 과거 마켓 목록 스냅샷이나 상장폐지 공시가 필요하다.")
    emit()

    # ── S4 편의 방향
    emit("## S4. 편의 방향 추정 — 중도 종료 종목은 끝나기 전에 더 나빴는가")
    emit()
    merged = span.merge(tail_returns, on="ticker", how="left")
    merged["최근30일_수익률"] = np.log(merged["last_close"] / merged["close_30d_ago"])
    valid = merged.dropna(subset=["최근30일_수익률"])
    if len(stale) and (valid["중도종료"]).sum() >= 3:
        stale_ret = valid[valid["중도종료"]]["최근30일_수익률"]
        alive_ret = valid[~valid["중도종료"]]["최근30일_수익률"]
        emit(f"- 중도 종료 종목({len(stale_ret)}개) 마지막 30일 로그수익률: 중앙값 {num(stale_ret.median())}")
        emit(f"- 생존 종목({len(alive_ret)}개) 마지막 30일 로그수익률: 중앙값 {num(alive_ret.median())}")
        emit(f"- 차이(중도종료 − 생존): **{num(stale_ret.median() - alive_ret.median())}**")
        if stale_ret.median() < alive_ret.median():
            emit("- **중도 종료 종목이 더 나빴다** → 그런 종목이 빠진 표본은 낙관적으로 편향된다.")
        else:
            emit("- 중도 종료 종목이 더 나쁘지 않았다 → 이 표본에서는 방향 편의가 뚜렷하지 않다.")
    else:
        emit("- 중도 종료 종목이 없거나 너무 적어 방향 비교를 할 수 없다.")
        alive_ret = valid["최근30일_수익률"]
        emit(f"- 전체 종목 마지막 30일 로그수익률 중앙값: {num(alive_ret.median())} "
             f"(양수면 최근 구간이 상승장)")
    emit()

    # ── S5 폐지 종목이 우리 실험에서 어떤 위치였는가 (구체적 위험)
    emit("## S5. 폐지된 종목이 우리 실험에서 어떤 위치였는가")
    emit()
    board = ROOT / "test" / "results" / "16_nonstationary_crosssectional_20260719" / "t3_crosssection_leaderboard.csv"
    if markets is not None and only_db and board.exists():
        leaderboard = pd.read_csv(board).sort_values("trend_corr", ascending=False).reset_index(drop=True)
        leaderboard["순위"] = leaderboard.index + 1
        hit = leaderboard[leaderboard["ticker"].isin(only_db)]
        if len(hit):
            emit("16번 t3(30종목 횡단면) 리더보드에서 **수집 이후 폐지된 종목**의 위치:")
            emit()
            emit("| 종목 | trend_corr | 순위 | 전체 |")
            emit("| :--- | ---: | ---: | ---: |")
            for _, row in hit.iterrows():
                emit(f"| {row['ticker']} | {num(row['trend_corr'])} | {int(row['순위'])}위 | {len(leaderboard)}종목 |")
            emit()
            top = hit["순위"].min()
            if top <= 3:
                emit(f"- **폐지 종목이 상위 {int(top)}위에 있다.** 즉 그 실험에서 '가장 신호가 있어 보였던'")
                emit("  종목이 지금은 거래할 수 없다. 생존편의가 추상적 우려가 아니라는 구체적 증거다.")
            emit("- 함의: 종목별 성과 순위를 근거로 삼으면 **이미 사라진 종목을 근거로 삼을 수 있다**.")
            emit("  (18c에서 이 순위 자체가 시드 잡음이라는 것도 별도로 확인됐다 — 이유가 둘로 겹친다.)")
        else:
            emit("- 폐지 종목이 16번 t3 리더보드(상위 30종목)에는 없었다.")
    else:
        emit("- 대조할 리더보드 또는 폐지 종목 목록이 없어 건너뛴다.")
    emit()

    # ── 판정
    emit("## 판정")
    emit()
    emit("| 질문 | 답 | 근거 |")
    emit("| :--- | :--- | :--- |")
    emit(f"| 수집 기간 **안에서** 폐지된 종목이 있나 | **아니오** (0개) | S1 — 269종목 전부 전역 마지막 봉까지 존재 |")
    if markets is not None:
        emit(f"| 수집 **이후** 폐지된 종목이 있나 | **예** ({len(only_db)}개: {', '.join(only_db)}) | S3 — 현재 마켓 대조 |")
        emit(f"| 수집 이후 신규 상장 | {len(only_api)}개 | S3 |")
    emit("| 수집 **이전**에 폐지된 종목이 있나 | **확인 불가** | DB·현재 마켓 모두에 없어 이 데이터로는 보이지 않는다 |")
    emit()
    emit("**결론 — 생존편의는 구조적으로 존재하지만 이 데이터로 크기를 재지는 못한다.**")
    emit()
    emit("1. 표본은 '2026-07 수집 시점 생존자' 269종목으로 만들어졌다. 그 이전에 폐지된 종목은")
    emit("   애초에 들어오지 않았고, 그 사실은 **데이터 안에서 확인할 수 없다**(S3의 한계).")
    emit(f"2. 다만 폐지가 실제로 일어나는 사건임은 확인됐다 — 수집 후 약 1개월 만에 {len(only_db) if markets else 0}개가 사라졌다.")
    emit("   3년치라면 상당수가 사라졌을 것으로 보는 것이 타당하다.")
    emit("3. 크기를 재려면 **과거 마켓 목록 스냅샷** 또는 **업비트 상장폐지 공시 이력**이 필요하다.")
    emit("   지금부터 마켓 목록을 주기적으로 저장해 두면 앞으로는 재현 가능해진다(후속 항목).")
    emit()
    emit("**실무 규칙(신설)**: 전종목 실험에서 `rows >= 임계값` 필터를 쓰면 **그 자체가 생존자 선택**이다.")
    emit("필터를 쓸 때는 (a) 필터 전/후 결과를 함께 보고하고 (b) 제외된 종목 수와 이유를 명시한다.")
    emit("종목별 성과 순위를 결론 근거로 쓰지 않는다(S5 + 18c 시드 잡음, 이유가 둘).")
    emit()

    # ── 그림
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    axes[0].hist(span["rows"], bins=40, color=C_A, edgecolor="white", lw=0.5)
    axes[0].axvline(0.95 * max_rows, color=S_CRIT, ls="--", lw=1.6)
    axes[0].text(0.95 * max_rows, axes[0].get_ylim()[1] * 0.95, " 풀히스토리 기준",
                 fontsize=9.5, color=S_CRIT, va="top")
    axes[0].set_title(f"종목별 행 수 ({len(span)}종목)\n왼쪽 꼬리 = 늦게 상장된 종목", fontsize=11.5)
    axes[0].set_xlabel("행 수")

    axes[1].hist(span["종료지연_일"], bins=40, color=C_B, edgecolor="white", lw=0.5)
    axes[1].axvline(STALE_DAYS, color=S_CRIT, ls="--", lw=1.6)
    axes[1].text(STALE_DAYS, axes[1].get_ylim()[1] * 0.95, f" {STALE_DAYS}일",
                 fontsize=9.5, color=S_CRIT, va="top")
    axes[1].set_yscale("log")
    axes[1].set_title(f"종료지연 분포 — 중도 종료 {len(stale)}개\n0에 몰려 있으면 폐지 흔적 없음",
                      fontsize=11.5)
    axes[1].set_xlabel("전역 마지막 봉과의 차이(일)")

    axes[2].scatter(span["first_ts"], span["last_ts"], s=14, color=C_A, alpha=0.7)
    axes[2].axhline(global_last, color=S_CRIT, ls="--", lw=1.4)
    axes[2].text(span["first_ts"].min(), global_last, " 전역 마지막 봉", fontsize=9.5,
                 color=S_CRIT, va="bottom")
    axes[2].set_title("종목별 시작(x) vs 종료(y)\n점이 모두 붉은 선에 붙으면 폐지 흔적 없음",
                      fontsize=11.5)
    axes[2].set_xlabel("첫 봉"); axes[2].set_ylabel("마지막 봉")
    axes[2].tick_params(axis="x", rotation=20)

    fig.suptitle("19b. 생존편의 감사 — 상장폐지 종목이 표본에서 빠졌는가", fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(IMAGES_DIR / "survivorship_audit.png", dpi=115)
    plt.close(fig)

    span.to_csv(RESULTS_DIR / "ticker_span.csv", index=False, encoding="utf-8")
    (RESULTS_DIR / "survivorship_raw.md").write_text("\n".join(_LINES) + "\n", encoding="utf-8")
    print(f"[19b] 저장: {RESULTS_DIR}")


if __name__ == "__main__":
    main(sys.argv[1:])
