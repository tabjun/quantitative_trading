"""DuckDB에 적재된 15분봉 데이터를 빠르게 조회·시각화하는 확인용 도구.

사용:
    uv run test/scripts/quick_look.py                 # 상위 20종목 종가 그래프
    uv run test/scripts/quick_look.py --normalize     # 시작=100 정규화(같은 축에서 비교)
    uv run test/scripts/quick_look.py --col volume    # 거래량으로
    uv run test/scripts/quick_look.py --single        # 20개를 한 축에 겹쳐 그리기

DB 구조(long form):
    ticker(VARCHAR) | timestamp(TIMESTAMP) | open/high/low/close/volume/value(DOUBLE)
    269종목 × 15분봉, 약 1,608만 행, 2023-07-18 ~ 2026-07-19
    → 상위 20종목은 조회 시 필터링한다(DB 자체에는 269종목 전부 있음).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["Noto Sans CJK KR", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "data" / "upbit_data.db"
sys.path.insert(0, str(ROOT / "test" / "scripts"))
from report_header import study_universe  # noqa: E402


def load_long(tickers: list[str], col: str = "close") -> pd.DataFrame:
    """상위 종목만 long form으로 조회."""
    con = duckdb.connect(str(DB), read_only=True)
    q = f"""
        SELECT ticker, timestamp, {col}
        FROM upbit_krw_candle
        WHERE ticker IN ({','.join(['?'] * len(tickers))})
        ORDER BY ticker, timestamp
    """
    df = con.execute(q, tickers).df()
    con.close()
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--col", default="close", help="close/volume/value 등")
    ap.add_argument("--normalize", action="store_true", help="시작=100으로 정규화")
    ap.add_argument("--single", action="store_true", help="한 축에 20개 겹쳐 그리기")
    ap.add_argument("--out", default="test/images/quick_look.png")
    a = ap.parse_args()

    tickers, _ = study_universe()          # 거래대금 상위10 + 변동성 상위10
    print(f"조회 종목 {len(tickers)}개: {', '.join(tickers)}")

    df = load_long(tickers, a.col)
    print(f"\nlong form shape: {df.shape}")
    print(f"컬럼: {list(df.columns)}")
    print(f"ticker.unique() 개수: {df['ticker'].nunique()}")   # ← 20이 나온다
    print(f"기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
    print("\n샘플 5행:")
    print(df.head().to_string(index=False))

    # wide form으로 피벗(시각화용)
    wide = df.pivot(index="timestamp", columns="ticker", values=a.col)
    wide = wide[tickers]                    # 선정 순서 유지
    if a.normalize:
        wide = wide / wide.bfill().iloc[0] * 100

    if a.single:
        fig, ax = plt.subplots(figsize=(15, 7))
        for t in wide.columns:
            ax.plot(wide.index, wide[t], lw=0.7, alpha=.8, label=t.replace("KRW-", ""))
        ax.set_yscale("log")
        ax.legend(fontsize=7, ncol=4)
        ax.set_title(f"상위 20종목 {a.col}" + (" (시작=100 정규화, 로그축)" if a.normalize else " (로그축)"))
        ax.grid(alpha=.25)
    else:
        fig, axes = plt.subplots(4, 5, figsize=(20, 12), sharex=True)
        for ax, t in zip(axes.flat, wide.columns):
            ax.plot(wide.index, wide[t], lw=0.6, color="#1f77b4")
            ax.set_title(t.replace("KRW-", ""), fontsize=10)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=.2)
        for ax in axes.flat[len(wide.columns):]:
            ax.axis("off")
        fig.suptitle(f"상위 20종목 {a.col} 시계열 (15분봉)" +
                     (" — 시작=100 정규화" if a.normalize else ""), fontsize=15, y=1.00)

    fig.tight_layout()
    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\n그림 저장: {out}")


if __name__ == "__main__":
    main()
