"""서비스 진입점: 한 번 실행하면 시세 → 뉴스 → 지표 묶음 → 규칙 기준선 → (선택) LLM 판단을 순서대로 돈다.

    python main.py                 # LLM 판단 없이 수집·지표·기준선(기본, 연구 GPU 우선 기간)
    python main.py --judge         # LLM 판단까지(단발, 사람이 켤 때만)
    python main.py --no-news       # 뉴스 수집 생략(빠른 확인용)

단계 구성과 각 단계의 책임은 `pipelines/service_pipeline.py`에 있다. 진행 상태와 자원 기록은 `logs/service/`.
(2026-10-18 교체: 이전 main.py는 키워드·종목을 고정한 시연용 데모였다.)
"""

from __future__ import annotations

import argparse
import sys

from pipelines.service_pipeline import ServiceConfig, ServicePipeline


def parse_args(argv: list[str] | None = None) -> ServiceConfig:
    ap = argparse.ArgumentParser(description="LLM 위험 판단 서비스 파이프라인을 한 번 실행한다.")
    ap.add_argument("--judge", action="store_true", help="LLM 판단 단계까지 실행(GPU 단발)")
    ap.add_argument("--tickers", default=None, help="LLM 판단 종목(쉼표 구분). 없으면 전부")
    ap.add_argument("--no-news", action="store_true", help="뉴스 수집 생략")
    ap.add_argument("--candles", type=int, default=400, help="종목당 받을 최근 15분봉 수")
    a = ap.parse_args(argv)
    return ServiceConfig(run_judge=a.judge, judge_tickers=a.tickers.split(",") if a.tickers else None,
                         collect_news=not a.no_news, candle_count=a.candles)


def main(argv: list[str] | None = None) -> int:
    pipeline = ServicePipeline(parse_args(argv))
    ctx = pipeline.run()
    print(ServicePipeline.summary(ctx))
    return 1 if any("error" in r for r in ctx.results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
