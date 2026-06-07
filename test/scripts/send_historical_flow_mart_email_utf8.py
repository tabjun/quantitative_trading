"""Send the historical flow mart report as UTF-8 Korean email."""

from __future__ import annotations

import os
import smtplib
import subprocess
from email.message import EmailMessage
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = "test/research_materials/historical_flow_datamart_research_20260608.md"
GITHUB_BASE = "https://github.com/tabjun/personal_ai_project"
REPORT_URL = f"{GITHUB_BASE}/blob/stock/quantitative_trading/{REPORT_PATH}"
MART_URL = f"{GITHUB_BASE}/blob/stock/quantitative_trading/historical_flow_mart.py"
BUILD_URL = f"{GITHUB_BASE}/blob/stock/quantitative_trading/build_historical_flow_mart.py"
QUERY_URL = f"{GITHUB_BASE}/blob/stock/quantitative_trading/query_historical_flows.py"


def load_env() -> None:
    for env_path in [ROOT / "test/.env", ROOT / "test/scripts/.env", ROOT / ".env"]:
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def git_head_short() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return result.stdout.strip()


def build_body(commit_hash: str) -> str:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    return f"""교수님 안녕하세요.

업비트 KRW 마켓 전체 종목을 대상으로 과거 유사 사건과 흐름을 미리 계산해 저장하는 historical flow data mart 설계 및 1차 코드 구축 내용을 공유드립니다.

보고서 링크:
{REPORT_URL}

구축 코드 링크:
- historical_flow_mart.py: {MART_URL}
- build_historical_flow_mart.py: {BUILD_URL}
- query_historical_flows.py: {QUERY_URL}

커밋 링크:
{commit_url}

핵심 내용은 다음과 같습니다.

1. 전체 KRW 마켓 기준 설계
- 단일 비트코인 기준이 아니라 `ticker` 축을 포함한 `upbit_krw_candle` 전체 KRW 마켓 원천 테이블을 기준으로 설계했습니다.
- 전체 원천 window는 저장하고, 기본 검색 인덱스는 거래대금 상위 유동성 종목 subset으로 구성하도록 했습니다.

2. 단순 차트 형태 비교의 한계 보완
- 과거 유사 사건은 가격 모양만 같다고 같은 사건이 아니므로, `shape_distance`, `factor_distance`, `context_distance`를 합친 복합 거리로 매칭하도록 설계했습니다.
- factor에는 변동성, MDD, trend slope, 거래대금/거래량 z-score, RSI, ROC, Bollinger Band Width, Amihud illiquidity가 포함됩니다.

3. 변곡점 당시 원인/상황 매칭
- 이미 구축된 `text_features_15m`이 존재하면 감성, 이벤트 수, shock z-score, 리스크/매크로/규제/유동성 토픽 count를 historical window에 자동 결합합니다.
- 이를 통해 현재 흐름의 원인이 과거 변곡점의 원인과 맞는지 함께 평가할 수 있습니다.

4. 이용 방식
- 서버 또는 자동화 환경에서 `uv run build_historical_flow_mart.py --window-lengths 16,48,96,288 --stride 4 --top-k 10 --liquid-top-n 50` 형태로 사전 마트를 구축합니다.
- 분석 시점에는 `uv run query_historical_flows.py --ticker KRW-SOL --window-length 96 --top-k 10`처럼 현재 종목의 최근 흐름과 가장 유사한 과거 사건을 조회합니다.
- 조회 결과의 `query_composite_distance`가 낮을수록 가격 형태, 독립변수 상태, 당시 원인/맥락이 함께 유사한 과거 사례입니다.

5. 저장 테이블
- `historical_flow_windows`
- `historical_flow_features`
- `historical_flow_neighbors`
- `historical_flow_event_stats`
- `historical_regime_stats`
- `historical_flow_run_log`

6. 연구 근거
- TimeRAG의 DTW 기반 time-series knowledge base
- kNN-MTS의 cached series datastore
- sliced Wasserstein k-means의 다차원 market regime 분류
- DTW lead-lag 분석
- Matrix Profile 기반 motif/discord 탐색

이번 작업은 코드 작성과 경량 문법/소형 synthetic 검증만 수행했고, 로컬에서 전체 KRW 마켓 빌드나 대규모 백테스트/학습은 실행하지 않았습니다.

감사합니다.
"""


def main() -> None:
    load_env()
    sender = os.environ.get("NAVER_EMAIL_ID", "").strip()
    password = os.environ.get("NAVER_APP_PASSWORD", "").strip()
    receiver = os.environ.get("RECEIVER_EMAIL", "").strip()
    if sender and "@" not in sender:
        sender = f"{sender}@naver.com"
    if not sender or not password or not receiver:
        raise SystemExit("Missing NAVER_EMAIL_ID, NAVER_APP_PASSWORD, or RECEIVER_EMAIL")

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = "[시계열/퀀트 연구] 과거 유사 사건 데이터마트 구축 보고"
    msg.set_content(build_body(git_head_short()), subtype="plain", charset="utf-8", cte="base64")

    with smtplib.SMTP_SSL("smtp.naver.com", 465, timeout=30) as server:
        server.login(sender, password)
        server.send_message(msg)
    print(f"Sent UTF-8 historical flow mart email to {receiver}")


if __name__ == "__main__":
    main()
