"""Reusable UTF-8 email sender for research reports.

Usage examples:
    python test/scripts/send_email.py
    python test/scripts/send_email.py --preset text_context
    python test/scripts/send_email.py --preset independent_variables
    python test/scripts/send_email.py --preset historical_flow_mart
"""

from __future__ import annotations

import argparse
import os
import smtplib
import subprocess
from email.message import EmailMessage
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
GITHUB_BASE = "https://github.com/tabjun/personal_ai_project"
REPO_BLOB_BASE = f"{GITHUB_BASE}/blob/stock/quantitative_trading"


def load_env() -> None:
    for env_path in (ROOT / "test/.env", ROOT / "test/scripts/.env", ROOT / ".env"):
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip())


def normalize_sender(sender: str) -> str:
    sender = sender.strip()
    if sender and "@" not in sender:
        return f"{sender}@naver.com"
    return sender


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


def github_blob(path: str) -> str:
    return f"{REPO_BLOB_BASE}/{path}"


def simulation_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    body = f"""교수님 안녕하세요.

실시간 모의투자 시뮬레이션 결과 보고 메일입니다.

커밋 링크:
{commit_url}

이번 메일은 `analysis_report.md`와 `analysis_report.pdf`가 존재하면 첨부해 전송하도록 구성했습니다.
기본 용도는 `pipelines/simulate_and_send.py`가 생성한 결과 전달입니다.

감사합니다.
"""
    attachments = [ROOT / "analysis_report.md", ROOT / "analysis_report.pdf"]
    subject = "[시계열/퀀트 연구] 시뮬레이션 결과 보고"
    return subject, body, attachments


def text_context_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_path = "test/results/text_context_feature_report_20260608_012237.md"
    body = f"""교수님 안녕하세요.

실시간 텍스트 데이터(뉴스, 리포트, SNS)를 독립변수로 반영하는 환경 구축 내용을 공유드립니다.

커밋 링크:
{commit_url}

보고서:
{github_blob(report_path)}

주요 코드:
- contexts/text_context.py: {github_blob('contexts/text_context.py')}
- pipelines/ingest_text_context.py: {github_blob('pipelines/ingest_text_context.py')}
- pipelines/simulate_and_send.py: {github_blob('pipelines/simulate_and_send.py')}

핵심 반영 사항:
1. `text_events_raw`, `text_features_15m` DuckDB 테이블 추가
2. 감성, 이벤트 수, shock, 토픽 count 기반 독립변수 구성
3. `text_risk_guard`를 통한 신규 진입 차단 로직 추가

감사합니다.
"""
    subject = "[시계열/퀀트 연구] 실시간 텍스트 독립변수 반영 환경 구축 보고"
    return subject, body, []


def independent_variables_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_path = (
        "test/research_materials/independent_variables_literature_review_20260608.md"
    )
    body = f"""교수님 안녕하세요.

주식/코인 예측용 독립변수 후보를 논문 근거 기반으로 정리한 보고서를 공유드립니다.

커밋 링크:
{commit_url}

보고서:
{github_blob(report_path)}

핵심 내용:
1. 가격/변동성/유동성/오더북 변수
2. 뉴스/리포트/SNS 감성 변수
3. 매크로/글로벌 유동성/온체인/파생상품 변수
4. 논문별 5단계 포맷 정리 및 실제 변수 설계안 반영

감사합니다.
"""
    subject = "[시계열/퀀트 연구] 독립변수 설계 논문 조사 보고"
    return subject, body, []


def historical_flow_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_path = "test/research_materials/historical_flow_datamart_research_20260608.md"
    body = f"""교수님 안녕하세요.

과거 유사 사건/흐름을 KRW 전체 종목 기준으로 조회하는 historical flow data mart 설계 및 구현 내용을 공유드립니다.

커밋 링크:
{commit_url}

보고서:
{github_blob(report_path)}

주요 코드:
- marts/historical_flow.py: {github_blob('marts/historical_flow.py')}
- pipelines/build_historical_flow_mart.py: {github_blob('pipelines/build_historical_flow_mart.py')}
- pipelines/query_historical_flows.py: {github_blob('pipelines/query_historical_flows.py')}

핵심 내용:
1. KRW 전체 종목 원천 저장 + 유동성 subset index
2. `shape + factor + context` 복합 유사도
3. 텍스트 컨텍스트를 포함한 변곡점 원인 매칭

감사합니다.
"""
    subject = "[시계열/퀀트 연구] Historical Flow Data Mart 구축 보고"
    return subject, body, []


PRESETS = {
    "simulation": simulation_email,
    "text_context": text_context_email,
    "independent_variables": independent_variables_email,
    "historical_flow_mart": historical_flow_email,
}


def build_message(preset: str, sender: str, receiver: str) -> tuple[EmailMessage, list[Path]]:
    commit_hash = git_head_short()
    subject, body, attachments = PRESETS[preset](commit_hash)
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.set_content(body, subtype="plain", charset="utf-8", cte="base64")
    return msg, attachments


def attach_files(msg: EmailMessage, attachments: list[Path]) -> None:
    for path in attachments:
        if not path.exists():
            continue
        data = path.read_bytes()
        maintype = "application"
        subtype = "octet-stream"
        if path.suffix.lower() == ".pdf":
            subtype = "pdf"
        msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send reusable UTF-8 report email.")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESETS.keys()),
        default="simulation",
        help="Email template preset. Default keeps compatibility with simulate_and_send.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env()
    sender = normalize_sender(os.environ.get("NAVER_EMAIL_ID", ""))
    password = os.environ.get("NAVER_APP_PASSWORD", "").strip()
    receiver = os.environ.get("RECEIVER_EMAIL", "").strip()
    if not sender or not password or not receiver:
        raise SystemExit("Missing NAVER_EMAIL_ID, NAVER_APP_PASSWORD, or RECEIVER_EMAIL")

    msg, attachments = build_message(args.preset, sender, receiver)
    attach_files(msg, attachments)

    with smtplib.SMTP_SSL("smtp.naver.com", 465, timeout=30) as server:
        server.login(sender, password)
        server.send_message(msg)
    print(f"Sent preset '{args.preset}' email to {receiver}")


if __name__ == "__main__":
    main()
