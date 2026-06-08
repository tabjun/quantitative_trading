import os
import smtplib
from email.headerregistry import Address
from email.message import EmailMessage
from pathlib import Path


def load_env_files() -> None:
    for env_path in [Path("test/.env"), Path("test/scripts/.env"), Path(".env")]:
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


def normalize_naver_sender(sender: str) -> str:
    sender = sender.strip()
    if sender and "@" not in sender:
        return f"{sender}@naver.com"
    return sender


def build_message(sender: str, receiver: str) -> EmailMessage:
    commit_url = "https://github.com/tabjun/personal_ai_project/commit/cb847a3"
    report_url = (
        "https://github.com/tabjun/personal_ai_project/blob/stock/"
        "quantitative_trading/test/results/text_context_feature_report_20260608_012237.md"
    )
    readme_url = (
        "https://github.com/tabjun/personal_ai_project/blob/stock/"
        "quantitative_trading/README.md#7-realtime-text-context-pipeline"
    )
    text_context_url = (
        "https://github.com/tabjun/personal_ai_project/blob/stock/"
        "quantitative_trading/contexts/text_context.py"
    )
    ingest_url = (
        "https://github.com/tabjun/personal_ai_project/blob/stock/"
        "quantitative_trading/pipelines/ingest_text_context.py"
    )
    sim_url = (
        "https://github.com/tabjun/personal_ai_project/blob/stock/"
        "quantitative_trading/pipelines/simulate_and_send.py"
    )

    body = f"""교수님 안녕하세요.

앞서 보내드린 메일에서 한글 본문 인코딩이 깨져 보여 다시 정정하여 보내드립니다.
이번 메일은 UTF-8 문자셋을 명시해 발송했습니다.

[커밋 링크]
{commit_url}

[구성 문서]
{readme_url}

[검증 리포트]
{report_url}

이번 작업의 목적은 시계열 기반 퀀트 분석 프로젝트에 실시간 텍스트 데이터(뉴스, 증시 리포트, SNS성 텍스트)를 독립변수로 반영할 수 있는 환경을 구축하는 것이었습니다.

1. 실시간 텍스트 수집 계층
- 기본 수집원은 Google News RSS입니다. 별도 키 없이 동작합니다.
- NAVER_CLIENT_ID, NAVER_CLIENT_SECRET이 있으면 Naver News Search API를 자동으로 사용할 수 있습니다.
- TEXT_LOCAL_CSVS 환경변수로 증권사 리포트 CSV, SNS export CSV를 추가할 수 있습니다.
- TEXT_RSS_URLS 환경변수로 사용자 지정 뉴스/리포트 RSS 피드를 확장할 수 있습니다.

2. DuckDB 텍스트 데이터마트
- text_events_raw: 원문 텍스트, 출처, 발행시각, 감성 점수, 토픽 플래그를 저장합니다.
- text_features_15m: 기존 15분봉 시계열과 결합 가능한 독립변수 테이블입니다.
- 생성 변수는 text_event_count, text_sentiment_mean, text_shock_z, text_sentiment_momentum_1h, text_macro_count, text_risk_count, text_crypto_count, text_regulation_count, text_liquidity_count 등입니다.

3. 분석 코드 반영
- pipelines/simulate_and_send.py에 텍스트 피처 결합 로직을 추가했습니다.
- 악재성 텍스트가 강하고 리스크 토픽이 감지되면 신규 진입을 차단하는 text_risk_guard를 추가했습니다.
- 리포트 테이블에는 텍스트 이벤트 수, 감성 점수, 쇼크 Z-score, 리스크 가드 상태가 함께 표시되도록 했습니다.

4. 검증 결과
- 이번 실행에서 RSS 텍스트 90건을 refresh했습니다.
- DuckDB raw table에는 총 91건의 텍스트 row가 유지되고 있습니다.
- 기존 15분봉 가격 인덱스와 정렬된 text_features_15m 500행을 생성했습니다.
- 문법 검증은 contexts/text_context.py, pipelines/ingest_text_context.py, pipelines/simulate_and_send.py 모두 통과했습니다.

[주요 코드 링크]
- contexts/text_context.py: {text_context_url}
- pipelines/ingest_text_context.py: {ingest_url}
- pipelines/simulate_and_send.py: {sim_url}

다음 단계는 실시간 Upbit 캔들 마트를 최신 시점까지 갱신하여, 2026-06-08 텍스트 타임스탬프와 가격 타임스탬프가 실제로 겹치도록 만든 뒤, 학교 서버 커널에서 텍스트 독립변수가 반영된 백테스트/리포트 pass를 수행하는 것입니다.

감사합니다.
"""

    msg = EmailMessage()
    local_part, domain = sender.split("@", 1)
    msg["From"] = Address(display_name="문태준", username=local_part, domain=domain)
    msg["To"] = receiver
    msg["Subject"] = "[시계열/퀀트 연구] 실시간 텍스트 독립변수 반영 환경 구축 보고 재발송"
    msg.set_content(body, subtype="plain", charset="utf-8", cte="base64")
    return msg


def main() -> None:
    load_env_files()
    sender = normalize_naver_sender(os.environ.get("NAVER_EMAIL_ID", ""))
    password = os.environ.get("NAVER_APP_PASSWORD", "").strip()
    receiver = os.environ.get("RECEIVER_EMAIL", "").strip()
    if not sender or not password or not receiver:
        raise SystemExit("Missing NAVER_EMAIL_ID, NAVER_APP_PASSWORD, or RECEIVER_EMAIL")

    msg = build_message(sender, receiver)
    with smtplib.SMTP_SSL("smtp.naver.com", 465, timeout=30) as server:
        server.login(sender, password)
        server.send_message(msg)
    print(f"UTF-8 email sent to {receiver}")


if __name__ == "__main__":
    main()
