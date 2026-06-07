"""Send the independent-variable literature report as UTF-8 Korean email."""

from __future__ import annotations

import os
import smtplib
import subprocess
from email.message import EmailMessage
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = (
    "test/research_materials/"
    "independent_variables_literature_review_20260608.md"
)
GITHUB_BASE = "https://github.com/tabjun/personal_ai_project"
REPORT_URL = f"{GITHUB_BASE}/blob/stock/quantitative_trading/{REPORT_PATH}"


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

주식 및 코인 예측 모델에서 사용할 독립변수 후보를 최근 논문 근거에 맞춰 정리한 보고서를 작성하여 공유드립니다.

보고서 링크:
{REPORT_URL}

커밋 링크:
{commit_url}

이번 문서는 실제 로컬 학습, 백테스트, 노트북 실행 없이 논문 조사와 변수 설계만 수행한 결과입니다.

핵심 내용은 다음과 같습니다.

1. 가격/수익률/변동성 변수
- log return, realized volatility, high-low range, Garman-Klass volatility 등 기본 가격 구조를 설명하는 변수를 우선 구성했습니다.

2. 기술지표 및 유동성 변수
- RSI, ROC, Bollinger Band Width, moving-average spread, breakout distance, volume z-score, Amihud illiquidity 등을 포함했습니다.

3. 텍스트 기반 변수
- 뉴스, 증시 리포트, SNS 텍스트에서 감성 평균/합계/최소/최대, topic-level sentiment, text_event_count, text_shock_z 등을 만들도록 설계했습니다.
- 한국어 금융 텍스트는 entity-level/aspect-level sentiment와 TGT-Masking 방식의 entity leakage 통제 필요성을 함께 반영했습니다.

4. 매크로/레짐 변수
- CPI, 실업률, 금리, 장단기 금리차, 환율, 인플레이션 surprise, Global M2 liquidity, DXY, VIX 등을 포함했습니다.
- 코인에서는 글로벌 유동성, 온체인 지표, 펀딩비, 미결제약정, 청산 규모를 별도 후보로 정리했습니다.

5. 오더북/시장 미시구조 변수
- bid-ask spread, order imbalance, depth imbalance, multi-level limit order book volume profile을 단기 예측용 변수로 정리했습니다.

보고서에는 각 논문별로 요약, 서론, 분석 기법, 결과, 결론 및 설계 결정 항목을 분리해 작성했습니다.

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
        raise SystemExit(
            "Missing NAVER_EMAIL_ID, NAVER_APP_PASSWORD, or RECEIVER_EMAIL"
        )

    commit_hash = git_head_short()
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = "[시계열/퀀트 연구] 독립변수 설계 논문 근거 보고"
    msg.set_content(
        build_body(commit_hash),
        subtype="plain",
        charset="utf-8",
        cte="base64",
    )

    with smtplib.SMTP_SSL("smtp.naver.com", 465, timeout=30) as server:
        server.login(sender, password)
        server.send_message(msg)

    print(f"Sent UTF-8 report email to {receiver}")


if __name__ == "__main__":
    main()
