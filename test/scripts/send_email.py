"""Reusable UTF-8 email sender for research reports.

Usage examples:
    python test/scripts/send_email.py
    python test/scripts/send_email.py --preset text_context
    python test/scripts/send_email.py --preset independent_variables
    python test/scripts/send_email.py --preset historical_flow_mart
    python test/scripts/send_email.py --preset optimization_context_brief
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

실시간 모의투자 결과 보고 메일입니다.

커밋 링크:
{commit_url}

이번 메일은 `analysis_report.md`와 `analysis_report.pdf`가 존재하면 첨부해 전송하도록 구성했습니다.
기본 용도는 `pipelines/simulate_and_send.py`가 생성한 결과 전달입니다.

감사합니다.
"""
    attachments = [ROOT / "analysis_report.md", ROOT / "analysis_report.pdf"]
    subject = "[시계열 연구] 모의투자 결과 보고"
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

감사합니다.
"""
    subject = "[시계열 연구] 실시간 텍스트 독립변수 반영 환경 구축"
    return subject, body, []


def independent_variables_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_path = "test/research_materials/independent_variables_literature_review_20260608.md"
    body = f"""교수님 안녕하세요.

주식/코인 예측용 독립변수 설계에 관한 문헌 정리 보고서를 공유드립니다.

커밋 링크:
{commit_url}

보고서:
{github_blob(report_path)}

감사합니다.
"""
    subject = "[시계열 연구] 독립변수 설계 문헌 리뷰"
    return subject, body, []


def historical_flow_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_path = "test/research_materials/historical_flow_datamart_research_20260608.md"
    body = f"""교수님 안녕하세요.

KRW 전 종목 기준 historical flow data mart 설계 및 구축 내용을 공유드립니다.

커밋 링크:
{commit_url}

보고서:
{github_blob(report_path)}

주요 코드:
- marts/historical_flow.py: {github_blob('marts/historical_flow.py')}
- pipelines/build_historical_flow_mart.py: {github_blob('pipelines/build_historical_flow_mart.py')}
- pipelines/query_historical_flows.py: {github_blob('pipelines/query_historical_flows.py')}

감사합니다.
"""
    subject = "[시계열 연구] Historical Flow Data Mart 구축 보고"
    return subject, body, []


def optimization_context_brief_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_path = "test/research_materials/optimization_context_professor_brief_20260611.md"
    body = f"""교수님 안녕하세요.

오늘 논의한 내용을 반영하여, 아래 두 방향으로 연구 문서와 테스트 코드를 수정했습니다.

1. 코인 분석 입력 설계 관점 수정
- 주식처럼 증시 리포트나 일반 뉴스 중심으로 입력을 구성하는 방식은 코인 시장 설명력에 한계가 있다고 판단했습니다.
- 그래서 코인 시장에서는 거래소 흐름, 규제/거시 이벤트, 유동성 변화, 위험 심리, SNS 반응처럼 시장 미시구조와 이벤트 전파가 더 직접적으로 반영되는 변수 구성이 필요하다는 방향으로 정리했습니다.

2. 최적화 문제 해석 관점 수정
- 비정상 시계열 학습 과정에서 보이는 문제를 단순 기울기 소실이나 성능 저하로만 보지 않고, objective와 prediction head가 허용하는 가장 쉬운 해로 붕괴하는 현상인지 먼저 확인하는 방향으로 재정리했습니다.
- 이에 따라 `/test` 연구 공간에서는 단순 리더보드식 비교가 아니라, 학습 곡선을 보면서 `0 수익률 예측`, `lag-1 복사`, `flat output` 같은 shortcut collapse를 먼저 진단하는 구조로 문서와 코드를 수정했습니다.

3. 5번 최적화 진단 코드를 경량 quick probe 구조로 재설계
- 이번 수정의 핵심은 `5_optimization_diagnostics_test`를 독립변수 전체를 붙인 본격 예측 실험이 아니라, 최적화 경로만 빠르게 확인하는 경량 진단 코드로 바꾼 것입니다.
- 기본 실행은 텍스트 독립변수를 제외하고, 업비트 시계열 내부 변화만 반영하는 작은 endogenous feature set(`log_return_1`, `return_4`, `realized_vol_16`, `hl_range_pct`, `volume_z_96`, `spread_proxy`)만 사용합니다.
- 데이터 길이도 `seq_len=32`, `max_rows=2500`, `max_windows=512`, `window_stride=4`로 줄여, 겹치는 구간을 많이 학습시키기보다 몇 분 내에 학습 곡선과 collapse 징후를 확인할 수 있게 했습니다.
- 기본 점검 케이스는 `quick_probe`로 두고, 여기서 `next_close_level + MSE`, `next_log_return + Huber`, `next_log_return + directional hybrid` 세 경우를 먼저 비교하게 했습니다.
- 아키텍처 비교도 처음부터 TCN/Transformer까지 넓히지 않고, Linear/LSTM/GRU 정도만 남겨서 objective 차이와 구조 차이를 빠르게 분리해서 보도록 했습니다.

4. 5번 코드에 반영한 최신 연구 방향
- `Huber`는 최근 금융 시계열 연구에서 turbulent regime와 outlier에 강건한 손실함수로 쓰이는 흐름을 참고했습니다. 예: Liu et al., 2025, *Adapting to the Unknown: Robust Meta-Learning for Zero-Shot Financial Time Series Forecasting* (arXiv:2504.09664).
- directional hybrid는 단순 point error 최소화만으로는 `0 수익률 근처`로 붕괴할 수 있다는 문제를 보완하기 위해, 값 예측과 방향 일치를 함께 보려는 실험적 설계입니다. 이 문제의식은 Guo et al., 2025, *A Novel Loss Function for Deep Learning Based Daily Stock Trading System* (arXiv:2502.17493)처럼 금융 목적에 더 맞는 손실함수 설계 흐름과 연결됩니다.
- 또한 Wu et al., 2024, *Review of deep learning models for crypto price prediction* (arXiv:2405.11431)를 참고해, 코인 예측에서는 다변량 접근이 중요하더라도 최적화 경로 진단 단계에서는 먼저 작은 내생 변수 집합으로 collapse 여부를 보는 것이 더 타당하다고 판단했습니다.

5. 이 코드로 앞으로 수행하려는 순서
- 1차: `quick_probe`로 level target과 return target, robust loss, directional penalty의 차이를 빠르게 확인
- 2차: collapse가 줄어드는 objective를 기준으로 `objective_probe` 확장
- 3차: 필요할 때만 `architecture_probe`로 구조 비교
- 4차: 이 단계가 정리된 후에만 텍스트 독립변수 실험(4번 코드)과 더 큰 데이터 범위로 확장

커밋 링크:
{commit_url}

5번 코드 링크:
- ipynb: {github_blob('test/models/5_optimization_diagnostics_test.ipynb')}
- py mirror: {github_blob('test/models/5_optimization_diagnostics_test.py')}

아래 링크에서 수정 내용과 관련 문서를 함께 보실 수 있습니다.

루트 프로젝트 개요 및 아키텍처:
{github_blob('README.md')}

연구 실험 공간 설명 및 최적화 문제 정리:
{github_blob('test/README.md')}

상세 브리프:
{github_blob(report_path)}

아키텍처 이미지:
{github_blob('materials/quant_architecture.png')}

특히 이번에는 메일 본문만 읽어도 어떤 점을 수정했는지 바로 파악하실 수 있도록, 수정 관점을 1번과 2번으로 먼저 적었습니다.
이번 재발송 메일에는 5번 코드의 경량화 방향, 기본 실행 구조, 참고한 최근 논문 흐름, 그리고 앞으로 실제로 어떤 순서로 실험을 진행할지를 본문에 직접 적었습니다.
상세 브리프에서는 각 문제를 왜 연구상 중요하게 보는지, 그리고 이를 코드와 실험 구조에 어떻게 반영했는지를 조금 더 풀어서 정리했습니다.

참고문헌은 통계 용어 설명과 함께 브리프 하단에 정리해 두었습니다.

감사합니다.
"""
    subject = "[시계열·코인 연구] 오늘 논의 반영: 코인 변수 설계 및 5번 최적화 quick probe 재구성"
    return subject, body, []


PRESETS = {
    "simulation": simulation_email,
    "text_context": text_context_email,
    "independent_variables": independent_variables_email,
    "historical_flow_mart": historical_flow_email,
    "optimization_context_brief": optimization_context_brief_email,
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
