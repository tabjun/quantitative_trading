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
    report_path = "test/results/5_optimization_diagnostics_quick_probe_20260613.md"
    design_path = "test/experiment_specs/5_optimization_diagnostics_framework_design_20260613.md"
    next_path = "test/experiment_specs/6_optimization_stabilization_plan_20260613.md"
    body = f"""교수님 안녕하세요.

교수님 ~~ 에 대한 것 보냅니다. 분석 해석 및  보고서 양식에도 시간 많이 드는데, 서버에서 커널을 빌려오다보니 제약사항에 대해 해결하느라 시간이 참많이드네요. 이것때문에 차라리 컴퓨터를 하나 맞추서 분석하고 싶어서 견적내보다가 가격이 제 월급나온거보고 접고 다시 있는 환경에서 최선을 다하기로했습니다. ㅋㅋㅋㅋ

이번에는 업비트 비정상 시계열에서 모델 학습 손실이 줄어도 예측이 실제로 좋아지는지, 아니면 직전가 복사나 0 수익률 근처의 쉬운 해로 무너지는지를 확인한 5번 최적화 진단 보고서를 정리했습니다.

무엇을 수행했는지
- `Linear`, `LSTM`, `GRU`, `TCN`, `Transformer` 5개 대표 구조를 비교했습니다.
- 각 구조에 대해 `level_mse`, `return_huber`, `return_directional_hybrid`를 비교했습니다.
- 단순 loss만 보지 않고, train/validation curve, 일반화 간격, gradient, persistence gap, collapse score, near-zero return share, sign agreement를 함께 확인했습니다.
- 노트북 출력 이미지를 `test/images`로 추출해 보고서에서 그래프 해석과 함께 읽을 수 있도록 정리했습니다.

왜 했는지
- 이후 텍스트 독립변수, historical flow mart, 온체인/유동성 변수 등을 붙이려면 먼저 모델이 기본 시계열에서도 쉬운 해로 무너지지 않는지 확인해야 합니다.
- 최적화 문제가 해결되지 않은 상태에서 독립변수를 늘리면, 성능 변화가 변수 때문인지 손실함수/target 문제 때문인지 해석하기 어렵습니다.
- 그래서 5번은 최종 예측 모델을 고르는 단계가 아니라, 다음 연구 확장 전에 학습 붕괴를 진단하는 단계로 두었습니다.

전체 결과
- `level_mse`는 5개 구조 전체에서 사실상 실패했습니다. 가격 레벨 직접 회귀는 자기상관이 강한 가격 데이터를 다루면서 복사형 함정에 빠지기 쉬웠습니다.
- `return_huber`는 손실은 잘 줄였지만, 여러 케이스에서 0 수익률 근처나 축소된 분산 예측으로 수렴하는 위험이 남았습니다.
- `directional_hybrid`는 방향성 제약을 통해 shortcut을 줄이려는 시도였고, 그중 `tcn_return_directional_hybrid`가 가장 덜 무너졌습니다.
- 다만 이 후보도 개선 폭이 작고 near-zero 예측 성향이 남아 있어 최종 채택이 아니라 다음 안정화 실험의 출발점으로 보는 것이 맞습니다.

시사점
- 지금 단계에서는 독립변수를 더 붙이는 것보다 target, normalization, loss, model selection 기준을 먼저 안정화해야 합니다.
- 단순히 train loss가 내려가는지보다, 검증선도 같이 내려오는지와 단순 직전가 복사 기준선을 이기는지가 더 중요합니다.
- 이번 결과는 기울기 소실이라기보다, 목적함수가 너무 쉬운 답을 허용해서 모델이 빠르게 shortcut을 찾는 문제에 가깝습니다.

한계
- 현재 결과는 사용자가 실행한 로컬 노트북 출력 기준의 quick probe입니다.
- 서버에서 같은 seed/split/scaler로 재현되는지 한 번 더 확인해야 합니다.
- 아직 텍스트 독립변수나 historical flow mart는 붙이지 않았습니다. 일부러 붙이지 않은 이유는 최적화 문제가 먼저 정리되어야 이후 독립변수 효과를 해석할 수 있기 때문입니다.

다음 수행 계획
- 6번 실험에서는 바로 더 큰 모델로 가지 않고, 5번 결과를 기준으로 하나씩 수정합니다.
- 1단계는 5번 결과 재현 확인, 2단계는 level target 제거와 return target 고정, 3단계는 normalization ablation, 4단계는 loss ablation, 5단계는 collapse-aware model selection입니다.
- 이 단계를 통과한 뒤에 텍스트 독립변수, historical flow mart, 온체인/유동성 변수 결합으로 넘어가겠습니다.

커밋 링크:
{commit_url}

보고서 및 코드 링크:
- 최종 5번 보고서:
{github_blob(report_path)}

- 5번 실험 설계서:
{github_blob(design_path)}

- 6번 후속 안정화 계획서:
{github_blob(next_path)}

- ipynb: {github_blob('test/models/5_optimization_diagnostics_test.ipynb')}
- py mirror: {github_blob('test/models/5_optimization_diagnostics_test.py')}
- 6번 다음 단계 ipynb: {github_blob('test/models/6_optimization_stabilization_test.ipynb')}
- 6번 다음 단계 py mirror: {github_blob('test/models/6_optimization_stabilization_test.py')}

보고서를 자세히 보실 때는 `최종 5번 보고서`에서 그래프별 해석, 케이스별 해석, collapse 진단, 결론을 보시면 됩니다.
왜 5번과 6번을 나누었는지와 다음에 무엇을 수정하며 확인할지는 `6번 후속 안정화 계획서`에 정리해 두었습니다.
6번 코드는 실제 학습을 바로 실행하는 코드가 아니라, 서버에서 stage별로 `수행 > 결과 확인 > 수정 > 재수행`을 반복하기 위한 실행 계획 코드입니다.

감사합니다.
"""
    subject = "[시계열 연구] 5번 최적화 진단 최종 보고 및 6번 안정화 계획"
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
