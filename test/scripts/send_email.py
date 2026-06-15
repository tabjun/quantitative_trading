"""Reusable UTF-8 email sender for research reports.

Usage examples:
    python test/scripts/send_email.py
    python test/scripts/send_email.py --preset text_context
    python test/scripts/send_email.py --preset independent_variables
    python test/scripts/send_email.py --preset historical_flow_mart
    python test/scripts/send_email.py --preset optimization_context_brief
    python test/scripts/send_email.py --preset forecasting_methodology_review
    python test/scripts/send_email.py --preset optimization_stabilization_stage
    python test/scripts/send_email.py --preset breadth_expansion_interpretation
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


def forecasting_methodology_review_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_path = "test/research_materials/forecasting_methodology_literature_review_20260613.md"
    report_3_path = "test/results/3_time_series_multi_ticker_test_report_20260526_020000.md"
    report_5_path = "test/results/5_optimization_diagnostics_quick_probe_20260613.md"
    plan_6_path = "test/experiment_specs/6_optimization_stabilization_plan_20260613.md"
    readme_path = "test/README.md"
    body = f"""교수님 안녕하세요.

현재 진행 중인 업비트/주식 시계열 예측 연구의 논문화 방향과, 왜 지금 최적화 안정화 문제를 먼저 정리해야 하는지를 문헌 리뷰 형태로 정리해 공유드립니다.

이번 정리의 핵심은 최종 목표가 단순히 특정 모델 하나로 가격을 맞히는 것이 아니라, 비정상 고빈도 금융 시계열에서 예측 모델이 실제 신호를 학습했는지 검증하고, 이후 텍스트 독립변수와 historical flow data mart 같은 외생변수를 붙였을 때 개선 원인을 설명할 수 있는 연구 구조를 만드는 것입니다.

문헌을 보면 Autoformer, FEDformer, PatchTST, TimesNet, iTransformer, TimeXer, ModernTCN, Mamba/MambaTS, Chronos/TimesFM/Moirai, TFT, RevIN처럼 최근 시계열 예측 알고리즘은 매우 다양하게 연구되고 있습니다. 다만 금융·코인 예측 논문은 모델 구조만 제시하지 않고, target 정의, normalization, leakage-safe split, validation-only tuning, loss function, baseline, 거래비용, MDD, sensitivity analysis를 함께 둡니다.

따라서 현재 4번, 5번, 6번 실험은 연구 목적에서 벗어난 우회 작업이 아니라 논문으로 넘어가기 위한 필수 기반 작업입니다.

- 3번에서는 Autoformer가 수치상 좋아 보이는 현상이 있었지만, 가격 레벨 직접 회귀와 MinMax 스케일 때문에 직전가 복사 또는 평균 수렴형 착시였을 가능성이 큽니다.
- 4번은 텍스트 독립변수와 외생변수 후보가 실제 정보 가치를 갖는지 준비하는 단계입니다.
- 5번은 Linear, LSTM, GRU, TCN, Transformer 대표 구조에서 target/loss 조합이 쉬운 해로 붕괴하는지 진단한 단계입니다.
- 6번은 독립변수 확장 전에 return target, normalization, loss, collapse-aware model selection을 안정화하는 단계입니다.

보고서에서 특히 보시면 좋은 부분은 다음과 같습니다.

- `3번 Autoformer 결과를 어떻게 이해해야 하는가`: 좋아 보였던 결과가 왜 실제 예측력이라기보다 평가 착시일 수 있는지 정리했습니다.
- `최근 시계열 예측 알고리즘 흐름`: Autoformer/PatchTST뿐 아니라 TimeXer, iTransformer, ModernTCN, MambaTS, foundation model까지 다음 후보군을 비교했습니다.
- `논문별 상세 정리`: 각 논문이 무엇을 했고, 학습/최적화 문제를 어디에 넣었고, 결과를 어떤 지표와 그림으로 출력했는지 정리했습니다.
- `출판되는 논문들은 최적화 문제를 어디에 넣는가`: 최적화가 논문 제목에 드러나지 않더라도 방법론의 data/preprocessing, experimental setup, model training, loss, evaluation, ablation에 포함된다는 점을 정리했습니다.
- `현재 연구에 바로 반영해야 할 기준`: 6번 이후 어떤 알고리즘을 어떤 순서로 확장해야 하는지 정리했습니다.

이번 보고서의 결론은, 이 논문들처럼 예측 연구로 통과하려면 단순히 모델 종류를 늘리는 것보다 먼저 학습 붕괴와 평가 착시를 충분히 해소해야 한다는 것입니다. 그래야 이후 텍스트 독립변수, historical flow data mart, 온체인/유동성 변수 등을 붙였을 때 성능 개선이 실제 변수 효과인지 논문에서 설득력 있게 주장할 수 있습니다.

커밋 링크:
{commit_url}

문헌 리뷰 보고서:
{github_blob(report_path)}

관련 연구 흐름:
- 3번 다종목 실험 보고서:
{github_blob(report_3_path)}

- 5번 최적화 진단 보고서:
{github_blob(report_5_path)}

- 6번 최적화 안정화 계획서:
{github_blob(plan_6_path)}

- test 연구 공간 안내:
{github_blob(readme_path)}

자세한 알고리즘별 논문 요약과 다음 실험 후보는 문헌 리뷰 보고서를 보시면 되고, 현재 로컬 노트북 결과 기준의 collapse 진단은 5번 보고서, 이후 서버에서 수행할 안정화 단계는 6번 계획서에 정리해 두었습니다.

감사합니다.
"""
    subject = "[시계열 연구] 예측 방법론 문헌 리뷰 및 최적화 안정화 필요성"
    return subject, body, []


def optimization_stabilization_stage_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_5_path = "test/results/5_optimization_diagnostics_quick_probe_20260613.md"
    report_6_path = "test/results/6_optimization_stabilization_stage_report_20260613.md"
    plan_6_path = "test/experiment_specs/6_optimization_stabilization_plan_20260613.md"
    plan_7_path = "test/experiment_specs/7_optimization_breadth_expansion_plan_20260613.md"
    notebook_6_path = "test/models/6_optimization_stabilization_test.ipynb"
    notebook_7_path = "test/models/7_optimization_breadth_expansion_test.ipynb"
    body = f"""교수님 안녕하세요.

5번 최적화 진단 이후, 후속 연구를 어떤 순서로 진행해야 하는지 정리한 6번 단계 보고와 7번 확장 실험 계획을 함께 공유드립니다.

이번에 6번을 따로 만든 이유
- 5번에서는 `Linear`, `LSTM`, `GRU`, `TCN`, `Transformer` 5개 대표 구조를 기준으로, 가격 레벨 직접 회귀와 수익률 기반 목표가 어떤 식으로 무너지는지 약식 진단했습니다.
- 그 결과 train loss는 줄어도 예측이 직전가 복사나 0 수익률 근처의 쉬운 해로 붕괴하는 문제가 확인됐습니다.
- 이 상태에서 바로 텍스트 독립변수나 데이터마트를 더 붙이면, 개선/악화가 변수 때문인지 최적화 문제 때문인지 해석하기 어려워집니다.

6번의 실제 역할
- 6번은 대규모 확장 실험 완료본이 아니라, 5번 결과를 바탕으로 후속 안정화 실험을 어떤 순서로 실행할지 고정하는 오케스트레이터입니다.
- 현재 저장된 6번 노트북 출력도 성능표 묶음이 아니라 stage plan 출력입니다.
- 즉 6번의 핵심 결과는 “무엇을 먼저 확인하고, 어떤 그림이면 통과이며, 어떤 경우 재설계해야 하는가”라는 연구 의사결정 기준입니다.

이번 단계에서 다시 확인한 핵심 방법론
- `level_mse`는 다음 가격 수준을 직접 맞히는 방식이고, 자기상관이 강한 가격 데이터에서는 직전가 복사형 답으로 빠지기 쉽습니다.
- `return_huber`는 다음 수익률을 보다 안정적으로 학습하려는 손실입니다.
- `return_directional_hybrid`는 값 오차뿐 아니라 방향성도 같이 보려는 손실입니다.
- 성능 판단은 단순 loss 감소가 아니라 `KRW MAE/RMSE`, `DA`, `MASE`, `persistence_gap`, `collapse_score`, `near_zero_return_share`, `sign_agreement`를 함께 봐야 합니다.

현재 결과와 해석
- 6번만으로는 “어떤 모델이 최종 우승이다”라고 결론낼 수 없습니다.
- 대신 5번에서 드러난 붕괴 문제를 기준으로, 6번에서 `target -> normalization -> loss -> resource scale -> independent variable gate` 순서로 확인해야 한다는 점을 고정했습니다.
- 이 결과는 연구 흐름상 중요하지만, 실제로 더 넓은 모델 변형과 앙상블을 비교한 결과까지 포함하지는 않습니다.

그래서 7번이 필요한 이유
- 사용 중인 기본 5개 구조만으로는 후속 방향을 충분히 고르기 어렵습니다.
- 7번에서는 `Autoformer-like`, `PatchTST-like`, `DLinear/NLinear-like`, `TimesNet/TimeXer-like`, `iTransformer-like`, `ModernTCN-like`, `Mamba-like` 같은 확장 단일 모델군과, `LSTM + Autoformer`, `TCN + Transformer`, `validation weighted top-k ensemble` 같은 앙상블군까지 포함해 더 넓은 폭으로 확인할 계획입니다.
- 즉 6번은 “순서를 정하는 단계”, 7번은 “더 넓게 실제 비교해 방향을 고르는 단계”로 역할을 분리했습니다.

커밋 링크:
{commit_url}

보고서 및 계획 링크:
- 5번 진단 보고서:
{github_blob(report_5_path)}

- 6번 안정화 단계 보고서:
{github_blob(report_6_path)}

- 6번 안정화 실행 계획서:
{github_blob(plan_6_path)}

- 7번 넓은 폭 후속 확장 실험 계획서:
{github_blob(plan_7_path)}

관련 노트북:
- 6번 노트북:
{github_blob(notebook_6_path)}

- 7번 노트북:
{github_blob(notebook_7_path)}

보고서를 보실 때,
- 5번 보고서에서는 실제 붕괴 양상과 대표 5개 구조의 quick probe 해석을,
- 6번 보고서에서는 왜 6번이 결과 보고서이면서도 동시에 연결 문서인지,
- 7번 계획서에서는 다음 단계에서 어떤 모델군과 앙상블군을 어떤 기준으로 넓게 볼지를 확인하실 수 있습니다.

이번 연구의 핵심 목표는 결국 업비트 비정상 시계열에서 모델이 쉬운 해로 무너지지 않도록 학습 구조를 먼저 안정화한 뒤, 그 위에 독립변수와 데이터마트를 결합해 논문 형태의 예측 연구로 확장하는 것입니다.

감사합니다.
"""
    subject = "[시계열 연구] 6번 안정화 단계 보고 및 7번 후속 확장 계획"
    return subject, body, []


def breadth_expansion_interpretation_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_7_path = "test/results/7_optimization_breadth_expansion_interpretation_20260616.md"
    report_8_path = "test/experiment_specs/8_optimization_breadth_training_plan_20260616.md"
    body = f"""교수님 안녕하세요.

7번 확장 실험 결과를 정리해 공유드립니다. 이번 7번은 실제 학습 결과가 아니라, 6번 안정화 이후 어떤 breadth expansion을 어떤 자원 프로필과 단계로 돌려야 하는지 정리한 stage plan 출력이었습니다.

무엇을 확인했는지
- 학교 서버 환경에서 `school_4090_15gb` 자원 프로필을 기준으로 CPU/RAM/GPU/CUDA/PyTorch 상태를 감지했습니다.
- `breadth_probe`, `ensemble_probe`, `normalization_cross_check`, `loss_cross_check`, `scale_confirmation` 같은 후속 실험 순서를 정리했습니다.
- 다만 7번 코드 자체에는 아직 확장 모델군 전체를 실제로 학습하는 backend가 없어서, 학습 곡선/예측 그래프/collapse 진단을 보여주는 결과는 아니었습니다.

왜 이 해석이 중요한가
- 7번을 성능 결과로 읽으면 안 되고, 다음 실험 설계와 자원 관리 기준을 고정한 중간 산출물로 읽어야 합니다.
- 실제로 필요한 것은 알고리즘 이름을 늘리는 것뿐 아니라 preprocessing, normalization, loss, optimizer, scheduler, gradient policy, ensemble 조합까지 넓히는 것입니다.
- 그래서 8번은 새 번호로 분리해 실제 GPU 학습 실험으로 작성했습니다.

현재 결론
- 7번은 확장 학습의 결과가 아니라 확장 학습을 어떻게 안전하게 돌릴지 정리한 계획 문서입니다.
- 따라서 7번만으로는 모델 우열을 말할 수 없고, 실제 판단은 8번 breadth training에서 해야 합니다.
- 8번에서는 같은 5개 기본 모델뿐 아니라 전처리/정규화/손실/최적화/기울기 안정화/앙상블 조합까지 함께 비교하도록 바꾸었습니다.

커밋 링크:
{commit_url}

7번 해석 보고서:
{github_blob(report_7_path)}

8번 계획서:
{github_blob(report_8_path)}

조금 더 자세히 보실 때는 7번 해석 보고서에서 “왜 7번이 stage plan인지”와 “왜 8번이 필요한지”를 보시면 되고, 8번 계획서에서는 앞으로 어떤 축으로 실제 학습을 돌릴지 보실 수 있습니다.

감사합니다.
"""
    subject = "[시계열 연구] 7번 확장 실험 해석 및 8번 breadth training 계획"
    return subject, body, []


PRESETS = {
    "simulation": simulation_email,
    "text_context": text_context_email,
    "independent_variables": independent_variables_email,
    "historical_flow_mart": historical_flow_email,
    "optimization_context_brief": optimization_context_brief_email,
    "forecasting_methodology_review": forecasting_methodology_review_email,
    "optimization_stabilization_stage": optimization_stabilization_stage_email,
    "breadth_expansion_interpretation": breadth_expansion_interpretation_email,
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
