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
    python test/scripts/send_email.py --preset preprocessing_matrix_results
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

7번 확장 실험 결과를 정리해 공유드립니다. 결론부터 말씀드리면, 7번은 모델 성능 결과가 아니라 6번 안정화 이후의 후속 breadth expansion을 어떤 자원 프로필과 실행 순서로 돌릴지 정리한 stage plan입니다.

이번 7번에서 실제로 나온 것
- 학교 서버/로컬 환경의 CPU, RAM, GPU, CUDA, PyTorch 상태를 감지했습니다.
- `school_4090_15gb` 자원 프로필을 적용했고, 과도한 병렬화와 장시간 단일 실행을 피하도록 실행 원칙을 고정했습니다.
- `breadth_probe`, `ensemble_probe`, `normalization_cross_check`, `loss_cross_check`, `scale_confirmation` 같은 후속 실험 suite를 정의했습니다.

왜 이 해석이 필요한가
- 7번은 학습 곡선, 예측 그래프, collapse 진단, leaderboard를 만든 실험이 아닙니다.
- 따라서 7번을 성능 순위표로 읽으면 안 되고, 다음 실험을 돌릴 때 어떤 축을 먼저 확인해야 하는지 정리한 중간 산출물로 읽어야 합니다.
- 특히 지금 연구의 핵심은 모델 이름만 늘리는 것이 아니라 preprocessing, normalization, loss, optimizer/scheduler, gradient policy, ensemble 조합까지 함께 넓혀서 무엇이 실제로 영향을 주는지 보는 것입니다.

현재 결론
- 7번은 확장 학습의 결과가 아니라 확장 학습을 안전하게 설계하는 문서입니다.
- 7번만으로는 모델 우열을 말할 수 없고, 실제 판단은 8번 breadth training에서 해야 합니다.
- 그래서 8번은 새 번호로 분리해 실제 GPU 학습 실험으로 다시 작성했습니다.

교수님께서 보실 때는
- 7번 보고서는 “왜 이 뒤에 8번이 필요한가”를 설명하는 해석 문서로 보시면 됩니다.
- 8번 계획서는 “앞으로 어떤 축을 실제 학습으로 비교할 것인가”를 보여줍니다.
- 이 두 문서를 합쳐 보시면, 현재 연구가 단순 모델 비교가 아니라 학습 붕괴를 먼저 해소한 뒤 외생변수와 데이터마트를 붙이려는 흐름이라는 점을 확인하실 수 있습니다.

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


def preprocessing_uncertainty_diagnostics_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_8_path = "test/results/8_optimization_breadth_training_report_20260623.md"
    plan_9_path = "test/experiment_specs/9_preprocessing_uncertainty_diagnostics_plan_20260623.md"
    notebook_9_path = "test/models/9_preprocessing_uncertainty_diagnostics_test.ipynb"
    video_black_scholes = "https://www.youtube.com/watch?v=99BHnu64pu8"
    video_double_descent = "https://www.youtube.com/watch?v=5ruiphjlOwo"
    body = f"""교수님 안녕하세요.

업비트 비정상 시계열의 최적화 문제를 확인하기 위해 수행한 8번 breadth training 결과와, 이를 반영해 설계한 9번 전처리·불확실성 진단 실험을 공유드립니다.

앞서 보낸 메일에서 `persistence`, `0수익률 평탄화`, `예측 분산`, `방향성`을 용어 중심으로 너무 압축해 설명한 부분이 있어, 이번 메일에서는 각 용어의 뜻과 숫자 예시, 이번 실험에서의 의미, 좋은 신호와 나쁜 신호를 구분해 보충드립니다.

1. 왜 8번을 수행했는가

앞선 5번과 6번에서는 loss가 감소하더라도 예측이 직전 가격 복사 또는 0수익률 근처로 붕괴할 수 있다는 문제를 확인했습니다. 7번은 실제 학습 결과가 아니라 서버 자원과 실행 단계를 정리한 계획이었기 때문에, 8번에서는 Linear, LSTM, GRU, TCN, Transformer와 Autoformer-like, PatchTST-like, DLinear/NLinear-like, TimesNet-like, TimeXer-like, iTransformer-like, ModernTCN-like, Mamba-like를 같은 조건에서 실제 학습했습니다.

2. 8번 결과와 핵심 용어 설명

먼저 persistence baseline이 무엇인지 설명드리겠습니다.

`Persistence`는 “다음 가격도 지금 가격과 같을 것”이라고 예측하는 가장 단순한 기준선입니다. 예를 들어 현재 비트코인 가격이 1억 원이면, 별도의 모델 없이 다음 15분 가격도 1억 원이라고 예측합니다. 수익률로 표현하면 다음 수익률을 항상 0%라고 예측하는 것과 같습니다.

금융 가격은 직전 가격과 매우 비슷한 경우가 많기 때문에 이 단순한 방법도 의외로 강합니다. 그래서 복잡한 모델은 최소한 persistence보다 가격 오차가 작아야 “직전 가격을 그대로 복사하는 것보다 새로운 정보를 학습했다”고 말할 수 있습니다. 이번 실험에서 persistence의 평균절대오차(MAE)는 약 190,608원이었습니다. MAE는 각 시점에서 예측 가격과 실제 가격이 평균적으로 얼마만큼 떨어져 있었는지를 원화로 나타낸 값이며, 작을수록 좋습니다.

이번 14개 모델은 모두 persistence의 MAE를 이기지 못했습니다. 즉 모델을 복잡하게 만들었지만, 평균 가격 오차만 놓고 보면 “다음 가격은 현재 가격과 같다”는 단순 예측보다 정확하지 않았습니다.

다음으로 “0수익률 근처로 평평해졌다”는 표현의 의미를 설명드리겠습니다.

실제 다음 수익률이 순서대로 `+1.0%, -0.8%, +0.6%, -1.2%`처럼 오르내렸다고 가정하겠습니다. 평탄화된 모델은 이를 `+0.02%, -0.01%, 0.00%, -0.02%`처럼 거의 0에 가까운 값으로만 예측합니다. 그래프에서는 실제 수익률 선은 위아래로 움직이는데 예측선은 0을 따라 거의 직선처럼 보입니다.

이런 예측은 급등과 급락의 크기를 맞히지 못했지만, 대부분의 15분 수익률이 작은 값이라는 특성을 이용해 평균 오차를 무난하게 줄일 수 있습니다. 따라서 loss나 MAE가 persistence에 가까워 보여도 실제 시장 움직임을 배운 결과라고 볼 수 없습니다. 이번에 “예측 분산이 실제의 1% 미만”이라는 말은 예측 오차가 1%라는 뜻이 아닙니다. 예측 수익률이 위아래로 퍼진 정도, 즉 움직임의 폭이 실제 수익률 변동 폭의 1%도 되지 않았다는 뜻입니다.

TimesNetLike, NLinearLike, DLinearLike, TCN이 여기에 해당했습니다. 이 모델들은 MAE가 persistence에 비교적 가까웠지만, 예측 분산이 실제의 1% 미만으로 줄었습니다. 따라서 상대적으로 낮은 MAE는 실제 상승·하락 패턴을 잘 맞혀서 얻은 성과라기보다, 대부분의 예측을 0수익률 근처에 놓아 큰 판단을 피한 결과로 해석하는 것이 맞습니다.

“방향성과 예측 분산을 보존했다”는 표현도 좋은 결과와 같은 뜻은 아닙니다.

`방향성`은 다음 수익률이 양수인지 음수인지, 즉 상승과 하락의 부호를 얼마나 맞혔는지를 뜻합니다. 실제가 상승일 때 상승, 하락일 때 하락으로 예측하면 방향을 맞힌 것입니다. 방향 정확도가 50% 정도라면 대체로 동전 던지기 수준이고, 여러 구간과 seed에서 지속적으로 50%를 넘어야 유용한 신호 가능성을 검토할 수 있습니다.

`예측 분산`은 예측값이 평균 주변에서 얼마나 넓게 움직이는지를 나타냅니다. 실제 시장이 크게 움직이는데 예측 분산이 거의 0이면 평탄화이고, 반대로 실제보다 지나치게 크면 폭주입니다. 따라서 예측 분산을 보존한다는 것은 실제처럼 어느 정도 위아래로 움직였다는 뜻이며, 모델이 무조건 0만 출력하지 않았다는 점에서는 좋은 신호입니다. 다만 움직임의 방향과 크기가 실제와 맞아야 하므로, 분산 보존만으로 정확한 모델이라고 할 수는 없습니다.

Linear와 PatchTSTLike는 평탄화된 모델보다 상승·하락 방향과 움직임의 폭을 더 많이 남겼습니다. 이는 “실제 신호를 학습할 후보로 추가 점검할 가치가 있다”는 점에서는 긍정적입니다. 그러나 두 모델의 원화 MAE는 persistence보다 더 컸습니다. 쉽게 말하면 가만히 있지는 않고 시장처럼 움직이려고 했지만, 움직인 방향이나 크기가 충분히 정확하지 않아 가격 예측 오차는 오히려 커졌습니다. 따라서 이 결과는 최종적으로 좋은 모델이라는 결론이 아니라, 평탄화 문제는 덜하지만 정확도 문제가 남은 별도의 실패 유형으로 해석해야 합니다.

반대쪽 실패도 있었습니다. LSTM과 GRU의 variance ratio는 각각 약 2.28, 4.39였습니다. `Variance ratio`는 예측 수익률 분산을 실제 수익률 분산으로 나눈 값입니다. 1이면 움직임의 폭이 실제와 비슷하고, 1보다 매우 작으면 평탄화, 1보다 매우 크면 실제보다 과도하게 흔들리는 폭주를 뜻합니다. 따라서 LSTM은 실제보다 약 2.28배, GRU는 약 4.39배 큰 분산을 만들었습니다.

AutoformerLike는 MAE 약 163만 원, variance ratio 약 59.39로 가장 심하게 폭주했습니다. 이는 실제 수익률의 변동 폭보다 예측이 지나치게 크게 움직였다는 뜻입니다. 과거 3번 실험에서 좋게 보였던 Autoformer 결과를 구조 자체의 우위로 해석할 수 없다는 점이 다시 확인됐습니다.

정리하면 이번 결과에는 두 종류의 실패가 동시에 있었습니다.

- 평탄화 실패: 예측을 거의 0수익률로 만들어 persistence와 비슷해 보이지만 실제 변동을 놓칩니다.
- 분산 폭주 실패: 예측은 활발하게 움직이지만 실제보다 너무 크게 움직여 가격 오차가 커집니다.

좋은 모델은 이 두 극단 사이에서 단순히 움직임만 만드는 것이 아니라, persistence보다 낮은 원화 MAE, 실제에 가까운 분산, 우연 수준을 넘는 방향 정확도를 여러 seed와 시간 구간에서 동시에 보여야 합니다.

따라서 현재 단계에서 최신 모델을 더 추가하는 것만으로는 문제가 해결되지 않습니다. 전처리, 분포 이동, 변동성, 불확실성, 모델 용량과 seed 변동을 분리해 확인해야 합니다.

3. 9번에서 무엇을 추가했는가

9번은 8번을 덮어쓰지 않고 새 번호로 분리했습니다.

- 극단값 처리: winsorization 강도 3종, Hampel filter
- heavy-tail 처리: robust asinh, signed log, winsor+asinh, Hampel+asinh
- 추세 처리: first/seasonal difference, EMA residual, linear detrend, median residual
- 주파수 처리: high-pass 강도 3종, band-pass, winsor+frequency 조합
- 변동성 처리: local volatility scaling과 asinh 결합
- 총 28개 preprocessing pipeline
- Linear, PatchTSTLike, TimesNetLike, AutoformerLike 비교
- seed ensemble과 conformal prediction interval
- interval coverage, width, miss distance
- hidden width와 parameter count 변화에 따른 generalization error
- 원시 데이터부터 전처리 전후, 학습곡선, gradient, 예측, 불확실성, heatmap까지 notebook inline 시각화

4. 영상 내용을 어떻게 반영했는가

Black–Scholes 관련 영상:
{video_black_scholes}

이 영상에서 참고한 핵심은 미래 가격을 하나의 값으로 단정하지 않고 확률분포와 변동성으로 표현하는 관점입니다. Black–Scholes의 로그정규 가정을 코인에 그대로 적용하지는 않고, seed ensemble과 conformal interval을 이용해 예측구간의 실제 coverage와 폭을 평가하도록 반영했습니다.

Double Descent 관련 영상:
{video_double_descent}

이 영상에서 참고한 핵심은 모델이 커질수록 일반화 오차가 단순히 증가하거나 감소한다고 가정할 수 없다는 점입니다. 9번에서는 hidden width와 parameter count를 단계적으로 늘리고 seed와 epoch 변화까지 함께 기록해 interpolation threshold 부근의 불안정성을 확인하도록 구성했습니다.

영상은 문제를 이해하기 위한 참고 자료로 사용했고, 실제 설계 근거는 RevIN, Dish-TS, FAN, FredNormer, NoRIN, Double Descent, conformal time-series forecasting 관련 원 논문을 사용했습니다.

5. 현재 시사점

이번 연구의 핵심은 점예측 성능 하나를 높이는 것이 아니라, 모델이 어떤 방식으로 틀리는지를 분리하는 것입니다. MAE가 낮아도 0수익률 collapse일 수 있고, 방향 정확도가 높아도 가격 오차가 클 수 있으며, prediction interval coverage가 높아도 구간이 지나치게 넓으면 실용적이지 않을 수 있습니다.

따라서 9번에서는 다음 조건을 동시에 만족하는 조합만 후속 독립변수·데이터마트 결합 단계로 넘길 예정입니다.

- persistence보다 낮은 MAE
- 예측 분산 보존
- 0수익률 쏠림 감소
- seed 간 안정성
- 적절한 interval coverage와 width
- 모델 용량 변화에 대한 일반화 안정성

커밋 링크:
{commit_url}

8번 결과 보고서:
{github_blob(report_8_path)}

9번 실험 설계서:
{github_blob(plan_9_path)}

9번 실행 노트북:
{github_blob(notebook_9_path)}

8번 보고서에서는 각 모델이 persistence에 미달한 이유와 평탄화/폭주 collapse의 차이를 보실 수 있습니다. 9번 설계서에서는 두 영상과 관련 논문을 전처리·불확실성·모델 용량 실험으로 어떻게 변환했는지 확인하실 수 있습니다.

감사합니다.
"""
    subject = "[보충 설명] 8번 breadth 결과 용어 해설 및 9번 후속 진단"
    return subject, body, []


def preprocessing_matrix_results_email(commit_hash: str) -> tuple[str, str, list[Path]]:
    commit_url = f"{GITHUB_BASE}/commit/{commit_hash}"
    report_path = "test/results/9_preprocessing_uncertainty_diagnostics_report_20260623.md"
    notebook_path = "test/models/9_preprocessing_uncertainty_diagnostics_test.ipynb"
    plan_10_path = "test/experiment_specs/10_objective_ensemble_confirmation_plan_20260623.md"
    notebook_10_path = "test/models/10_objective_ensemble_confirmation_test.ipynb"
    summary_image = "test/images/9_preprocessing_uncertainty_diagnostics_test_cell002_230.png"
    heatmap_image = "test/images/9_preprocessing_uncertainty_diagnostics_test_cell002_231.png"
    best_image = "test/images/9_preprocessing_uncertainty_diagnostics_test_cell002_81.png"
    direction_image = "test/images/9_preprocessing_uncertainty_diagnostics_test_cell002_33.png"
    flat_image = "test/images/9_preprocessing_uncertainty_diagnostics_test_cell002_11.png"
    explosion_image = "test/images/9_preprocessing_uncertainty_diagnostics_test_cell002_13.png"
    body = f"""교수님 안녕하세요.

9번 전처리 진단 112개 케이스 실행이 완료되어, 결과와 다음 10번 본실험 방향을 정리해 공유드립니다.

이번 실행은 비트코인 15분봉 39,935행을 시간 순서대로 train 70%, validation 15%, test 15%로 나누고, Linear·PatchTSTLike·TimesNetLike·AutoformerLike 네 모델과 28개 전처리를 조합한 실험입니다. 목표는 다음 15분 가격 자체가 아니라 다음 로그수익률이며, 예측 수익률을 다시 원화 가격으로 복원해 평가했습니다.

먼저 기준선인 persistence는 “다음 가격도 현재 가격과 같다”고 예측하는 방법입니다. 현재 가격이 1억 원이면 다음 15분 가격도 1억 원이라고 예측하며, 수익률로는 항상 0%를 예측하는 것과 같습니다. 이번 persistence 평균절대오차는 약 190,608원이었습니다. 복잡한 모델은 이 값보다 낮아야 직전 가격 복사 이상의 정보를 학습했다고 말할 수 있습니다.

9번의 결론은 전처리만으로 persistence를 이긴 조합은 없었다는 것입니다.

- 최저 MAE는 `PatchTSTLike + seasonal_diff16`의 약 227,412원이었습니다. persistence보다 약 19.3% 큰 오차입니다.
- `PatchTSTLike + winsor_025`는 방향 정확도 약 55.45%로 상대적으로 높았지만, MAE는 약 240,822원으로 persistence보다 약 26.3% 컸습니다.
- 따라서 최종 우승 모델은 없지만, 다음 objective 실험에 넘길 후보와 버려야 할 실패 형태는 분명해졌습니다.

1. 전체 결과를 보는 그래프

전체 요약:
{github_blob(summary_image)}

왼쪽 막대의 x축은 모델·전처리 조합, y축은 모델 MAE를 persistence MAE로 나눈 copy-risk ratio입니다. 검은 선 1 아래로 내려가야 persistence를 이기지만 모든 막대가 1보다 위에 있습니다.

가운데 산점도의 x축은 예측 분산을 실제 분산으로 나눈 variance ratio, y축은 상승·하락 방향 정확도입니다. x축 1 근처이면서 y축 0.5보다 안정적으로 위에 있어야 합니다. x축 0.01 미만의 점들은 예측이 0수익률에 가깝게 평평해진 경우입니다.

전처리·모델 heatmap:
{github_blob(heatmap_image)}

y축은 전처리, x축은 모델, 색은 `모델 MAE / persistence MAE`입니다. AutoformerLike는 대부분 전처리에서 큰 오차가 유지됐고, TimesNetLike는 1에 가까워 보이지만 수익률 움직임을 포기해 persistence와 비슷해진 경우였습니다. PatchTSTLike와 Linear는 전처리에 따라 결과가 달라져 다음 실험 후보로 남겼습니다.

2. 다음 방향을 잡는 데 유용한 후보

최저 MAE 후보:
{github_blob(best_image)}

이 그림은 `PatchTSTLike + seasonal_diff16`의 test 결과입니다. 위쪽 x축은 test 시점 순서, y축은 다음 로그수익률이며 파란 선이 실제, 주황 선이 예측입니다. 예측은 0에 붙어 있지는 않지만 실제 급등락을 축소했습니다. 중간 패널의 y축은 원화 종가이며, 주황 예측선이 초록 persistence에서 벗어나기는 하지만 실제 파란 선보다 일관되게 정확하지 않았습니다.

방향성 후보:
{github_blob(direction_image)}

`PatchTSTLike + winsor_025`는 예측선이 실제처럼 양수와 음수를 자주 오가며 방향 정확도 약 55.45%를 기록했습니다. 다만 큰 하락 폭을 충분히 맞히지 못해 원화 MAE가 더 컸습니다. 즉 방향 정보는 일부 남겼지만 가격 변화의 크기를 정확히 맞히지는 못했습니다.

3. 좋지 않은 결과로 확인한 두 실패 유형

0수익률 평탄화:
{github_blob(flat_image)}

`TimesNetLike + none`은 위쪽 수익률 그래프에서 주황 예측선이 거의 0에 붙어 있습니다. 중간 원화 가격 그래프에서는 예측과 persistence가 거의 겹쳐 실제 가격에도 가까워 보이지만, 이는 새로운 신호를 학습한 것이 아니라 다음 변화가 없다고 예측한 결과입니다.

출력 분산 폭주:
{github_blob(explosion_image)}

`AutoformerLike + none`은 실제 15분 수익률보다 훨씬 큰 약 -7%~+6%의 가짜 장기 파동을 예측했습니다. 원화 복원 가격도 실제와 persistence에서 크게 이탈했습니다. 학습 loss와 gradient는 감소했으므로 단순 기울기 소실 문제가 아니라, 현재 손실함수가 지나치게 큰 출력 변동을 충분히 막지 못한 문제로 해석했습니다.

4. 다음 10번 본실험

9번 결과를 토대로 10번은 전처리를 더 늘리지 않고 다음 축을 확인하도록 수정했습니다.

- 주 실험 모델: Linear, PatchTSTLike
- 실패 통제군: 평탄화 TimesNetLike, 폭주 AutoformerLike
- 입력 후보: seasonal_diff16, frequency_bandpass, median_residual_5, linear_detrend+asinh_robust, winsor_025, none
- objective: Huber, 방향, 분산, 상관, tail, 변동성 regime, anti-collapse, balanced composite
- 보조 손실이 Huber를 압도하지 않도록 크기를 Huber 기준으로 정규화하고 첫 3 epoch 동안 점진적으로 적용
- seed 42·137·2026 재현성 확인
- test 결과를 선택에 사용하지 않는 validation-only ensemble
- calibration scatter의 축 공유 오류 수정

10번에서는 persistence보다 낮은 MAE, 실제에 가까운 분산, 우연 수준을 안정적으로 넘는 방향 정확도를 여러 seed에서 동시에 만족하는 조합만 통과시키겠습니다.

커밋:
{commit_url}

9번 최종 보고서:
{github_blob(report_path)}

9번 실행 노트북:
{github_blob(notebook_path)}

10번 실험 계획서:
{github_blob(plan_10_path)}

10번 실행 노트북:
{github_blob(notebook_10_path)}

감사합니다.
"""
    subject = "[시계열 연구] 9번 전처리 진단 최종 결과 및 10번 본실험 계획"
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
    "preprocessing_uncertainty_diagnostics": preprocessing_uncertainty_diagnostics_email,
    "preprocessing_matrix_results": preprocessing_matrix_results_email,
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
