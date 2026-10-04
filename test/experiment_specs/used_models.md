# 사용 모델 현황

이 문서는 번호를 붙이지 않는다. 특정 실험 하나의 계획서가 아니라, **우리 파이프라인에서
각 모델이 지금 어떤 상태인가**(기존 유지/공식 구현 교체/신규 도입/미채택)를 추적하는
평시 트래커다. 모델이 바뀔 때마다 새 `{번호}_*.md` 계획서를 만드는 대신, 이 표의 행을
갱신하고 **비고 칸에 적용 연구 번호만** 남긴다. 모델 자체의 아키텍처·논문 근거는
`test/research_materials/model_catalog.md`를 본다 — 겹치는 설명을 여기 다시 쓰지 않는다.
어느 연구가 실제로 무엇을 했는지(실행 결과)는 `test/results/{번호}_*/`에 남는다 — 여기는
"지금 쓸 수 있는 상태인가"만 추적한다.

## 전체 모델 상태

| 모델 | 상태 | 실행 환경 | 적용 연구(비고) |
| :--- | :--- | :--- | :--- |
| naive | 기존 유지 | 메인 venv | 24, 25 |
| HAR-RV | 기존 유지 | 메인 venv | 24, 25 |
| Linear | 기존 유지 | 메인 venv | 24, 25 |
| Ridge | 기존 유지 | 메인 venv | 24, 25 |
| KernelRidge-RBF | 기존 유지 | 메인 venv | 24, 25 |
| SVR-RBF | 기존 유지 | 메인 venv | 24, 25 |
| Nystroem+Ridge | 기존 유지 | 메인 venv | 24, 25 |
| LightGBM | 기존 유지 | 메인 venv | 24, 25 |
| XGBoost | 기존 유지 | 메인 venv | 24, 25 |
| HistGBM | 기존 유지 | 메인 venv | 24, 25 |
| GARCH-t | 기존 유지 | 메인 venv | 24, 25 |
| MS-GARCH | 기존 유지 | 메인 venv | 24, 25 |
| TAR-GARCH | 기존 유지 | 메인 venv | 24, 25 |
| GARCH+LightGBM | 기존 유지 | 메인 venv | 24, 25 |
| GRU | 기존 유지 | 메인 venv | 24, 25 |
| LSTM | 기존 유지 | 메인 venv | 24, 25 |
| PatchTSTLike(`engine/models.py` 자체구현) | 폐기 → `neuralforecast.models.PatchTST`로 교체 | 메인 venv | 24(자체구현판, 제외됨) → 26(공식판, 예정) |
| ITransformerLike(`engine/models.py` 자체구현) | 폐기 → `neuralforecast.models.iTransformer`로 교체 | 메인 venv | 24(자체구현판, 제외됨) → 26(공식판, 예정) |
| TCN | 신규 도입(`neuralforecast.models.TCN`) | 메인 venv | 26(예정) |
| DLinear | 신규 도입(`neuralforecast.models.DLinear`) | 메인 venv | 26(예정) |
| NLinear | 신규 도입(`neuralforecast.models.NLinear`) | 메인 venv | 26(예정) |
| Autoformer | 신규 도입(`neuralforecast.models.Autoformer`) | 메인 venv | 26(예정) |
| TimesNet | 신규 도입(`neuralforecast.models.TimesNet`) | 메인 venv | 26(예정) |
| TimeXer | 신규 도입(`neuralforecast.models.TimeXer`) | 메인 venv | 26(예정) |
| ModernTCN | 신규 도입(공식 GitHub, 어댑터 작성 필요) | 메인 venv(추정, 미확인) | 26(예정) |
| S-Mamba | 신규 도입(공식 GitHub, 어댑터 작성 필요) | 메인 venv(추정, 미확인) | 26(예정) |
| VanillaTransformer | 공식 구현 있음(`neuralforecast.models.VanillaTransformer`), 비채택 | — | 중복 비교 가치 낮다고 판단, 로스터 제외 |
| Chronos-Bolt | 신규 도입 | 메인 venv | 26(예정) |
| TimesFM | 신규 도입 | 메인 venv | 26(예정) |
| TTM | 신규 도입 | 메인 venv | 26(예정) |
| Moirai-2.0 | 신규 도입 | 격리 venv `.venvs/moirai_py312_20261004_192438` | 26(예정) |
| Sundial | 신규 도입 | 격리 venv `.venvs/legacy_hf_py312_20261004` | 26(예정) |
| Time-MoE | 신규 도입 | 격리 venv `.venvs/legacy_hf_py312_20261004`(Sundial과 공유) | 26(예정) |
| Lag-Llama | 신규 도입 | 격리 venv `.venvs/lagllama_py312_20261004` | 26(예정) |
| TimeGPT | 미채택 | — | API 계정/키 필요 |
| xLSTM | 미채택 | — | 서버에 `nvcc` 없어 CUDA 커널 빌드 불가 |
| GDN(Gated DeltaNet) | 미채택 | — | 시계열 전용 공식 구현 없음(언어모델용 빌딩블록만 존재) |
| TSMamba, TimeFound, MOMENT | 미채택 | — | 공식 구현 미확정, 재조사 필요 |

## 메인 venv와 다른 격리 venv 3종

`pyproject.toml`의 torch 전용 인덱스 고정 덕분에 메인 venv는 패키지를 추가해도 조용히
버전이 밀리지 않는다. 메인 venv와 충돌하는 모델만 아래처럼 전용 venv를 쓴다.

| venv | 핵심 버전 |
| :--- | :--- |
| `.venvs/moirai_py312_20261004_192438` | `uni2ts==2.0.0`, `torch==2.4.1+cu121` |
| `.venvs/legacy_hf_py312_20261004` | `transformers==4.40.1` |
| `.venvs/lagllama_py312_20261004` | `gluonts<=0.14.4`, `pandas==2.1.4`, `setuptools<81` |

버전 충돌을 풀어낸 구체적인 과정(에러 메시지·해결 순서)과 속도 실측값은
`test/results/26_fm_env_log_20261004/26_fm_env_log_report.md`에 있다 — 그건 실행 로그라
여기 다시 적지 않는다.
