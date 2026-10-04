# 26번 계획서 — 모델 로스터 확대(최신 아키텍처·파운데이션 모델 전면 편입)

작성일: 2026-10-04 · 브랜치: `stock` · 근거: `process.md`의 미해결 항목 2건(최신 파운데이션
모델 미검증, `engine/models.py`에 구현만 돼 있고 24번이 쓰지 않은 모델 10개), 25번 완료 후
사용자 지시("최신 핫한 구조 전부 다 가져와봐 arxiv든 hf든 어디든 다 검색해서 찾아봐")

> 이 문서는 **26번이 실제로 어떤 모델을 쓰기로 했는가(결정)**만 다룬다. 모델 자체의
> 아키텍처 특징과 논문 근거는 전부 `test/research_materials/model_catalog.md`(번호 없는
> 평시 카탈로그 — 어느 실험이 썼는지와 무관하게 누적되는 모델 백과사전, 2026-10-04 신설)에
> 있다. 이 계획서는 그 카탈로그를 "26번에 쓸지 말지" 결정하는 자리이고, 환경 구축 중
> 겪은 버전 충돌·실측 속도 같은 **이번 실행에 한정된 로그**는
> `test/results/26_fm_env_log_20261004/26_fm_env_log_report.md`에 따로 둔다. 세 문서의
> 역할: 카탈로그=모델이 무엇인가(불변), 이 계획서=무엇을 쓰기로 했나(결정), env_log=
> 어떻게 돌렸나(실행 기록).

## 1. 왜 지금 이 실험인가

24번·25번은 17개 모델(통계·트리·커널·GRU/LSTM·self-built Transformer 2종)만 비교했다.
`process.md`에 두 가지 공백이 남아 있었다.

- **파운데이션 모델이 하나도 없다**: Chronos·TimesFM·Moirai·TTM 등 사전학습 모델을
  패키지조차 설치하지 않은 채 "최신 모델 비교"라고 부르고 있었다.
- **이미 구현된 모델 10개가 24번에 편입되지 않았다**: `engine/models.py`에 TCN·DLinear·
  NLinear·PatchTST·Autoformer·iTransformer·ModernTCN·Mamba·TimesNet·TimeXer 10개 아키텍처가
  전부 구현돼 있다(8~13번 시절 작성). 이 중 PatchTSTLike·ITransformerLike 2개만 24번에
  실제로 쓰였고 둘 다 학습 실패로 제외됐다. 나머지 8개(TCN·DLinear·NLinear·Autoformer·
  ModernTCN·Mamba·TimesNet·TimeXer)는 **구현만 있고 한 번도 돌려본 적이 없다.**

게다가 `engine/models.py`의 모든 Transformer·Mamba 계열은 **원 논문 설계를 직접 간이
재현한 자체 구현**이다(AGENTS.md 2.13 신설 계기가 된 PatchTSTLike의 채널혼합 오류가 바로
이 문제). 26번에서는 가능한 모든 모델을 **원 저자 공식 구현(pip 패키지 또는 공식 GitHub)**
으로 교체하거나, 공식 구현이 없는 경우에만 자체 구현을 유지한다.

## 2. 연구 환경

| venv | Python | 핵심 버전 | 용도 |
| :--- | :--- | :--- | :--- |
| `.venv`(메인) | 3.12 | `torch==2.10.0+cu126`(전용 인덱스 고정, `pyproject.toml`), `neuralforecast>=3.2.2`, `chronos-forecasting>=2.3.2`, `timesfm>=3.0.2`, `granite-tsfm>=0.3.2` | 기존 17개 모델 + neuralforecast 전체 아키텍처 + Chronos-Bolt·TimesFM·TTM |
| `.venvs/moirai_py312_20261004_192438` | 3.12 | `uni2ts==2.0.0`, `torch==2.4.1+cu121` | Moirai-2.0(메인 venv의 torch 2.10과 비호환) |
| `.venvs/legacy_hf_py312_20261004` | 3.12 | `transformers==4.40.1`(공식 요구사항 고정) | Sundial, Time-MoE(둘 다 메인 venv의 transformers 5.16.1과 비호환) |
| `.venvs/lagllama_py312_20261004` | 3.12 | `gluonts<=0.14.4`, `pytorch-lightning`, `pandas==2.1.4`, `setuptools<81` | Lag-Llama(독자적인 구버전 스택 요구) |

메인 venv의 torch 버전은 `pyproject.toml`의 PyTorch 전용 인덱스(`pytorch-cu126`)로 고정돼
있어, 다른 패키지를 추가해도 조용히 밀리지 않는다(89c19b4 사고 이후 재발 방지 장치).
새 모델 추가로 메인 venv가 깨질 위험이 있으면(xLSTM 사례처럼) **즉시 전용 격리 venv를
새로 판다** — 사용자 승인(2026-10-04, "라이브러리 충돌하면 venv 하나 더 만들어서 거기도
하면 되지")에 따른 표준 대응이다. 격리 venv 3종을 구축하며 겪은 버전 충돌과 해결 순서,
KRW-BTC 단일 종목 속도 실측값은 `test/results/26_fm_env_log_20261004/
26_fm_env_log_report.md`에 재현 가능하게 기록돼 있다.

## 3. 전체 모델 로스터

### 3-A. 24·25번에서 유지되는 기존 모델 15종 (그대로 둔다)

naive, HAR-RV, Linear, Ridge, KernelRidge-RBF, SVR-RBF, Nystroem+Ridge, LightGBM, XGBoost,
HistGBM, GARCH-t, MS-GARCH, TAR-GARCH, GARCH+LightGBM, GRU, LSTM — 25번에서 유의성까지
확인됐으므로 재설계하지 않는다.

### 3-B. 기존 self-built Transformer 2종 → 공식 구현으로 교체

| 기존(제외됨) | 교체 대상 | 근거 |
| :--- | :--- | :--- |
| PatchTSTLike(`engine/models.py`) | `neuralforecast.models.PatchTST` | 원 논문(채널독립)을 자체 구현이 정반대로(채널혼합) 구현해 AGENTS.md 2.13의 계기가 됐다. probe로 로드·적합·예측 전부 확인됨(`probe26_full.log`: 적합 2.3초, 예측 1회 0.011초) |
| ITransformerLike(`engine/models.py`) | `neuralforecast.models.iTransformer` | neuralforecast가 공식 iTransformer를 제공해(확인됨: `import neuralforecast.models` → `iTransformer` 존재) 자체 재구현을 유지할 이유가 없다 |

### 3-C. `engine/models.py` 미사용 8종 → 공식 구현 교체 결정

| 자체 구현(미사용) | 결정 | 교체처 | 비고 |
| :--- | :--- | :--- | :--- |
| TCNForecaster | 교체 | `neuralforecast.models.TCN` | 공식 패키지 제공 |
| DLinearLike | 교체 | `neuralforecast.models.DLinear` | 공식 패키지 제공 |
| NLinearLike | 교체 | `neuralforecast.models.NLinear` | 공식 패키지 제공 |
| AutoformerLike | 교체 | `neuralforecast.models.Autoformer` | 공식 패키지 제공 |
| TimesNetLike | 교체 | `neuralforecast.models.TimesNet` | 공식 패키지 제공 |
| TimeXerLike | 교체 | `neuralforecast.models.TimeXer` | 공식 패키지 제공 |
| ModernTCNLike | 교체(별도 통합 필요) | [`github.com/luodhhh/ModernTCN`](https://github.com/luodhhh/ModernTCN)(ICLR 2024 Spotlight 공식 구현, pip 패키지 없음) | neuralforecast 미제공. 공식 GitHub 코드를 직접 가져와 우리 데이터 파이프라인에 연결하는 어댑터 작성 필요 — 아직 미착수 |
| MambaLike | 교체(별도 통합 필요) | [`github.com/wzhwzhwzh0921/S-D-Mamba`](https://github.com/wzhwzhwzh0921/S-D-Mamba)(S-Mamba, *Is Mamba Effective for Time Series Forecasting?*, Neurocomputing 2024 공식 구현) | neuralforecast 미제공. ModernTCN과 마찬가지로 어댑터 작성 필요 — 아직 미착수 |

**TransformerForecaster**(바닐라 Transformer, `engine/models.py`에 구현돼 있으나 24번 비교에
없었음)는 `neuralforecast.models.VanillaTransformer`로 대체 가능하나, GRU/LSTM/PatchTST가
이미 순환·어텐션 계열을 대표하고 있어 **이번 회차 로스터에는 포함하지 않는다**(공식 구현이
없어서가 아니라 중복 비교 가치가 낮다고 판단 — 필요하면 후속 회차에 추가).

### 3-D. 신규 파운데이션 모델 7종 (상세는 `model_catalog.md` §6 참조)

Chronos-Bolt, TimesFM, TTM, Moirai-2.0, Sundial, Time-MoE, Lag-Llama — 전부 로드·추론
성공 확인됨. 논문·아키텍처 특징은 `model_catalog.md`에, 실측 속도·버전 충돌 해결 기록은
`26_fm_env_log_20261004/26_fm_env_log_report.md`에 있다.

### 3-E. 미채택 모델 전체 목록과 사유

| 모델 | 사유 |
| :--- | :--- |
| TimeGPT(Nixtla) | API 계정/키 필요(2026-10-04 사용자 지시로 제외) |
| xLSTM | 서버에 `nvcc` 없어 CUDA 커널 JIT 컴파일 실패(격리 venv에서도 동일), 메인 venv를 깨뜨린 사고 있었음(`uv remove xlstm`으로 복구) |
| GDN(Gated DeltaNet) | NVLabs 공식 구현(`github.com/NVlabs/GatedDeltaNet`)은 언어모델용 선형어텐션 빌딩블록이지 시계열 예측 모델이 아니다. 예측 헤드를 직접 얹으면 그 자체가 자체 구현이 되므로, 공식 시계열 예측 구현이 나오기 전까지는 채택하지 않는다 |
| TSMamba, TimeFound, MOMENT | arXiv·GitHub 재조사에서 유지보수 중인 공식 구현을 확정하지 못했다. 공식 구현이 명확해지면 재검토한다 |

## 4. 아직 해결 안 된 설계 과제

1. **neuralforecast 입력 오염 버그 반영**: 결측 시점을 학습구간 평균으로 채우고
   `available_mask=0`으로 손실만 제외하는 해결책이 검증은 됐으나(정상범위 -0.02~0.09 확인),
   26번 드라이버 코드에는 아직 반영되지 않았다. 드라이버 작성 1순위 과제.
2. **ModernTCN·S-Mamba 어댑터**: 두 모델 다 공식 GitHub 코드가 neuralforecast 인터페이스를
   쓰지 않으므로, 우리 `(r, |r|)` 2채널 입력·실현변동성 타깃 형식에 맞춰 데이터를 넣고
   꺼내는 어댑터를 새로 짜야 한다. 어느 venv에서 돌릴지(메인 vs 격리)도 의존성 확인 후 결정.
3. **격리 venv 3종 결과 통합**: Moirai·Sundial/Time-MoE·Lag-Llama는 메인 venv 밖에서 돌아
   예측값만 파일로 받아와야 한다. 24·25번처럼 한 프로세스 안에서 전부 적합하는 구조가 아니라,
   **서브프로세스 호출 + 예측값 직렬화(npz/csv) + 메인 드라이버에서 로드**하는 2단계 구조가
   필요하다. 아직 설계하지 않았다.
4. **hist_exog(외생변수) 지원 확인**: 교체 대상인 TCN·DLinear·NLinear·Autoformer·TimesNet·
   TimeXer가 `hist_exog_list`를 지원하는지 아직 확인하지 않았다(기존에 확인된 건 NHITS·TFT만
   지원, DLinear·PatchTST·iTransformer·TSMixer·TimeMixer는 미지원이라는 사실뿐). 드라이버
   작성 전에 전수 확인 필요.

## 5. 산출물 계획

- 드라이버: `test/models/26_model_expansion_test.py`(아직 미작성)
- 결과: `test/results/26_model_expansion_{YYYYMMDD}/26_model_expansion_*.{md,csv,npz}`
- 이미지: `test/images/26_model_expansion_{YYYYMMDD}/26_model_expansion_*.png`
- 완료 후 25번과 같은 방식(DM+MCS)으로 재검정해 전체 모델(기존 15 + 교체 10 + 신규 7 = 32종)의
  순위를 다시 매긴다.
