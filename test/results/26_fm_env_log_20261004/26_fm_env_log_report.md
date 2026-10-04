# 26번 — 시계열 파운데이션 모델·최신 아키텍처 카탈로그 (2026-10-04)

> 이 문서는 **파운데이션 모델 7종의 상세 근거 자료**다. 전체 모델 로스터(기존 15종 +
> 공식 구현 교체 10종 + 이 문서의 파운데이션 모델 7종)와 연구 환경, 미채택 사유는
> `test/experiment_specs/26_model_expansion_plan_20261004.md`(단일 출처)에 정리돼 있다.

## 작성 배경과 범위

24번·25번을 거치며 Transformer 계열(PatchTST·iTransformer)이 naive보다 열세였던 원인을
"추세를 못 따라가는 것인지, 입력·아키텍처 선택이 잘못된 것인지" 구분해 달라는 요청이 있었다.
이를 계기로 윈도우 슬라이딩 방식의 어텐션 Transformer뿐 아니라, Mamba 계열과 최신 시계열
파운데이션 모델(언어 모델과 유사하게 대규모 사전학습 후 zero-shot 추론하는 구조)까지 조사
범위를 넓혔다. arXiv·Hugging Face를 모두 검색해 찾을 수 있는 최신 구조를 전부 조사했고,
공식 패키지(pip)가 없어도 **공식 GitHub 구현이면 채택 가능**하다는 기준(AGENTS.md 2.13 관련
확인)에 따라 자체 재구현 없이 원 저자 코드를 그대로 가져다 썼다. 계정·API 키가 필요한 모델
(TimeGPT)은 이번 결정에 따라 제외한다.

이 문서에 적은 추론 속도는 **KRW-BTC 단일 종목, 단일 설정에서 측정한 환경·처리량 확인용
수치**이며, 모델 성능(정확도) 비교가 아니다. 성능 비교는 기존 정책대로 26번 드라이버에서
**전체 20종목을 한 번에** 돌려 25번과 같은 유의성 검정으로 수행한다(단일 종목 사전검증 금지
원칙 — 이 문서의 속도 측정은 "모델이 이 서버에서 돌아가는가"만 확인하는 것이라 이 원칙의
적용 대상이 아니다).

## 최종 채택 — 파운데이션 모델 7종

| 모델 | 소속/저자 | 핵심 논문 | 공식 구현 | 실행 환경 | 실측 속도(1000건 기준) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Chronos-Bolt | Amazon | Ansari 외, *Chronos: Learning the Language of Time Series* (2024) | `huggingface.co/amazon/chronos-bolt-*`, pip `chronos-forecasting` | 메인 venv | 로드·추론 성공 확인(세부 처리시간 재측정 필요) |
| TimesFM | Google Research | Das 외, *A decoder-only foundation model for time-series forecasting* (2024) | `github.com/google-research/timesfm`, pip `timesfm` | 메인 venv | 로드·추론 성공 확인(세부 처리시간 재측정 필요) |
| TTM (Tiny Time Mixers) | IBM Granite | Ekambaram 외, *Tiny Time Mixers (TTMs): Fast Pretrained Models for Enterprise Time-series Forecasting* (2024) | `github.com/ibm-granite/granite-tsfm`, pip `granite-tsfm` | 메인 venv | 로드·추론 성공 확인(세부 처리시간 재측정 필요) |
| Moirai-2.0 | Salesforce AI Research | Woo 외, *Unified Training of Universal Time Series Forecasting Transformers* (2024) | `github.com/SalesforceAIResearch/uni2ts`(pip `uni2ts`) | 격리 venv `.venvs/moirai_py312_20261004_192438` (uni2ts 2.0.0, torch 2.4.1+cu121) | 0.13초 |
| Sundial | Tsinghua THUML | *Sundial: A Family of Highly Capable Time Series Foundation Models* (2025) | `github.com/thuml/Sundial` | 격리 venv `.venvs/legacy_hf_py312_20261004` (transformers 4.40.1 고정) | 0.68초 |
| Time-MoE | Time-MoE 연구팀 | *Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts* (2024) | `github.com/Time-MoE/Time-MoE` | 격리 venv `.venvs/legacy_hf_py312_20261004` (Sundial과 동일 환경 공유) | 1.87초 |
| Lag-Llama | time-series-foundation-models | Rasul 외, *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting* (arXiv:2310.08278, 2024) | `github.com/time-series-foundation-models/lag-llama` | 격리 venv `.venvs/lagllama_py312_20261004` (gluonts<=0.14.4, pytorch-lightning, pandas==2.1.4, setuptools<81) | 9.89초(batch_size=64) |

### 모델별 아키텍처 특징

**Chronos-Bolt** — T5 기반 인코더-디코더를 시계열에 맞춰 개조한 모델이다. 원조 Chronos가
값을 이산 토큰으로 양자화해 언어 모델처럼 한 스텝씩 자기회귀 생성했던 것과 달리, Bolt는
입력을 패치 단위로 묶어 인코더에 넣고 디코더가 **다중 스텝을 한 번에 직접 회귀**한다. 이
직접예측 구조 덕분에 원조 Chronos보다 수십 배 빠르면서 정확도는 유지된다.

**TimesFM** — 디코더 전용(decoder-only) Transformer를 쓰지만 자기회귀로 한 스텝씩 생성하는
LLM과 달리, 각 디코딩 스텝에서 **긴 출력 패치**(예: 128 스텝)를 한 번에 내놓는다. 입력도
패치 단위로 토큰화해 가변 길이 시계열과 가변 예측 구간을 하나의 사전학습 모델로 처리한다.

**TTM (Tiny Time Mixers)** — 어텐션을 전혀 쓰지 않는 경량 MLP-Mixer 계열이다. 채널 간,
패치 간 혼합을 전부 완전연결층으로 처리해 파라미터 수가 다른 파운데이션 모델 대비
수백 분의 1 수준이며, 그만큼 추론이 빠르다. "작고 빠른 사전학습 모델"을 목표로 설계되었다.

**Moirai / Moirai-2.0** — 마스크드 인코더 Transformer다. "any-variate attention"으로
다변량 시계열의 변수 수가 바뀌어도 같은 모델을 그대로 쓸 수 있게 했고, 여러 패치 크기에
대응하는 다중 입출력 프로젝션층을 두어 샘플링 주기가 다른 데이터를 함께 사전학습했다.
torch 2.4.1+cu121을 요구해 메인 venv(torch 2.10.0+cu126)와 충돌하므로 격리 venv에서 돌린다.

**Sundial** — 자기회귀 디코더이지만 각 스텝에서 다음 값 하나를 점으로 예측하는 대신,
**플로우 매칭(flow matching)** 으로 연속 확률분포 자체를 생성한다(TimeFlow loss). 패치
단위 입력이며 네이티브 확률적 예측(여러 미래 경로를 샘플링)을 지원한다. 공식 요구사항대로
`transformers==4.40.1`, Python 3.10 계열에서만 안정적으로 동작한다(메인 venv의
transformers 5.16.1에서는 `DynamicCache.seen_tokens` 속성 에러로 로드 실패).

**Time-MoE** — 디코더 전용 Transformer에 **희소 활성화 전문가 혼합(Mixture-of-Experts)** 층을
넣어, 추론 시 전체 파라미터 중 일부만 활성화시키면서도 모델 용량(파라미터 총량)은 키울 수
있게 한 구조다. Sundial과 같은 이유로 `transformers==4.40.1` 환경이 필요하다.

**Lag-Llama** — LLaMA 스타일 디코더 전용 Transformer를, 원시 값이 아니라 **lag(시차) 특징**
위에서 학습시킨 최초의 공개 시계열 파운데이션 모델이다. 각 타임스텝마다 Student-T 분포의
모수를 출력해 확률적 예측을 하며, 다중 스텝 예측을 **한 스텝씩 순차 샘플링**해 만든다
(원하는 샘플 경로 수만큼 이 과정을 반복). 이 순차 자기회귀 샘플링 때문에, 패치 단위로
다중 스텝을 한 번에 내놓는 Moirai·Sundial·Time-MoE보다 1000건당 처리시간이 5~75배
더 걸린다 — 버그가 아니라 구조적 특성이다. GluonTS(`gluonts<=0.14.4`) 위에 얹혀 있어
`pytorch_lightning`이 요구하는 `pkg_resources`가 최신 `setuptools`(81 이상)에서 기본
제거된 문제, `torch>=2.6`의 `weights_only=True` 기본값 변경으로 체크포인트 로드가 막히는
문제, `pandas>=2.2`에서 `'Q'` 오프셋 별칭이 폐기돼 GluonTS 내부 lag 계산표가 깨지는 문제까지
**세 가지 버전 충돌을 순서대로 해결**해야 동작했다(아래 "환경 구성 기록" 참고).

## 미채택 모델과 사유

| 모델 | 사유 |
| :--- | :--- |
| TimeGPT (Nixtla) | API 계정/키가 필요해 제외(2026-10-04 사용자 지시) |
| xLSTM | CUDA 커널을 import 시점에 JIT 컴파일하는데, 이 서버에 `nvcc`(CUDA 컴파일러)가 설치돼 있지 않다. 시스템 레벨 설치라 `sudo` 권한이 필요해 해결 불가. 전용 격리 venv(`.venvs/xlstm_py313_20261004`)에서도 동일하게 실패했다. 메인 venv에 한때 설치해 봤다가 `neuralforecast` 전체 import가 깨지는 사고로 이어져 즉시 제거했다(`uv remove xlstm`) |
| GDN, TSMamba, TimeFound, MOMENT | arXiv·GitHub 재조사에서 유지보수 중인 공식 구현을 확정하지 못했다(Sundial만 `github.com/thuml/Sundial`로 확정됨). "자체 재구현 금지" 원칙상 공식 구현이 불확실한 채로는 채택하지 않는다. 추후 공식 저장소가 명확히 확인되면 재검토한다 |

## 환경 구성 기록 (재현용)

### Moirai — `.venvs/moirai_py312_20261004_192438`
Python 3.12, `uni2ts==2.0.0`, `torch==2.4.1+cu121`. 메인 venv의 torch 2.10.0+cu126과
호환되지 않아 전용 venv로 분리했다. 0.13초/1000건으로 동작 확인.

### Sundial·Time-MoE — `.venvs/legacy_hf_py312_20261004`
Python 3.12, `transformers==4.40.1`(공식 README가 명시한 정확한 버전). 메인 venv
(transformers 5.16.1)에서는 `AttributeError: 'DynamicCache' object has no attribute
'seen_tokens'`로 실패했고, 처음 시도한 격리 venv(transformers 4.46.3)에서도 다른 에러
(`RuntimeError: tensor size 192 vs 12`, 패치 어텐션 마스크 처리 문제)가 나 공식 요구사항대로
4.40.1까지 정확히 맞추고서야 둘 다 성공했다. Sundial 0.68초, Time-MoE 1.87초(모두 1000건 기준).

### Lag-Llama — `.venvs/lagllama_py312_20261004`
Python 3.12, `gluonts[torch]<=0.14.4`(공식 `requirements.txt` 명시), `pytorch-lightning`,
`pandas==2.1.4`, `setuptools<81`. 해결한 순서대로 기록한다.

1. `ModuleNotFoundError: No module named 'pkg_resources'` — `pytorch_lightning`이 의존하는
   `lightning_fabric`이 `pkg_resources`를 직접 import하는데, 최신 `setuptools`(84.0.0)는 이
   모듈을 기본 제거했다. `setuptools<81`로 다운그레이드해 해결했다.
2. `ValueError: Invalid frequency: Q` — GluonTS의 내부 lag 계산표(`get_lags_for_frequency`)가
   `pandas>=2.2`에서 폐기된 `'Q'`(분기) 오프셋 별칭을 그대로 쓰고 있어, 최신 pandas에서
   예외가 난다. `pandas==2.1.4`로 고정해 해결했다(이 버전은 `'Q'`를 경고 없이 받아들인다).
3. `_pickle.UnpicklingError`(weights_only) — `torch>=2.6`부터 `torch.load`의 기본값이
   `weights_only=True`로 바뀌어, 체크포인트에 포함된 `gluonts.torch.distributions
   .studentT.StudentTOutput` 클래스를 신뢰할 수 없는 전역 객체로 간주해 로드를 거부한다.
   공식 Hugging Face 체크포인트(신뢰 가능한 출처)이므로 `torch.serialization
   .add_safe_globals([StudentTOutput])`를 등록하고 `torch.load`를 `weights_only=False`
   기본값으로 패치해 해결했다.

체크포인트(`lag-llama.ckpt`)는 `huggingface_hub.hf_hub_download`로 익명(토큰 없이)
다운로드했다 — 계정이나 API 키가 필요하지 않다.

## 다음 단계

이 카탈로그를 근거로 26번 드라이버 코드를 작성한다. neuralforecast 어댑터의 시간축 압축·
입력 오염 버그(결측 시점을 학습구간 평균으로 채우고 `available_mask=0`으로 손실만 제외하는
해결책, 이미 검증됨)를 드라이버에 반영해야 하며, 격리 venv 3종(Moirai·Sundial/Time-MoE·
Lag-Llama)에 걸쳐 실행되는 모델들의 예측 결과를 어떻게 모아 25번과 같은 유의성 검정
파이프라인에 합칠지 설계해야 한다.
