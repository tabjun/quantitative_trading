# 26번 — 파운데이션 모델 환경 구축·속도 실측 로그 (2026-10-04)

> 모델 자체의 특징·논문 근거·현재 채택 상태(비고 칸)는 전부
> `test/research_materials/model_catalog.md`(번호 없는 평시 카탈로그)에 있다. 이 문서는
> **격리 venv 3종을 실제로 구축하면서 겪은 버전 충돌과 그 해결 순서, KRW-BTC 단일 종목
> 속도 실측값만** 남긴다 — 재현용 실행 로그다.

이 문서에 적은 추론 속도는 **KRW-BTC 단일 종목, 단일 설정에서 측정한 환경·처리량 확인용
수치**이며, 모델 성능(정확도) 비교가 아니다. 성능 비교는 기존 정책대로 26번 드라이버에서
**전체 20종목을 한 번에** 돌려 25번과 같은 유의성 검정으로 수행한다(단일 종목 사전검증 금지
원칙 — 이 속도 측정은 "모델이 이 서버에서 돌아가는가"만 확인하는 것이라 적용 대상이 아니다).

## 실측 속도 요약(1000건 기준)

| 모델 | 실행 환경 | 실측 속도 |
| :--- | :--- | :--- |
| Chronos-Bolt | 메인 venv | 로드·추론 성공 확인(세부 처리시간 재측정 필요) |
| TimesFM | 메인 venv | 로드·추론 성공 확인(세부 처리시간 재측정 필요) |
| TTM | 메인 venv | 로드·추론 성공 확인(세부 처리시간 재측정 필요) |
| Moirai-2.0 | 격리 venv `.venvs/moirai_py312_20261004_192438` | 0.13초 |
| Sundial | 격리 venv `.venvs/legacy_hf_py312_20261004` | 0.68초 |
| Time-MoE | 격리 venv `.venvs/legacy_hf_py312_20261004`(Sundial과 공유) | 1.87초 |
| Lag-Llama | 격리 venv `.venvs/lagllama_py312_20261004` | 9.89초(batch_size=64) |

Lag-Llama가 느린 이유(순차 자기회귀 샘플링 vs 나머지의 패치 단위 직접예측)는 모델 자체의
구조적 특성이므로 `model_catalog.md`의 Lag-Llama 항목에 적었다 — 이 문서는 수치만 남긴다.

## 환경 구성 기록 (재현용)

### Moirai — `.venvs/moirai_py312_20261004_192438`
Python 3.12, `uni2ts==2.0.0`, `torch==2.4.1+cu121`. 메인 venv의 torch 2.10.0+cu126과
호환되지 않아 전용 venv로 분리했다.

### Sundial·Time-MoE — `.venvs/legacy_hf_py312_20261004`
Python 3.12, `transformers==4.40.1`(공식 README가 명시한 정확한 버전). 메인 venv
(transformers 5.16.1)에서는 `AttributeError: 'DynamicCache' object has no attribute
'seen_tokens'`로 실패했고, 처음 시도한 격리 venv(transformers 4.46.3)에서도 다른 에러
(`RuntimeError: tensor size 192 vs 12`, 패치 어텐션 마스크 처리 문제)가 나 공식 요구사항대로
4.40.1까지 정확히 맞추고서야 둘 다 성공했다.

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
