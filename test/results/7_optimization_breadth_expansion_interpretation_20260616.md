# 7번 최적화 확장 실험 결과 해석

작성일: 2026-06-16

## 1. 한 줄 결론

7번은 모델 성능 리더보드가 아니라, 6번 안정화 이후 어떤 breadth expansion을 어떤 자원 profile과 실행 순서로 돌릴지 정리한 **자원 인식형 stage plan**이다. 따라서 7번 출력은 “누가 더 잘 맞혔는가”를 말해주지 않고, “어떤 실험을 다음에 어떤 조건으로 돌려야 하는가”를 말해준다.

## 2. 7번에서 실제로 나온 것

7번 노트북 출력은 다음 네 가지를 보여준다.

1. 학교 서버/로컬 환경의 CPU, RAM, GPU, CUDA, PyTorch 상태
2. `school_4090_15gb` 자원 프로필
3. `breadth_probe`, `ensemble_probe`, `normalization_cross_check`, `loss_cross_check`, `scale_confirmation` 같은 후속 suite 정의
4. 각 suite를 어떤 backend로 연결할지에 대한 실행 계획

즉 출력의 중심은 모델 손실 곡선이 아니라 실행 계획이다.

## 3. 왜 이게 중요한가

연구 흐름상 7번은 쓸모없는 텍스트가 아니다. 6번에서 target, normalization, loss, model selection을 안정화한 뒤, 다음 단계에서 무엇을 넓게 볼지 정하지 않으면 다시 실험이 흔들린다.

특히 7번은 다음 문제를 먼저 막기 위해 존재한다.

- 서버 자원을 넘는 무제한 병렬화
- OOM 발생 시 중간 결과를 잃는 장시간 단일 실행
- 모델 이름만 늘리고 학습 방식은 그대로 두는 실험 왜곡
- 결과가 없는데도 성능 해석을 먼저 해버리는 상황

## 4. 왜 사용자의 의도와 달랐는가

사용자가 7번에 기대한 것은 “6번 이후의 실제 확장 학습 결과”였다. 하지만 현재 7번 코드에는 확장 모델군 전체를 직접 학습하는 backend가 없었다. 그래서 결과 셀에는 다음이 없었다.

- 학습 곡선
- 예측 vs 실제 그래프
- collapse diagnostics
- leaderboard CSV
- preprocessing / loss / optimizer / gradient policy 비교 결과

따라서 7번을 성능 결과로 읽으면 안 되고, 실험 준비본으로 읽어야 한다.

## 5. 7번 출력의 해석

7번 출력이 의미하는 것은 다음과 같다.

- `school_4090_15gb`는 GPU가 있는 서버에서 순차적으로 돌릴 기준 자원 프로필이다.
- `breadth_probe`는 확장 모델군을 넓게 훑는 약식 비교다.
- `ensemble_probe`는 단일 모델보다 혼합 구조가 collapse를 줄이는지 본다.
- `normalization_cross_check`는 스케일링 방식이 결과를 얼마나 바꾸는지 본다.
- `loss_cross_check`는 목적함수가 shortcut collapse를 줄이는지 본다.
- `scale_confirmation`은 작은 실험에서 본 결론이 더 큰 학습 규모에서도 유지되는지 확인한다.

즉 7번의 핵심 결과는 “실험군의 범위와 순서를 고정했다”는 것이다.

## 6. 무엇을 반영해야 하는가

7번을 읽고 바로 반영해야 하는 것은 다음이다.

- 알고리즘만 늘리지 말고 preprocessing과 normalization도 늘려야 한다.
- 손실함수도 Huber만 보지 말고 directional, anti-collapse, quantile 계열을 함께 봐야 한다.
- optimizer와 scheduler, gradient clipping 방식도 학습 결과를 바꿀 수 있으므로 비교해야 한다.
- 앙상블이 실제로 persistence baseline을 넘어서는지 봐야 한다.
- 각 case마다 중간 결과를 저장해, 한 case가 실패해도 전체 실험이 날아가지 않게 해야 한다.

## 7. 다음 단계

7번 자체는 성능 결과가 아니므로, 실제 비교 판단은 8번에서 해야 한다. 8번은 아래를 포함하도록 새 번호로 분리했다.

- preprocessing cross-check
- normalization cross-check
- loss cross-check
- optimization cross-check
- gradient cross-check
- core tuning probe

## 8. 교수님께 전달할 때의 해석

교수님께는 7번을 다음처럼 읽어야 한다.

- 7번은 결과가 아니라 다음 실험을 위한 실행 기준이다.
- 7번이 보여주는 것은 모델 우열이 아니라 실험 순서, 자원 제약, 비교 축이다.
- 실제 결과 해석은 8번의 GPU 학습 출력이 생겨야 가능하다.

## 9. 관련 문서

- [7번 stage plan](C:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/test/experiment_specs/7_optimization_breadth_expansion_plan_20260613.md)
- [8번 training plan](C:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/test/experiment_specs/8_optimization_breadth_training_plan_20260616.md)
- [8번 training code](C:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/test/models/8_optimization_breadth_training_test.py)

