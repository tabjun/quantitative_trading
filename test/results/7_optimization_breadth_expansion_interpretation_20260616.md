# 7번 최적화 확장 실험 결과 해석

## 결론

현재 `7_optimization_breadth_expansion_test.ipynb`의 출력은 실제 모델 학습 결과가 아니라, 6번 안정화 연구 이후 어떤 확장 실험을 돌릴지 정리한 **자원 인식형 stage plan 출력**이다. 따라서 7번 결과만으로는 Autoformer-like, PatchTST-like, DLinear/NLinear-like, iTransformer-like, Mamba-like, 앙상블 모델이 shortcut collapse를 줄였는지 판단할 수 없다.

## 7번이 실제로 한 일

7번은 학교 서버 환경을 염두에 두고 CPU/RAM/GPU/CUDA/PyTorch 정보를 감지하고, `school_4090_15gb` 자원 프로필을 적용하며, `breadth_probe`, `ensemble_probe`, `normalization_cross_check`, `loss_cross_check`, `scale_confirmation` 같은 후속 실험 suite를 어떤 순서로 실행할지 출력하도록 설계되었다.

즉, 7번은 “학습기가 잘 돌았는가”를 보여준 것이 아니라 “학습기를 돌릴 때 어떤 자원 제한과 실험 단계를 적용해야 하는가”를 보여준 중간 산출물이다.

## 왜 사용자의 의도와 어긋났는가

사용자가 요청한 7번의 본래 의도는 6번에서 정리한 target, normalization, loss, model selection 기준을 실제 알고리즘 학습에 반영하고, 알고리즘과 테스트 케이스를 늘려 학습 곡선, 예측 시각화, 붕괴 진단 그래프, 성능 지표를 함께 비교하는 것이었다.

하지만 현재 7번 코드는 확장 모델군을 설명하고 실행 계획을 출력할 뿐, 확장 모델군 전체를 직접 학습하는 backend가 없다. 특히 현재 코드에는 “현재 저장소에는 이 suite를 직접 학습하는 전용 backend가 아직 없다”는 문구가 남아 있으므로, 7번은 실험 완료본이 아니라 실험 준비본으로 보아야 한다.

## 7번 결과에서 얻을 수 있는 의미

7번 결과의 의미는 제한적이지만 완전히 무의미하지는 않다. 학교 서버에서 GPU 학습을 수행할 때 무제한 병렬 처리, 과도한 DataLoader worker, 과도한 Optuna 병렬 trial, 장시간 단일 실행을 피해야 한다는 자원 운영 기준을 코드화했다는 점은 유효하다.

다만 이 출력은 모델 성능, loss 수렴, 예측 형태, 직전가 복사 여부, 0 수익률 붕괴 여부를 검증하지 않는다. 따라서 연구 결론으로는 “7번은 확장 학습 실험 전 단계의 오케스트레이션 산출물이고, 실제 확장 학습 판단은 8번에서 수행해야 한다”가 맞다.

## 다음 조치

8번은 7번을 덮어쓰지 않고 새 번호로 분리한다. 8번은 실제 GPU 학습형 실험으로 작성하며, 다음 산출물을 생성해야 한다.

- 기초 통계량과 입력 피처 요약
- 시간 순서 train/validation/test split 요약
- 모델별 학습 곡선
- 실제 수익률과 예측 수익률 비교 그래프
- KRW 복원 가격 기준 예측 그래프
- collapse 진단 지표와 시각화
- 모델/손실함수/정규화/앙상블별 leaderboard
- 독립적으로 읽을 수 있는 Markdown 결과 보고서

