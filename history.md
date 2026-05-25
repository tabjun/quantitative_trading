# 연구 수행 이력 (Research History / State Persistence)

이 파일은 Antigravity CLI(`agy`) 에이전트가 새로운 세션으로 넘어가거나 세션이 주기적으로 리셋될 때 **'기억 상실'을 방지하기 위한 영구 저장소(Memory)**입니다.
에이전트는 단일 작업이나 단계를 마무리하기 전, 반드시 `.antigravityrules`에 의거하여 이 파일의 테이블과 요약 섹션을 최신화해야 합니다.

## 📝 History Log

| Date | Phase / Step | Task Details (수행 내용) | Output & Result (산출물) | Next Action (다음 목표) |
| :--- | :--- | :--- | :--- | :--- |
| 2026-05-16 | Phase 0 | 프로젝트 기획안 분석 및 다변량 분석 구조 설계, 아키텍처 다이어그램 업데이트 | `README.md`, `process.md`, `skills.md`, `history.md` 재구성 완료 | Phase 1 (Data Mart 구축) 진입 대기 |
| 2026-05-18 | Phase 2 / Test | 시계열 예측 모델 성능 비교 분석 수행 및 보고서 작성 | `test/1_time_series_test.ipynb`, `test/analysis_report.md`, `README.md` 업데이트 | Phase 2 (다변량 분석 엔진) 고도화 및 감성 데이터 결합 |
| 2026-05-19 | Phase 2 / Advance | 고도화된 단변량 분석 (DuckDB 연동, TCN, XGBoost, 가중 앙상블) | `test/2_time_series_advance_test.ipynb`, `upbit_data.db` | 다변량 분석 및 뉴스 데이터 결합 |
| 2026-05-20 | Phase 2 / Reporting | 유틸리티 확충, MCP 논문 탐색, 모델 해석 고도화 및 리포트 자동화 | `test/scripts/*.py`, `test/results/advanced_analysis_report.md` | 다변량 분석(Phase 2.3) 및 감성 데이터 결합 |
| 2026-05-20 | Env Migration | Gemini CLI에서 Antigravity CLI(`agy`)로 이식 및 규칙 전사화 완료 | `.antigravityrules` 생성, `process.md`, `history.md`, `skills.md`, `README.md` 개편 완료 | Phase 2 / Step 2.1 (FastDTW 패턴 매칭 엔진 개발) |
| 2026-05-24 | Phase 2 / Step 2.0 (Advance) | 지연 예측(Lagging) 극복 및 시계열 전처리 다변화(차분, 로그수익률, Z-score), 2년 훈련/1년 예측 strict 분할, 잔차 통계 진단(DW, JB, Ljung-Box) 및 한글 레포트 생성 모듈, 5종 고화질(300 DPI) 로컬 이미지 저장 아카이브 구축, PyTorch LSTM/GRU `input_size` 차원 불일치(Expected 1, got 60) shape 문제 해결 | `test/models/2_time_series_advance_test.ipynb`, `test/models/2_time_series_advance_test.py`, `test/images/*` | Phase 2 / Step 2.1 (FastDTW 패턴 매칭 엔진 개발) |
| 2026-05-25 | Phase 2 / Framework Doc | 6대 시계열 전처리 수식, 역변환 매커니즘, 5대 고도화 손실함수, 15종 알고리즘 상세, 8대 종합 평가지표(MASE, DTW 등) 및 3대 오차 검정법을 총망라한 종합 실험 프레임워크 명세서 구축 완료. 프로젝트 내 4개 유틸리티 스크립트(create_advance_nb.py 등)에 상세 독스트링 기술 설명 전면 반영 | `test/experiment_specs/2_time_series_advance_test_framework_design_20260525_050836.md` | Phase 2 / Step 2.1 (FastDTW 패턴 매칭 엔진 개발) |

---

## 🔄 세션 인계를 위한 핵심 요약 (Context for New Session)

새로운 AI 세션이 시작되거나 세션 리셋 후 복원될 때, `agy` 에이전트는 이 섹션을 정밀하게 읽고 즉각적으로 현재 프로젝트의 스탠스와 목표를 파악해야 합니다.

- **프로젝트 핵심 기조:** 단일 시계열 분석을 배제하고 뉴스/매크로가 결합된 **다변량 분석(Multivariate)**을 수행. 매매 전략의 제1원칙은 **"최하방 방어(MDD 최소화)"**.
- **가장 최근에 완료된 작업 (2026-05-25):** 
  - 15종 시계열 모델들의 예측 지연(Lagging) 극복을 위해 타겟을 가격 원본에서 1차 차분(Difference) 및 로그 수익률(LogReturns)로 개편 및 역복원(Reconstruction) 파이프라인 개발 완료.
  - 학습 손실이 1 에포크만에 수직낙하 하던 identity mapping 편법 문제를 규명 및 지연극복 전처리 파이프라인으로 해결 완료 (점진적 손실 하강 달성).
  - 통계적 잔차 진단(Durbin-Watson, Jarque-Bera, Ljung-Box 검정)에 기반한 한글 잔차 자동 진단 리포터 모듈을 탑재하여 퀀트 모델의 설명력 제고 완료.
  - 엄격한 2년 학습, 1년 테스트 시계열 strict split 구현 완료. 300 DPI 5종 이미지 로컬 저장 연동.
  - **종합 실험 설계 명세화:** 6대 전처리(수식 및 역변환 매핑 포함), 2대 분할 전략, 5대 손실 함수(수익률 방향 패널티 `DALC` 포함), 15종 딥러닝 아키텍처(Mamba, mTAND 등 SOTA 포함), 8대 평가지표(MASE 및 기하학적 DTW 거리 포함), 3대 통계 잔차 검정법을 총정리한 종합 실험 프레임워크 설계 명세서(`test/experiment_specs/2_time_series_advance_test_framework_design_20260525_050836.md`) 구축 완료.
  - **PyTorch Shape Mismatch 해결:** `PreprocessingPipeline.transform` 전처리 메서드에서 `.flatten()`을 제거하여 최종 스케일 데이터가 `(N, 1)` 형태를 유지하도록 고침으로써 `create_sequences` 출력의 차원을 `(num_samples, 60, 1)`로 정비. 이를 통해 PyTorch RNN/LSTM 모델 계열의 `input_size=1` 차원과 완벽하게 정합되도록 조치 완료 (기존 `input.size(-1) Expected 1, got 60` 에러 전면 해소).
- **지금 당장 시작해야 할 작업 (Next Step):** `process.md`의 **Phase 2, Step 2.1** (시계열 패턴 매칭 DTW 구현) 및 Step 2.3 (다변량 통합 로직 결합).
- **에이전트 주의사항:** 로컬 환경에서는 학교 서버 GPU 연산 환경을 대용하여 python 파일 수정 및 ipynb 셀 동기화 작업까지만을 수행하고 직접 학습 실행은 수행하지 않을 것.
 
---
*(AI 에이전트에게: 작업을 종료할 때 이 하단 요약 내용과 위 테이블을 수정하여 다음 에이전트가 완벽히 이어받을 수 있게 하십시오.)*
