# 모델 카탈로그: 연구에 쓴 모든 알고리즘의 특징, 장단점, 문헌, 코드 저장소, 사용한 실험

이 문서는 이 연구가 지금까지 쓴 알고리즘을 이해하려고 읽는 참고 문서다. 모델마다 **특징(원리), 장점, 단점, 참고문헌, 코드 저장소(GitHub 주소), 사용한 실험 번호(비고)**를 적었다. 2026-10-14에 코드와 결과 보고서를 하나씩 대조해 다시 썼고, 실험 결과 자체는 `test/results/{번호}_*/`에 있다. 알고리즘 외의 전처리, 학습 절차, 통계 검정, 평가 지표는 `methodology_catalog.md`가 맡는다.

| 부 | 내용 | 모델 수 |
| :--- | :--- | ---: |
| 제1부 | 고전 통계·계량 모델과 기계학습(기준선, 선형·회귀, GARCH 계열과 레짐 전환, 트리·커널, 결합 방식) | 표 행 35 |
| 제2부 | 딥러닝과 시계열 파운데이션 모델(순환, 합성곱·MLP, 어텐션, 상태공간, 파운데이션) | 34(실행 28, 검토만 6) |
| 부록 A | 실행 설정 점검표(새 모델을 돌리기 전 대조 필수) | |
| 부록 B | 이번 개정에서 바로잡은 기존 카탈로그의 서술 | |

**표기 규칙**
- **미검증 인용**: arXiv나 Crossref로 원문·초록을 대조하지 못한 인용이다. 서지 정보가 틀릴 수 있으므로 논문에 쓰기 전에 원문을 확인한다.
- **확인 못 함**: 저장소 주소나 구현 방식을 확인하지 못한 항목이다. 이유를 같은 칸에 적었다.
- **코드 저장소**: GitHub 주소는 개정 당시 접속해 확인한 것만 적었다. 저장소가 이동한 경우 이동 후 주소를 적었다.
- 퀀트 트레이딩 활용 자료(외부 자료, 논문 아님)는 제2부 7절에 모았다. TimesFM 활용 가이드와 같은 저장소의 다른 문서를 정리했다.

**이 문서를 읽을 때 주의할 점**
- 2~16번 실험의 딥러닝 모델 중에는 원 논문 이름만 딴 직접 근사 구현이 있다. 제2부 6절 "구현 구분 한눈에 보기"에서 공식 구현인지 직접 구현인지 확인한다.
- 27번 이후 신경망의 학습 절차는 "보정은 내부 모델 예측으로, 평가 예측은 재적합 모델로"였다(2026-10-13 확인). 모델 간 순위는 이 불일치의 영향을 받았을 수 있고, 28번에서 재적합을 없애 다시 본다.

---

## 제1부. 고전 통계·계량 모델과 기계학습

### 0. 읽기 전에

#### 0-1. 실험 번호별로 어떤 모델이 쓰였는가

| 실험 번호 | 하는 일 | 이 문서 범위의 모델 |
| :--- | :--- | :--- |
| 1~16 | 다음 봉 가격·수익률의 방향 예측(신경망 중심) | 기준선(직전값 복사 persistence, 추세 지속 momentum), 신경망의 시드 평균 앙상블(10, 15번) |
| 17, 17b, 21, 22 | 탐색적 분석, 전처리 계획 | 17, 21, 22번은 모델 적합 없음(검정과 그림만). 17b번은 18번의 GARCH(1,1)로 조건부 변동성 그림을 재현 |
| 18, 18d | 차분 여부와 전처리 사다리 판정 | 직접 구현한 GARCH(1,1) 정규, 수익률 AR(1), 직전값 기준선, HAR, EWMA, 롤링분산 |
| 19, 19c | 변동성 모델 선택(BTC 19번, 20종목 19c번) | GARCH 정규, GARCH-t, EWMA, 롤링분산, HAR-RV, HAR-RV+, 상수분산 |
| 20 | 방향 대 변동성 예측 가능성 대조 | 로지스틱 회귀(방향), 선형 회귀(변동성), 직전 RV 기준선 |
| 23 | 변동성 예측 모델 통합 비교(1시간 뒤 RV) | naive, HAR-RV, Linear, Ridge, KernelRidge, SVR, Nystroem+Ridge, LightGBM, XGBoost, HistGBM, GARCH-t, GARCH+LightGBM, 레짐 GARCH 예비검증 |
| 24 | 최대 규모 재적합 | 위 모델에 MS-GARCH, TAR-GARCH를 본편으로 편입, 규모와 조기종료 추가 |
| 25 | 순위의 통계적 유의성 | 24번 저장 예측만 읽음(재적합 없음) |
| 26 | 시간 기준 평가 재정립, 5개 예측 구간 | 24번의 모델 전부를 새 평가 틀로 재적합 |
| 26b | 견고성과 유의성 | 26번 저장 예측과 시드 반복을 읽음 |
| 26c | 최근 3년 창과 가격 정지 두 부분 모형 | 정지 확률 분류기(LightGBM)와 두 부분 결합 추가, Linear·Ridge·HAR-RV는 코드에서 뺌 |
| 27, 27b~27e | 신규 딥러닝 확대와 사후 점검 | 26c의 모델 결과를 비교 대상으로 읽음(고전·기계학습 모델은 재적합하지 않음) |

18번과 19번(19c 포함)에서 GARCH-t, HAR-RV로 모델을 골랐던 결과는 2026-09-04 방법론 재정립에서 "중구난방 시기 산물이라 결정에서 제외"하기로 했고(`process.md` 2026-09-04 갱신 항목, `test/models/21_volatility_eda_test.py` 서두), 모델은 EDA부터 다시 시작해 루프의 산출물로 정했다. 그래서 18~19c의 모델은 "그 시기 선택 실험에서 쓴 구현"으로만 적고, 23번 이후 비교와 구별한다.

#### 0-2. 모든 모델이 공유하는 설정(23번 이후)

- **데이터**: 업비트 KRW 마켓 15분봉. 거래대금 상위 10종목과 변동성 상위 10종목으로 20종목을 구성한다.
- **타깃**: 예측 구간 H(23번은 1시간, 26번 이후는 15분·30분·1시간·4시간·12시간) 동안의 실현변동성 RV, 곧 구간 안 15분 로그수익률 제곱합의 제곱근이다. 가격이 움직이지 않은 구간은 RV가 0이다.
- **입력 형태**: 모델군마다 다르게 준비한다. 통계 모형(GARCH 계열)은 주기 제거 수익률을 직접 받고, 회귀·커널·트리 모델은 과거 수익률에서 만든 표 형태 특성(아래 메모)을 받는다. 로그 타깃 모델은 로그 RV를 맞춘 뒤 지수로 되돌린다.
- **평가**: QLIKE를 주 지표로 쓴다(Patton 2011). 로그 타깃 모델은 26번부터 역변환 편향을 내부검증 구간에서 구한 보정 상수 c로 맞추고, 예측값은 학습 구간 타깃의 [최솟값, 최댓값] 밖이면 학습 평균으로 바꾸는 insanity filter(Bollerslev, Patton, Quaedvlieg 2016)를 거친다(24번은 0.5%~99.5% 분위수를 쓰는 더 공격적인 방식이었고, 26c 코드 주석 311~312행이 이를 원 논문 방식으로 되돌렸다고 적는다).
- **분할**: 23·24번은 종목마다 앞 70%를 학습, 나머지를 평가로 나눴고, 26번은 전 종목 공통 분할 시각(2025-08-26), 26c번은 2025-11-11이다. 24번 이후는 학습 구간을 다시 내부학습 85%와 내부검증 15%로 나눠 하이퍼파라미터와 조기종료를 정한다.
- **구현 라이브러리 버전**(`uv.lock`): lightgbm 4.7.0, xgboost 3.2.0 및 3.4.1(두 버전이 잠금 파일에 함께 있어 어느 쪽이 실행 환경인지 확인 못 함), scikit-learn 1.7.2, arch 8.0.0, statsmodels 0.14.6, scipy 1.15.3 및 1.17.1(같은 이유로 확인 못 함).

#### 0-3. 이 문서가 다루지 않는 것

검정 도구(ADF, KPSS, Ljung-Box, ARCH-LM, RESET, BDS, Breusch-Pagan, Diebold-Mariano, Model Confidence Set), 분수차분과 Hurst·GPH 추정, PCA·STL·상호정보량 같은 탐색 도구는 예측 모델이 아니라 진단 도구라서 표에 넣지 않았다.

### 1. 기준선

기준선은 학습할 모수가 거의 없는 단순 예측기이며, 복잡한 모델이 실제로 정보를 더하는지 재는 자로 쓴다. 이 연구는 두 시기에 서로 다른 기준선을 썼다. 1~16번(방향 예측)은 직전 가격 복사와 추세 지속을 썼고, 18번 이후(변동성 예측)는 직전 구간 실현변동성, 상수분산, 롤링분산, 지수가중 이동평균 분산을 썼다. 가격 이동평균(SMA) 교차 같은 이동평균 예측기는 코드에서 확인하지 못했고, 이동평균형 기준선은 아래 롤링분산과 EWMA가 대신한다.

| 모델 | 특징 | 장점 | 단점 | 참고문헌 | 코드 저장소 | 사용 연구(비고) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| naive(직전 구간 RV) | 다음 예측 구간의 RV를 직전에 끝난 같은 길이 구간의 RV로 둔다. 학습이 없다 | 추정 오차가 없고 설명이 필요 없다 | 급변이 시작될 때와 끝난 직후를 따라가지 못한다 | Hyndman & Koehler(2006) *Int. J. Forecasting* 22(4), MASE 분모로 쓰는 근거 | 직접 구현 `test/models/23_volatility_model_comparison_test.py`의 `targets`(변수 `pas`), 시간 격자판은 `test/models/26_timebased_reeval_test.py` | 20, 23, 24, 25, 26, 26b, 26c, 27(전 비교표에 포함, MASE 분모) |
| persistence(직전 가격 복사) | 다음 종가를 지금 종가로 두는 방향 예측 기준선(수익률 0 예측과 같다) | 계산이 필요 없고 가격 레벨에서는 매우 강한 비교 대상이다 | 가격 레벨에서는 R²가 1에 가까워 모델을 구별하지 못한다(18번 확인) | Hyndman & Koehler(2006) | 직접 구현 `engine/point.py`(`persistence_close`), `test/models/5_optimization_diagnostics_test.py`(`persistence_gap`), `test/models/18_differencing_decision_test.py`(D2) | 2, 5, 10, 15, 16, 18에서 확인. 12·14번은 `engine/point.py` 경유. 방향 예측 신경망이 직전값 복사를 못 넘는지로 채택을 판단 |
| momentum naive(추세 지속) | 직전 h봉 누적수익률이 다음 h봉에도 이어진다고 예측한다 | 추세를 따르는 가장 단순한 비교 대상이다 | 방향 신호가 거의 없는 자산에서는 무의미하다 | 확인 못 함(이 연구가 만든 기준선, 인용 없음) | 직접 구현 `test/models/16_nonstationary_crosssectional_test.py`(`mase_momentum`) | 16 |
| 다수 클래스(방향) | 학습 구간에서 더 많았던 방향을 항상 예측한다 | 방향 분류의 정확도 하한을 정해 준다 | 정보가 없다 | 해당 없음(표준 정의) | 직접 구현 `test/models/20_direction_vs_volatility_predictability_test.py`(`direction_metrics`의 `base`) | 20 |
| 상수분산 | 학습 구간의 표본 분산을 모든 시점의 예측으로 쓴다 | 변동성 군집을 전혀 쓰지 않는 하한선이다 | 시간 변화를 못 따른다 | 해당 없음(표준 정의) | 직접 구현 `test/models/19_volatility_model_selection.py`(`candidates["상수분산"]`) | 18(D3b), 19, 19c |
| 롤링분산(96봉) | 직전 96봉(하루) 제곱수익률 평균을 다음 분산 예측으로 쓴다 | 모수가 없고 변동성 군집을 가장 단순하게 쓴다 | 창 길이가 임의적이고 갑작스러운 창 이탈로 예측이 튄다 | 해당 없음(표준 정의) | 직접 구현 `test/models/19_volatility_model_selection.py`(`rolling(BARS_PER_DAY)`), 라이브러리는 pandas(`https://github.com/pandas-dev/pandas`) | 18(D3b), 19, 19c |
| EWMA(α=0.06) | 제곱수익률의 지수가중 이동평균을 분산 예측으로 쓴다. 최근 값에 큰 가중을 둔다 | 모수가 하나이고 GARCH(1,1)의 특수한 경우처럼 동작한다 | 장기평균으로 돌아가는 성질이 없다 | RiskMetrics 기술 문서(J.P. Morgan/Reuters 1996)는 미검증 인용 | 직접 구현 `test/models/19_volatility_model_selection.py`(`ewm(alpha=0.06, adjust=False)`), pandas | 18(D3b), 19, 19c |
| 수익률 AR(1) | 로그수익률을 직전 수익률 한 개로 선형 회귀한다 | 평균 방정식의 가장 단순한 형태다 | 수익률의 자기상관이 거의 없어 정보가 없다 | 해당 없음(표준 정의, 교과서 모형) | 직접 구현 `test/models/18_differencing_decision_test.py`(D2, 학습 구간 계수) | 18(레벨축과 차분축 평가를 비교하려는 실모델) |

메모

- **naive의 정의가 실험마다 다르다.** 18번 D2는 다음 로그가격이 현재 로그가격과 같다는 레벨 기준선이고, 20번은 직전 h봉 RV의 로그, 23번과 24번은 직전 4행(1시간)의 RV다. 26번 이후는 15분 시간 격자를 복원해 직전 4H봉(H시간) 창의 RV로 바꿨다. 행 기준이던 24번은 저유동 종목에서 평가 창의 최대 45%가 60분을 넘었다고 26번 설계 표에 적혀 있어, 23·24번과 26번 이후 수치를 직접 빼서 비교하지 않는다.
- **MASE의 분모**가 naive라서 naive의 정의가 바뀌면 MASE의 의미도 같이 바뀐다(2번은 학습 구간 인접 차분의 평균 절대값, 16번은 추세 지속, 23번 이후는 직전 구간 RV).
- 12번과 14번 융합 실험은 가격 방향 전략의 비교 기준으로 `buy_hold`(항상 보유)와 `no_trade`(거래 안 함) 정책도 썼다(`engine/fusion.py`의 `evaluate_fusion_policy`). 이는 예측 모델이 아니라 거래 정책 기준선이라 표에 넣지 않았다.
- 19번의 MASE 분모는 학습 표본 평균 분산(상수분산)이라 23번 이후와 다르다(`score_all`의 `naive`).

### 2. 선형·회귀 계열

로그 실현변동성을 과거 변동성 특성의 선형식으로 설명하는 회귀 모델들이다. 이 계열은 변동성 모델 통합 비교(23번)에도 들어갔지만, 선형성 검정(RESET, BDS)이 기각되어 2026-09-07 결정으로 "선형 모델은 비선형·커널 계열로 대체"하기로 했고, HAR-RV는 2026-10-05 사용자 결정으로 같은 이유와 통일성 때문에 함께 뺐다. 그래서 이 계열 모델은 모두 결론(순위·검정·결론 문장)에서 제외된다. 그 뒤로도 23번부터 26번까지 코드는 계속 적합했고, 값은 결과 CSV에 기록으로만 남아 있다. 26c번 코드부터는 이 모델들을 아예 적합하지 않는다.

| 모델 | 특징 | 장점 | 단점 | 참고문헌 | 코드 저장소 | 사용 연구(비고) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Linear(OLS) | 표준화한 11개 지연 특성(`feats_log`)으로 로그 RV를 최소제곱 회귀한다 | 해석이 쉽고 빠르다 | 비선형·문턱 구조를 못 잡는다. 이 연구에서 선형성 기각 | 해당 없음(표준 정의) | scikit-learn `LinearRegression`(`https://github.com/scikit-learn/scikit-learn`, 확인 200) | 20(변동성 선형 회귀), 23, 24, 26 적합. **결론에서 제외**(2026-09-07 선형성 기각 결정, `test/results/assumption_diagnosis_report_20260907.md` 8절 3번). 26c 이후 코드에서 삭제 |
| Ridge | 선형 회귀에 계수 제곱합 벌점(L2)을 더해 다중공선성에 안정적으로 만든다 | 상관된 특성에서 계수가 안정적이다 | 여전히 선형이다 | Hoerl & Kennard(1970) *Technometrics* 12(1) | scikit-learn `Ridge` | 23(alpha 1 고정), 24·26(alpha 0.01~100 5점 격자, 내부검증 QLIKE로 선택). **결론에서 제외**(Linear와 같은 결정) |
| HAR-RV(로그 RV판) | 로그 RV를 1봉, 1일(96봉), 1주(672봉) 평균 RV 세 항으로 선형 회귀한다. 서로 다른 투자 기간의 거래자를 근사한다는 해석이다 | 모수가 4개라 과적합이 거의 없다 | 선형 구조, 장기기억을 세 개 항으로만 근사한다 | Corsi(2009) *J. Financial Econometrics* 7(2), 174-196 | scikit-learn `LinearRegression`에 자체 특성(`test/models/23_volatility_model_comparison_test.py`의 `feats_har`) | 23, 24, 26 적합. **결론에서 제외**(2026-10-05 사용자 결정, 형식상 선형 회귀라 선형 제외 결정과 통일. `process.md`와 `test/models/26_timebased_reeval_test.py`의 `DECISION_EXCLUDED`) |
| HAR-RV(분산 수준판) | 제곱수익률의 1일·1주·1개월(30일) 평균 세 항으로 다음 구간 평균 분산을 최소제곱 회귀한다. 로그 변환 없이 분산 수준에서 직접 적합한다 | 분산 자체를 맞추므로 QLIKE의 요구와 맞는다 | 예측값이 음수가 될 수 있어 작은 양의 하한을 둔다(18번은 학습 분산의 1%) | Corsi(2009) | 직접 구현 `test/models/18_differencing_decision_test.py`(D3b)와 `test/models/19_volatility_model_selection.py`(`forecast_har`), numpy `lstsq` | 18(D3b), 19, 19c. 이 판은 결론 제외 결정 전의 모델 선택 단계에서 쓰였다 |
| HAR-RV+(분기 추가) | 위 분산 수준판에 90일 평균 항을 더한다 | 더 긴 기억을 포착한다 | 학습 표본이 더 필요하다 | Corsi(2009)의 확장을 이 연구가 정의함, 별도 인용 없음 | 직접 구현 `test/models/19_volatility_model_selection.py`(`forecast_har`) | 19, 19c |
| 로지스틱 회귀(방향 분류) | 과거 수익률 지연값과 변동성·모멘텀·거래량 z값으로 다음 h봉 누적수익의 상승 여부를 분류한다 | 분류의 표준 모델이고 확률을 낸다 | 선형 경계 | Cox(1958) *J. Royal Statistical Society B* 20(2) | scikit-learn `LogisticRegression` | 20(방향 예측 1회성 대조, 20종목, 다수 클래스 기준선과 비교). 이후 변동성 전환으로 미사용 |

메모

- **제외 결정의 원문**(`test/results/assumption_diagnosis_report_20260907.md` 8절 3번): 선형성 검정(RESET, BDS)이 전부 기각되어 선형 모델을 쓸 근거가 없으므로 비선형·커널 계열(SVR, KernelRidge, GP)로 대체한다. 이 문서가 확인한 사실은 두 가지다. 첫째, 23번 구현이 결정을 반영하지 않고 두 모델을 계속 적합했고(`process.md` 2026-10-05 정정 항목), 26번부터 순위·동률·검정·결론에서 뺐다. 둘째, 코드의 계열 이름은 HAR-RV를 "벤치마크"로 부르지만(`FAMILY` 사전) 이 연구의 결론에서는 기준점으로 쓰지 않는다.
- **선형성 기각의 수치.** 검정은 종목별 학습 구간의 OLS 잔차에 대해 한다. RESET의 귀무가설은 선형 함수형이 맞다(적합값의 거듭제곱항 계수가 0)는 것이고 대립가설은 비선형항이 필요하다는 것이다. BDS의 귀무가설은 잔차가 독립동일분포라는 것이고 대립가설은 비선형 의존(이분산 포함)이 남아 있다는 것이다. 기각은 선형 모형이 구조를 다 담지 못한다는 뜻이다. 23번에서 두 검정은 주기 제거 전후 모두 20종목 전부에서 기각됐다(`test/results/23_model_comparison_20260908/comparison_raw.md`). 26b번 5절이 예측 구간별로 다시 보니 RESET 기각 종목이 15분 14, 30분 13, 1시간 17, 4시간 16, 12시간 6개(각각 20종목 중)였고, 비선형 항이 늘리는 증분 R²는 0.0011~0.0034로 작았다. 같은 보고서는 표본이 크면 작은 비선형도 기각되므로 효과 크기를 함께 봐야 하고, 짧은 구간에서 트리가 선형보다 크게 앞서면 그 비선형은 RESET이 보는 거듭제곱 형태가 아니라 문턱과 상호작용 형태라고 적었다. 이 후속 점검은 제외 결정을 뒤집지 않았다.
- **HAR-RV 두 판의 차이.** 18·19번은 분산 수준에서 OLS를 하고(로그 적합 뒤 지수 변환은 Jensen 보정이 예측을 10배 이상 부풀려 쓰지 않는다고 코드에 적혀 있다), 23번 이후는 로그 RV에서 적합하고 보정 상수 c로 역변환 편향을 맞춘다. 구성 항도 다르다(18·19번은 1일·1주·1개월, 23번 이후는 1봉·1일·1주). 두 판의 수치를 직접 비교하지 않는다.
- **`engine/models.py`의 `Linear`는 선형 회귀가 아니다.** 1~16번 신경망 실험의 `Linear`는 은닉층 1개(Linear, ReLU, Linear)의 MLP이므로, 이 표의 Linear(OLS)와 이름만 같다. 딥러닝 담당 문서에서 다룬다.
- Lasso, ElasticNet 등 다른 선형 모델은 해당 없음(6절).

### 3. 변동성 모형(GARCH 계열과 레짐 전환)

수익률의 조건부 분산이 과거 충격과 과거 분산에 따라 시간에 따라 변한다는 변동성 군집을 직접 모형화하는 통계 모형이다. 평균 방정식은 0(또는 상수)으로 두고 조건부 분산만 재귀식으로 추정한다. 이 연구는 GARCH(1,1)을 정규 분포와 Student-t 분포로 쓰고, 국면 전환을 담은 MS-GARCH(마르코프 전환)와 TAR-GARCH(임계값 전환)를 직접 구현해 비교에 넣었다. EGARCH, GJR-GARCH 같은 비대칭 변형은 번호가 붙은 실험에는 들어가지 않았고, 2026-09-07 가정 진단의 탐색 격자에서만 쓰였다.

| 모델 | 특징 | 장점 | 단점 | 참고문헌 | 코드 저장소 | 사용 연구(비고) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GARCH(1,1) 정규 | 조건부 분산을 상수, 직전 제곱 잔차, 직전 조건부 분산의 선형결합으로 갱신하고 표준화 잔차를 정규로 가정한다 | 모수 3개로 변동성 군집을 잡는다 | 두꺼운 꼬리를 못 담고 지속성 α+β가 1에 붙기 쉽다 | Bollerslev(1986) *J. Econometrics* 31(3), 307-327. ARCH 원형은 Engle(1982) *Econometrica* 50(4) | 직접 구현 `test/models/18_differencing_decision_test.py`의 `fit_garch11`(scipy L-BFGS-B, 시작점 3개). scipy `https://github.com/scipy/scipy` | 17b(조건부 변동성 그림 재현), 18, 18d, 19, 19c. 09-06·07 임시 탐색에서도 사용(비커밋 스크립트). 23번 이후는 쓰지 않음 |
| GARCH(1,1)-t | 위 모형에서 표준화 잔차를 단위분산 Student-t로 바꿔 자유도 ν를 함께 추정한다 | 두꺼운 꼬리를 반영한다 | 대칭 분포라 비대칭을 못 담는다. 분포 규명 보고서에서 t분포는 후보 중 5위였다 | Bollerslev(1987) *Rev. Economics and Statistics* 69(3), 542쪽부터 | 19·19c번은 직접 구현 `test/models/19_volatility_model_selection.py`의 `fit_garch11_t`(scipy L-BFGS-B). 23번 이후는 `arch` 패키지 `arch_model(mean="Zero", vol="GARCH", p=1, q=1, dist="t")`(`https://github.com/bashtage/arch`, 확인 200) | 19, 19c, 23, 24, 25, 26, 26b, 26c, 27(비교 대상으로 읽음). 23번 이후의 모든 비교표에 포함. 2026-09-06·07 임시 탐색(비커밋 스크립트)에서는 GARCH 정규와 함께 사용 |
| EGARCH | 로그 조건부 분산을 모형화해 부호 제약 없이 충격의 비대칭 반응을 허용한다 | 레버리지 효과를 직접 모형화하고 양수 제약이 필요 없다 | 암호화폐에서는 비대칭 항이 유의하지 않다는 보고가 있다(Chi & Hao 2020, arXiv:2010.07402, 초록 대조 완료) | Nelson(1991) *Econometrica* 59(2), 347쪽부터 | 확인 못 함. 2026-09-07 탐색 코드가 저장소에 없다(보고서 머리말이 분석 코드가 커밋되지 않았다고 적음). 참고로 설치된 `arch` 8.0.0은 `EGARCH`를 제공하나 이 탐색이 그것을 썼는지는 확인 못 함 | 2026-09-07 가정 진단 756조합 탐색(`test/results/assumption_diagnosis_report_20260907.md` 5절)에서만 사용. 번호 실험에는 없음 |
| GJR-GARCH | 음의 충격에 추가 계수를 두어 비대칭 반응을 허용한다 | 구조가 GARCH(1,1) 확장이라 직관적이다 | 이 데이터는 수익률 분포가 좌우 대칭에 가까워(왜도 0.004) 이점이 제한적이라고 분포 규명 보고서가 적었다 | Glosten, Jagannathan & Runkle(1993) *J. Finance* 48(5), 1779-1801 | 확인 못 함(EGARCH와 같은 이유). `arch` 8.0.0의 `GARCH(o=1)`로 구현 가능하나 사용 여부는 확인 못 함 | 2026-09-07 756조합 탐색에서만 사용. 번호 실험에는 없음 |
| APARCH, FIGARCH, HARCH, GARCH(p,q) 고차 | APARCH는 절대 충격의 거듭제곱과 비대칭을, FIGARCH는 분수 적분으로 장기기억을, HARCH는 여러 해상도의 수익률을 쓴다. 고차는 (1,1)~(3,3) | 장기기억(FIGARCH)이나 다해상도를 직접 모형화한다 | 모수가 많고 추정이 불안정할 수 있다 | Ding, Granger & Engle(1993) *J. Empirical Finance* 1(1) / Baillie, Bollerslev & Mikkelsen(1996) *J. Econometrics* 74(1) / Müller 외(1997) *J. Empirical Finance* 4(2-3) | 확인 못 함(같은 이유). 설치된 `arch` 8.0.0에 `APARCH`, `FIGARCH`, `HARCH`가 있으나 사용 여부는 확인 못 함 | 2026-09-07 탐색에서 분포 t와 skew-t로 조합. 상위 14위가 모두 FIGARCH였고 4개 가정 동시 충족은 0개였다. 번호 실험에는 없음 |
| MS-GARCH(마르코프 전환) | 두 국면 각각이 GARCH(1,1)-t이고 국면은 관측되지 않는 마르코프 체인이다. Hamilton 필터로 국면 확률을 추정하고 Gray의 축약으로 경로 폭발을 피한다 | 잔잔한 국면과 요동 국면의 서로 다른 지속성을 함께 모형화한다 | 비볼록 우도라 국소해에 빠지기 쉽고 계산이 무겁다. 일부 종목에서 모수가 허용 경계에 붙는다 | Hamilton(1989) *Econometrica* 57(2), 357쪽부터. Gray(1996) *J. Financial Economics* 42(1), 27-62. Klaassen(2002) *Empirical Economics* 27(2), 363-394. 암호화폐 적용 Ardia, Bluteau & Rüede(2019) *Finance Research Letters* 29 | 직접 구현 `engine/regime_garch.py`(`ms_filter`, `fit_ms_garch`, `fit_ms_garch_bounded`, scipy Nelder-Mead). 예비검증 코드는 `test/results/23_model_comparison_20260908/regime_switching_prelim/ms_garch.py` | 23(예비검증), 24(본편 편입), 25, 26, 26b, 26c(제약 모수화), 27(비교 대상) |
| TAR-GARCH(임계값 전환) | 직전 1시간 실현변동성이 문턱값 τ를 넘는지로 두 국면의 GARCH(1,1)-t를 즉시 선택한다. τ는 후보 분위수마다 최대우도를 적합하고 로그우도가 가장 큰 값을 고른다 | 필터링이 없어 MS-GARCH보다 계산이 단순하고 국면이 관측 가능해 해석이 쉽다 | 임계변수와 후보 분위수(0.5, 0.65, 0.8, 0.9)를 사람이 정한다. 다단계 예측은 미래 국면을 대입으로 근사한다 | Tong & Lim(1980) *J. Royal Statistical Society B* 42(3)(SETAR), Hansen(2000) *Econometrica* 68(3)(문턱 추정). Zakoian(1994) *J. Economic Dynamics and Control* 18(5)은 수익률 부호 임계라 이 구현과 임계변수가 다름 | 직접 구현 `engine/regime_garch.py`(`tar_filter`, `fit_tar_garch`). 예비검증 `.../regime_switching_prelim/tar_garch.py` | 23(예비검증), 24, 25, 26, 26b, 26c, 27(비교 대상) |

메모

- **GARCH-t의 입력과 예측 방식.** 23번 이후는 수익률에 100을 곱해 퍼센트 단위로 넣는다. 23·24번은 주기 제거 수익률의 한 봉 조건부 분산을 4배로 환산해 1시간 분산을 만들었는데, 26번 설계 표는 이를 "한 봉 분산의 4배 환산"이라는 결함으로 기록하고 GARCH 3종 모두 목표 창 전체의 다단계 기대값 재귀로 바꿨다. 학습 구간(평가 전)에서만 모수를 추정하고, 평가 구간은 고정 모수로 전진 필터링한다.
- **MS-GARCH와 TAR-GARCH의 분산 척도 오류(2026-10-05 확인).** 두 필터의 t 밀도가 σ² = h(ν-2)²/ν²로 쓰여 있어 재귀 변수 h는 조건부 분산이 아니고 실제 분산은 κ·h(κ = (ν-2)/ν)다. 24번과 25번은 h를 분산으로 써서 분산을 약 1/κ배(ν가 약 5.7일 때 1.5배) 과대 예측했다. 26번부터 κ·h로 환산한다(`engine/regime_garch.py` 서두 주석). 24·25번의 두 모델 수치는 이 오차를 포함한다.
- **MS-GARCH의 모수화가 두 가지다.** 원 모수화(`fit_ms_garch`)는 ω에 상한이 없고 ν가 2.05 근처까지 내려가며 국면 지속 확률이 0까지 갈 수 있어, 26c번에서 BOUNTY와 TOKAMAK의 내부학습 적합이 경계로 붙어 보정계수가 0.04~0.08로 무너졌다. 제약 모수화(`fit_ms_garch_bounded`, 2026-10-06)는 무조건부 분산, 지속성, 국면 유지 확률(최소 10봉), ν(3.5~30)를 허용 영역 안으로 가둔다. 제약판으로 다시 적합하니 20종목 모두 수렴했고 보정계수는 0.60~1.22 범위로 정상화됐다(`test/results/26c_recent_twopart_20261005/26c_recent_twopart_ms_refit_diagnostics.csv`에서 계산). 그래도 20종목 중 10종목은 자유도 등이 허용 경계에 닿아 있다(같은 CSV의 `전체_경계접촉`). 원본은 `*_ms_original.*`에 보존돼 있다. 26c번과 그 결과를 읽은 27번은 제약판이고, 24·25·26번은 원 모수화다.
- **설정.** MS-GARCH는 두 국면, 최대 학습 30,000개(최근 구간) 관측으로 Nelder-Mead 최적화(24·26번 재시작 3회 최대 1,400회, 26c번 제약판 초기값 2개 최대 4라운드 1,500회). TAR-GARCH는 직전 4봉(1시간) 실현변동성을 임계변수로 쓰고 τ 후보 4개마다 재시작 2회 최대 600회다. 두 모델은 시드가 필요 없어 26b·26c의 시드 반복에서는 건너뛴다(`SEED_SKIP`).
- **TAR-GARCH가 레버리지 효과를 뜻하는 것은 아니다.** 이 구현의 임계변수는 수익률 부호가 아니라 직전 실현변동성 수준이다(24번 문헌 노트 4번). Chi & Hao(2020)와 Kakinaka & Umeno(2021, arXiv:2102.02865, 제목 대조 완료)는 비트코인에서 비대칭 반응이 약하거나 부호가 뒤집힌다고 보고한다.
- **문헌과 어긋나는 보고도 있다.** Koch 외(2024, arXiv:2401.03393, 초록 대조 완료)는 비트코인 일간 조건부 분산에서 지속성이 큰 단순 GARCH가 MS-GARCH를 이길 수 있다고 보고한다. 이 연구는 15분봉 실현변동성이라 대상이 다르다.
- **쓰지 않은 대안.** 분포 규명 보고서가 제안한 NIG-GARCH, 극단값 이론(EVT)은 적합하지 않았다. 24번 문헌 노트도 이 연구가 대칭 Student-t만 썼고 왜도를 허용한 분포는 후속 후보라고 적었다.
- `arch`의 GARCH 구현은 `mean="Zero"`로 평균 방정식을 두지 않는다. 18번·19번의 직접 구현은 평균을 뺀 뒤 표준화해 쓴다.

### 4. 트리·커널·기타 기계학습

과거 수익률에서 만든 표 형태 특성을 받아 로그 RV를 맞추는 비신경 회귀 모델이다. 커널 계열은 표준화한 11개 특성(`feats_log`)으로 표준화한 로그 RV를, 트리 계열은 표준화하지 않은 16개 특성(`feats_tree`)으로 로그 RV를 맞춘다. 두 입력의 구성은 표 아래 메모에 있다. 트리 계열은 시간 순서를 모르므로 지연 값과 창 간 비율을 열로 직접 넣어 시간 구조를 준다. 마지막 두 행(정지 확률 분류기, 유사 국면 검색)은 변동성 예측과 별개의 역할을 하는 기계학습 도구다.

| 모델 | 특징 | 장점 | 단점 | 참고문헌 | 코드 저장소 | 사용 연구(비고) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| KernelRidge-RBF | RBF 커널로 입력을 고차원 특징 공간에 보내 그 안에서 릿지 회귀를 푼다 | 비선형 관계를 닫힌 형태로 풀고 구현이 단순하다 | n×n 커널 행렬이라 표본이 커지면 메모리와 시간이 폭증해 최근 표본만 쓴다 | Saunders, Gammerman & Vovk(1998) ICML는 미검증 인용 | scikit-learn `KernelRidge`(`https://github.com/scikit-learn/scikit-learn`, 확인 200) | 23, 24, 25, 26, 26b, 26c, 27(비교 대상). 제외 없음 |
| SVR(RBF) | ε 둔감 손실과 RBF 커널로 서포트 벡터 회귀를 푼다. 오차가 ε 안이면 벌점이 없다 | 이상치에 덜 민감하고 희소한 해를 준다 | 학습이 표본 수의 2~3제곱이라 최대 8,000행까지만 쓴다. 하이퍼파라미터에 민감하다 | Smola & Schölkopf(2004) *Statistics and Computing* 14(3), 199-222. libsvm: Chang & Lin(2011) *ACM TIST* 2(3) | scikit-learn `SVR`(내부는 libsvm, `https://github.com/cjlin1/libsvm`, 확인 200) | 23, 24, 25, 26, 26b, 26c, 27(비교 대상). 제외 없음 |
| Nystroem+Ridge | Nystroem 근사로 RBF 커널 특징을 대표 표본 2,048개 성분으로 만들고 그 위에서 선형 릿지를 푼다 | 커널 모델을 선형 비용으로 근사해 전체 학습 표본을 쓴다 | 근사 오차와 성분 수 선택이 필요하다. 무작위 표본에 의존해 시드를 쓴다 | Williams & Seeger(2001) NIPS 13은 미검증 인용 | scikit-learn `Nystroem`과 `Ridge`의 `make_pipeline` | 23(성분 300), 24~27(성분 2,048). 제외 없음 |
| LightGBM | 히스토그램 기반 그래디언트 부스팅 트리. 잎 단위(leaf-wise)로 성장한다 | 빠르고 표 형태 특성에 강하다. 조기종료로 반복 수를 정한다 | 시간 구조를 모르고 노이즈에 과적합하기 쉽다 | Ke 외(2017) NeurIPS 30은 미검증 인용. Friedman(2001) *Annals of Statistics* 29(5), 1189-1232 | 공식 `lightgbm` 4.7.0(`https://github.com/microsoft/LightGBM`은 301 이동 후 `https://github.com/lightgbm-org/LightGBM`에서 200 확인) | 23, 24, 25, 26, 26b, 26c, 27. 27b의 계열별 대표 모델로도 지정 |
| XGBoost | 2차 근사와 정규화를 쓰는 그래디언트 부스팅 트리. 수준(level-wise)으로 성장한다 | 정규화가 정교하고 널리 검증됐다 | 같은 입력에서는 LightGBM과 비슷한 한계를 갖는다 | Chen & Guestrin(2016) KDD, arXiv:1603.02754(초록 대조 완료, Crossref DOI 10.1145/2939672.2939785) | 공식 `xgboost`(잠금 파일에 3.2.0과 3.4.1이 함께 있음, `https://github.com/dmlc/xgboost`, 확인 200) | 23(GPU가 있으면 GPU), 24~27(CPU, 히스토그램 방식) |
| HistGBM | scikit-learn 내장 히스토그램 그래디언트 부스팅 | 별도 설치 없이 쓰고 결측을 다룬다 | 내장 조기종료가 검증 분할을 무작위로 떼어 시계열에서 낙관적이라 직접 조기종료를 구현했다 | Friedman(2001). scikit-learn: Pedregosa 외(2011), arXiv:1201.0490(초록 대조 완료) | scikit-learn `HistGradientBoostingRegressor` | 23~27 |
| 정지 확률 π 분류기 | 다음 예측 구간에서 가격이 한 칸도 움직이지 않을 확률 π를 LightGBM 이진 분류기로 예측한다 | 정지(RV가 0)를 학습에서 뺀 로그 타깃 모델이 정지를 반영하지 못하는 한계를 메운다 | 정지 표본이 적은 구간과 종목에서는 적합하지 않는다 | 두 부분 모형: Duan, Manning, Morris & Newhouse(1983) *J. Business & Economic Statistics* 1(2), 115-126(Crossref 대조 완료). 분류기는 Ke 외(2017)(미검증) | 공식 `lightgbm`의 `LGBMClassifier`(위와 같은 저장소) | 26c에서 도입, 27은 26c의 π를 공유. 그 밖의 번호 실험에는 없음 |
| 유사 국면 검색(최근접 이웃 방식) | 길이가 같은 15분봉 구간(window)을 누적수익률 곡선, 요약 통계 벡터, 컨텍스트 벡터로 표현하고 가중 거리(0.5, 0.3, 0.2)로 가까운 과거 구간 이웃 10개를 찾아 그 이웃들의 이후 결과(변동성, MDD)로 현재 국면의 이후를 예측한다. K-Means(archetype)로 후보를 줄이고 밴드 DTW로 재순위한다 | 모델 가정이 없고 "과거에 비슷했던 때" 설명이 직관적이다 | 거리 정의와 창 길이에 결과가 달라지고, 클러스터는 곡선 모양보다 변동성과 수익률 국면으로 갈린다는 점이 방법 문서에 적혀 있다 | DTW: Sakoe & Chiba(1978) *IEEE TASSP* 26(1), 43-49. 고속 근사: Salvador & Chan(2007) *Intelligent Data Analysis* 11(5). K-Means: Lloyd(1982) *IEEE Trans. Information Theory* 28(2) | 직접 구현 `marts/historical_flow.py`. 라이브러리 scikit-learn `KMeans`, scipy `cdist`, `dtaidistance`(`https://github.com/wannesm/dtaidistance`, 확인 200), `fastdtw`(`https://github.com/slaypni/fastdtw`, 확인 200) | 번호 실험과 별개 트랙. 평가는 `test/scripts/historical_flow_retrieval_eval.py`(2026-07-23, 창 16·48·96·288봉)와 `test/results/historical_flow_*_20260723.md`. 변동성 예측 비교(23~27번)에는 들어가지 않음 |

메모

- **입력 특성의 구성.** `feats_log`(11개)는 로그 RV(창 1, 4, 16, 96, 672봉) 5개, 로그 평균 절대수익률(창 4, 16, 96) 3개, 평균 수익률(창 4, 16, 96) 3개다. `feats_tree`(16개)는 RV(창 1, 4, 16, 96, 672) 5개, 창 간 비율 3개(1/16, 4/96, 16/672), 절대수익률 지연값(1, 2, 4, 8, 16) 5개, 평균 수익률 3개다(`test/models/23_volatility_model_comparison_test.py`). 모두 직전 봉까지의 정보만 쓰도록 `shift(1)`을 적용한다. 15분봉 96봉이 하루, 672봉이 1주다.
- **타깃과 후처리.** 회귀·커널·트리 모델은 모두 로그 RV를 맞추고 지수로 되돌린다. 지수 역변환은 기하평균을 내 예측이 체계적으로 낮아지므로(QLIKE는 과소예측을 크게 벌한다) 26번부터 내부검증 구간에서 QLIKE가 최소가 되는 배율 c를 구해 보정한다.
- **하이퍼파라미터.** 23번은 작은 설정이다(트리 300그루, 학습률 0.06, 깊이 6, 커널 최근 6,000행). 24번 이후는 학습률 0.02, 깊이 8, 최대 4,000그루에 조기종료 200회(내부검증 L2), LightGBM 잎 127개와 행·열 80% 추출, XGBoost `min_child_weight` 10을 쓴다(26·26c도 같음). KernelRidge는 α 3점과 γ 3점(기본 γ의 1/4배, 1배, 4배)을 내부검증 QLIKE로 고르고 최근 표본 최대 약 14,000행(메모리 1.5GB에서 역산)을 쓴다. SVR은 C 3점(1, 10, 100), ε 0.05, 최근 최대 8,000행이다. Nystroem은 Ridge α 5점(0.01~100)을 격자로 고르고, 시드마다 성분 표본을 바꾼다.
- **시드.** 26번과 26c번의 시드 반복(0~4, 26b와 27d가 읽음)은 무작위성이 있는 모델(트리, Nystroem, 정지 분류기)만 다시 학습하고, KernelRidge와 SVR은 결정적이라 건너뛴다(`SEED_SKIP`).
- **정지 확률 분류기의 실제 적합 범위**(`test/results/26c_recent_twopart_20261005/26c_recent_twopart_zero_classifier.csv`에서 계산). 분류기는 LightGBM(학습률 0.03, 잎 63개, 깊이 8, 최대 2,000그루, 조기종료 100회, 결정론 모드)이고 입력은 직전 정지 봉 비율(창 4, 16, 96, 672), 무체결 봉 비율, 트리 특성, 달력(시, 요일, 분)이다. 예측 π는 0.99로 상한을 둔다. 내부학습 정지 표본이 200 미만이거나 내부검증이 30 미만이면 적합하지 않고 π는 0이다(단일 처리와 같아진다). 20종목 중 적합한 종목 수는 15분 20, 30분 17, 1시간 12이고, 4시간과 12시간은 0이다(정지 표본이 없어 "생략(정지 표본 부족, π=0)"). 평가 AUC 중앙값은 15분 0.658, 30분 0.732, 1시간 0.772다. 최종 분산은 (1−π)·c₊·(크기 예측)²이며, c₊는 내부검증에서 RV가 0보다 큰 시점만으로 구한다. GARCH 계열과 naive에는 적용하지 않는다(정지를 두 번 반영하게 되므로).
- **해당 없음**: 이 문서 범위의 트리·커널 모델 중 랜덤포레스트, ExtraTrees, KNN(scikit-learn `KNeighbors`), 가우시안 프로세스는 코드에서 쓰인 흔적이 없다(6절).

### 5. 앙상블·결합 방식

이 연구의 결합은 "여러 모델 예측을 평균내는 앙상블"보다 "서로 다른 구조를 한 예측으로 잇는 방식"이 중심이다. 변동성 비교(23~27번)에서는 모델 간 예측 평균이나 스태킹을 쓰지 않았고(코드에서 확인 못 함), 통계 모형의 출력을 트리 입력으로 넘기는 하이브리드와 정지 확률을 곱하는 두 부분 모형, 그리고 후처리 보정을 썼다. 신경망 예측을 평균내는 앙상블은 방향 예측 시기(10, 15번)에 있었다.

| 방식 | 특징 | 장점 | 단점 | 참고문헌 | 코드 저장소 | 사용 연구(비고) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GARCH+LightGBM(하이브리드) | GARCH-t의 조건부 변동성 예측(로그)을 트리 특성 16개 뒤에 열 하나로 붙여 LightGBM이 학습한다 | 통계 모형의 지속성 구조와 비모수 학습을 함께 쓴다 | GARCH 적합이 실패하면 적합하지 않는다. 학습 구간의 GARCH 열은 구간 내 적합값이고 평가 구간은 재귀 예측이라 두 구간의 열 분포가 다를 수 있다 | Chung(2024) arXiv:2405.19849(초록 대조 완료): 학습 모델은 과소예측, GARCH는 과대예측하므로 두 유형의 하이브리드 사용을 시사 | 직접 구현 `test/models/23_volatility_model_comparison_test.py`(하이브리드 블록)와 `test/models/26c_recent_twopart_test.py`(`lgb_on`). 부품은 `arch`와 `lightgbm` | 23, 24, 25, 26, 26b, 26c, 27(비교 대상). 2026-09-07 가정 진단 8절 다음 할 일 1순위 제안에서 출발 |
| 두 부분 모형 결합 | 정지 확률 π와 크기 예측을 곱해 최종 분산을 만든다: (1−π)·c₊·크기² | 정지에서 실제값이 0이면 QLIKE의 최적 예측은 (1−π)와 비정지 조건부 기대값의 곱이라는 이론과 맞는다 | π 예측이 틀리면 전 구간에 영향이 번진다. 분류기를 모든 로그 타깃 모델이 공유한다 | Duan 외(1983)(Crossref 대조 완료). QLIKE 최적 예측은 Patton(2011) *J. Econometrics* 160(1), 246-256 | 직접 구현 `test/models/26c_recent_twopart_test.py`(`Scorer.score_log`) | 26c, 27. 4·12시간은 π=0이라 단일 처리와 같음(4절 메모) |
| 보정 배율 c(후처리) | 내부검증 정시 시점에서 QLIKE를 최소로 하는 분산 배율을 구해 모든 모델의 예측에 곱한다 | 로그 역변환의 체계적 과소예측을 모델 공통으로 맞춘다 | 보정 전 값을 함께 기록해야 모델 자체의 질과 구분된다 | Patton(2011) | 직접 구현 `test/models/26_timebased_reeval_test.py`(`Scorer.calib`, `qlike_scale`) | 26, 26b, 26c, 27. 보정 전 QLIKE도 결과 CSV에 같이 저장 |
| insanity filter(후처리) | 예측값이 학습 구간 타깃의 [최솟값, 최댓값] 밖이면 학습 평균으로 대체한다(26c 이후. 24·25번은 0.5%~99.5% 분위수 밖을 대체하는 방식) | QLIKE가 예측 분산이 0에 붙을 때 발산하는 것을 막는다(Patton의 가정 A4) | 대체 비율(트리밍 비율)을 함께 보고해야 한다 | Bollerslev, Patton & Quaedvlieg(2016) *J. Econometrics* 192(1), 1-18. 범위는 Zhang 외(2024)를 따랐다고 코드 주석에 적혀 있으나 이 문서에서는 미검증 인용 | 직접 구현 `test/models/23_volatility_model_comparison_test.py`(`insanity_filter`). 26번과 26c번 파일에도 같은 이름의 함수가 따로 있다 | 23, 24, 26, 26c, 27 |
| 시드 평균·상위 K 결합(신경망) | 여러 시드와 구성의 예측을 단순 평균, 검증 점수 역수 가중 평균, 중앙값으로 묶는다. 검증 구간으로 conformal 예측 구간도 붙인다 | 시드 잡음을 줄이고 점예측을 안정화한다 | 평균이 신호를 올리지 못하면 한계가 모델 축에 있다는 뜻이다 | 확인 못 함(이 연구가 만든 구성, 인용 없음) | 직접 구현 `test/models/10_objective_ensemble_confirmation_test.py`(`ensemble_predictions`), `test/models/15_trend_capture_defense_test.py`(T6, 시드 42, 7, 123, 2026) | 10, 15. 방향 예측 신경망의 앙상블이라 딥러닝 문서와 겹침 |
| 점예측+위험 gate 융합(신경망) | 점예측이 매수 신호이고 위험 확률이 검증 분위수 문턱 이하일 때만 보유하는 AND 규칙이다 | 위험 구간을 거르는 규칙이 단순하다 | 문턱을 검증 분포로 정해 평가 구간에서 어긋날 수 있다 | 확인 못 함(이 연구가 만든 정책, 인용 없음) | 직접 구현 `engine/fusion.py`(`evaluate_fusion_policy`) | 12, 14. 가격 방향 정책이라 변동성 예측 비교와 별개 |

메모

- **23~27번에서 쓰지 않은 결합**: 서로 다른 모델 예측의 평균, 가중 결합, 스태킹은 해당 없음(코드에서 흔적을 찾지 못했다). 27d와 27번 보고서의 "시드 평균"은 시드마다 구한 손실을 평균한 것이고 예측을 평균한 앙상블이 아니다(`test/models/27_model_expansion_test.py`의 보고서 문구가 이를 명시한다). 모델 군을 순차(재귀형)와 병렬로 묶는 27d의 분류는 결과 해석용 묶음이지 결합 모델이 아니다.
- **하이브리드의 정보 시점.** 26c번은 내부학습 구간에서는 내부학습 구간까지 적합한 GARCH를, 최종 평가 예측에서는 학습 구간 전체 GARCH를 특성으로 쓴다(미래 정보가 들어가지 않게 하려는 설계). 23번은 학습 구간에서 GARCH의 구간 내 조건부 변동성을, 평가 구간에서 재귀 예측을 채운다.

### 6. 코드에서 쓰지 않은 모델(해당 없음)

아래 모델은 이 문서의 범위에 들어가지만 `test/models/*.py`, `engine/`, `marts/`, `pipelines/`와 노트북(`.ipynb`)을 클래스 이름으로 검색해 보니 적합하거나 호출한 흔적이 없었다. 문헌 검토 문서(`test/research_materials/forecasting_methodology_literature_review_20260613.md`)에 기준선 후보로 언급된 것도 있으나 구현되지 않았다.

| 모델 | 상태 | 근거 |
| :--- | :--- | :--- |
| 랜덤포레스트(RandomForest) | 해당 없음 | 클래스 이름 검색 결과 없음. 문헌 검토 문서의 기준선 후보로만 언급 |
| ExtraTrees | 해당 없음 | 같음 |
| KNN(scikit-learn `KNeighbors`, `NearestNeighbors`) | 해당 없음 | 같음. 최근접 이웃 성격의 도구는 4절의 유사 국면 검색(`marts/historical_flow.py`)뿐이고 `cdist`와 `argpartition`으로 직접 구현했다 |
| Lasso, ElasticNet | 해당 없음 | 같음. 선형 회귀 계열은 Linear, Ridge, HAR-RV뿐이다(2절, 로지스틱 회귀는 분류용) |
| ARIMA, ARFIMA | 해당 없음 | 18번 코드가 "HAR-RV·FIGARCH·ARFIMA on log-RV로 두어야 한다"는 메모만 출력한다. 적합 코드 없음 |
| 가우시안 프로세스 | 해당 없음 | 2026-09-07 보고서 8절 3번이 대체 후보로 적었으나 구현 없음 |
| CatBoost, AdaBoost, 일반 GradientBoosting | 해당 없음 | 클래스 이름 검색 결과 없음(`HistGradientBoostingRegressor`만 있음) |
| NIG-GARCH, 왜도 분포 GARCH, EVT | 해당 없음 | 분포 규명 보고서가 방향으로 제안했으나 적합하지 않았다. skew-t는 2026-09-07 탐색에서만 조합(3절) |
| SMA 교차 같은 이동평균 예측기 | 해당 없음 | 롤링분산과 EWMA만 쓰였다(1절) |
| Linear, Ridge, HAR-RV(결론 반영) | 제외 | 2절. 적합은 했으나 순위·검정·결론에서 제외 |
| DLinear·NLinear | 제외 | 선형 예측기라 2026-10-05 결정으로 27번에서 제외(`test/research_materials/model_catalog.md` 4절). 딥러닝 담당 문서 범위 |

### 7. 확인 현황

#### 7-1. 저장소 주소 확인

`curl -s -o /dev/null -w "%{http_code}"`로 확인했다(2026-10-14). 확인 9개, 확인 못 함 0개다. 단 LightGBM은 처음 주소가 301로 이동했고, 이동 대상에서 200을 확인했다.

| 주소 | 결과 |
| :--- | :--- |
| `https://github.com/scikit-learn/scikit-learn` | 200 |
| `https://github.com/cjlin1/libsvm` | 200 |
| `https://github.com/microsoft/LightGBM` | 301, 이동 대상 `https://github.com/lightgbm-org/LightGBM`에서 200 |
| `https://github.com/dmlc/xgboost` | 200 |
| `https://github.com/bashtage/arch` | 200 |
| `https://github.com/scipy/scipy` | 200 |
| `https://github.com/pandas-dev/pandas` | 200 |
| `https://github.com/wannesm/dtaidistance` | 200 |
| `https://github.com/slaypni/fastdtw` | 200 |

직접 구현한 모델의 저장소 안 경로는 표에 적었고 파일 존재를 확인했다. 코드를 확인하지 못한 항목은 EGARCH, GJR-GARCH, APARCH·FIGARCH·HARCH 묶음 3행이다(2026-09-07 탐색 코드가 저장소에 없다). 2026-09-06·07 임시 탐색(BTC, naive·Linear·Ridge·HistGBM·XGBoost·LightGBM·LSTM·GRU·GARCH 정규·GARCH-t)도 스크립트가 커밋되지 않아 코드를 확인하지 못했고 보고서(`test/results/scratch_volatility_report_20260907.md`)로만 확인했다.

#### 7-2. 참고문헌 대조

arXiv 논문은 `test/scripts/mcp_client.py`의 `get_abstract`로, 저널 논문은 Crossref API(`api.crossref.org`)로 제목, 저자, 연도, 저널, 권호, 쪽수를 대조했다. 쪽수 끝까지 확인하지 못한 세 편(Bollerslev 1987, Nelson 1991, Hamilton 1989)은 본문에 "쪽부터"로만 적었다.

| 구분 | 항목 |
| :--- | :--- |
| 초록 대조 완료(arXiv) | 1603.02754(XGBoost), 1201.0490(scikit-learn), 2010.07402(Chi & Hao), 2401.03393(Koch 외), 2405.19849(Chung) |
| 제목과 저자만 대조(arXiv) | 2102.02865(Kakinaka & Umeno) |
| Crossref 대조 완료 | Hyndman & Koehler 2006, Bollerslev 1986, Bollerslev 1987, Engle 1982, Nelson 1991, Glosten 외 1993, Ding 외 1993, Baillie 외 1996, Müller 외 1997, Hamilton 1989, Gray 1996, Klaassen 2002, Ardia 외 2019, Hansen 2000, Tong & Lim 1980, Zakoian 1994, Hoerl & Kennard 1970, Friedman 2001, Smola & Schölkopf 2004, Chang & Lin 2011, Duan 외 1983, Patton 2011, Bollerslev 외 2016, Corsi 2009, Cox 1958, Sakoe & Chiba 1978, Salvador & Chan 2007, Lloyd 1982 |
| **미검증 인용** | Saunders, Gammerman & Vovk(1998) ICML(KernelRidge). Williams & Seeger(2001) NIPS 13(Nystroem). Ke 외(2017) NeurIPS 30(LightGBM). RiskMetrics 기술 문서(J.P. Morgan/Reuters 1996)(EWMA). Zhang 외(2024) *J. Financial Econometrics* 22(2)(insanity filter 범위, 코드 주석에서 옮겼고 이 문서에서는 대조하지 못함) |

미검증 인용 중 Ke 외(2017), Williams & Seeger(2001), Saunders 외(1998), Zhang 외(2024)는 Crossref 검색에서 해당 논문이 나오지 않았다(Saunders 외는 같은 저자의 2000년 다른 논문만 나왔다). RiskMetrics 기술 문서는 논문이 아니라 Crossref 대조 대상으로 삼지 않았다. 이 항목의 서지 정보는 확인하지 못했으므로 서지 세부를 문서에 적지 않고 저자와 연도, 학회 이름만 남겼다.

---

## 제2부. 딥러닝과 시계열 파운데이션 모델

### 읽는 법

#### 구현 구분 표기

연구에서 같은 이름의 모델이라도 구현이 서로 다르므로 비고 칸에 아래 네 가지로 구분해 적는다.

| 표기 | 뜻 |
| :--- | :--- |
| 공식 구현 | 논문 저자 또는 해당 기관이 공개한 코드·체크포인트를 그대로 불러 쓴 경우이다. |
| 라이브러리 구현 | neuralforecast 같은 유지보수 중인 외부 패키지가 원 논문 구조를 구현해 제공한 것을 쓴 경우이다. |
| PyTorch 내장 모듈 위 자체 구성 | `torch.nn.LSTM`처럼 PyTorch가 제공하는 표준 층 위에 입출력 헤드만 직접 붙인 경우이다. |
| 자체 근사(~Like, ~Lite, 장난감) | 모델 이름만 따고 구조는 크게 단순화한 직접 구현이다. 원 논문의 핵심 설계가 빠져 있어 그 모델의 성능으로 읽으면 안 된다. |

#### 인용 검증 표기

논문은 저장소의 arXiv 도구(`test/scripts/mcp_client.py arxiv get_abstract`)로 제목·저자를 대조했다. 대조에 성공한 것은 "arXiv 대조 완료"로 적고, 대조하지 못한 것은 "미검증 인용"으로 적는다. 학회·학술지 게재 정보(ICLR, NeurIPS 등)는 arXiv 메타데이터에 없어 대조하지 못했으며, 기존 카탈로그 표기를 따르거나 아예 적지 않았다. 이는 게재처 미검증이다.

#### 저장소 주소 확인 표기

저장소 주소는 `curl`로 HTTP 응답 코드를 확인해 200이면 "확인(200)"으로 적었다. 확인하지 못한 것은 "확인 못 함"으로 적었다. 이 환경에는 `gh` CLI가 없다.

#### 공통 평가 틀(이 문서에서 "27번 틀"이라 부르는 것)

27번에서 신규 모델은 모두 "한 시계열의 과거에서 다음 한 값"을 푼다. 입력은 예측 구간 길이 H분 단위로 자정에 맞춰 겹치지 않게 자른 블록의 로그 실현변동성 시계열이며, 값이 없는 블록은 직전 값을 유지하고 손실에서 가린다. neuralforecast 6종과 ModernTCN, S-Mamba는 이 블록 시계열을 학습하고 조기종료·전체 재적합을 거치며, 파운데이션 7종은 재학습 없이(zero-shot) 직전 512블록(Lag-Llama는 1,124블록)만 읽는다. 반대로 GRU·LSTM(26c번 이후)은 15분봉 96개의 (수익률, 수익률 절댓값)을 읽는다. 이 입력 차이는 27c번에서 통제를 시도했다.

---

### 1. 순환(RNN) 계열

순환 계열은 창 안의 시점을 순서대로 읽으며 은닉 상태를 이어받는 모델이다. GRU와 LSTM은 23번 이후 변동성 예측의 딥러닝 대표 모델이며, 26c번과 27번 보고서에서 4시간·12시간 구간 상위권에 들었다. DeepAR와 ODE-RNN은 1~3번 방향 예측 시기에 이름만 딴 장난감 구현으로 잠깐 쓰였고 이후 쓰이지 않았다.

| 모델 | 처리 방식·특징 | 장점 | 단점 | 참고문헌 | 코드 저장소 | 사용 연구(비고) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LSTM | 입력 게이트, 망각 게이트, 출력 게이트와 별도의 셀 상태로 긴 의존성을 이어받는다. 창 안의 시점을 차례로 읽고 마지막 시점의 은닉 상태를 선형 헤드에 넣어 값 하나를 낸다. | 기울기 소실을 완화해 긴 창을 읽을 수 있다. 표본이 많은 15분봉 입력에서 4시간·12시간 구간에 최선과 사실상 동률(A등급)이었다. | 시점을 순서대로 처리해 병렬화가 어렵다. 학습률에 민감해 26c번은 학습률 후보 세 개를 비교해야 했다. 15분 구간은 최선 대비 격차가 0.139(C등급)로 컸다. | Hochreiter & Schmidhuber(1997), *Long Short-Term Memory*, *Neural Computation* 9(8). **미검증 인용**(arXiv에 없는 1997년 논문이라 도구로 대조하지 못함). | 저자 공식 저장소는 확인 못 함(1997년 논문이라 공개 저장소가 없는 것으로 보이나 조사하지 않음). 표준 구현은 PyTorch `torch.nn.LSTM`(https://github.com/pytorch/pytorch, 확인(200)). 이 연구가 쓴 구현은 `torch.nn.LSTM` 위에 자체 헤드를 붙인 것이다. | 사용 연구는 1번(2층·은닉 64), 2번·3번(1층·은닉 64), 4번(`LSTMRepresentative`), 5번(`RecurrentProbe`), 8번, 23번, 24번, 26번, 26c번(`engine/models.py`의 `RecurrentForecaster`, 은닉 128, 입력 96×2), 27c번(27번과 같은 블록 입력을 준 대조 실험)이다. 25번·26b번·27b번·27d번·27e번은 저장된 예측만 분석했고 재학습하지 않았다. 구현 구분은 PyTorch 내장 모듈 위 자체 구성이다. **투자 활용 참고(외부 자료, 논문 아님)**: Qlib 시작 가이드(https://github.com/gameworkerkim/vibe-investing/blob/main/TechDoc/Quant_Qlib/Qlib-getting-started-KR.md)의 모델 동물원 절이 이 모델을 주식 종목 예측용 비교 모델로 나열한다. |
| GRU | 리셋 게이트와 업데이트 게이트 두 개만 쓰는 경량 순환 구조이다. 셀 상태를 따로 두지 않아 LSTM보다 파라미터가 적다. 이 연구에서는 LSTM과 같은 껍데기(1층, 마지막 시점 은닉 상태, 선형 헤드)에 층만 바꿔 쓴다. | LSTM보다 가볍고 비슷한 성능을 낸다. 27번 보고서 기준 4시간·12시간 구간에서 최선과 사실상 동률(A등급)이었고 1시간은 B등급이었다. 27d번은 요동 국면(Q5)의 1시간·4시간·12시간 최선이 순환 신경망이라고 보고했고, 27e번이 입력을 맞춰 다시 점검했다. | LSTM과 같은 순차 처리의 한계가 있다. 15분 구간 격차가 0.146(C등급)으로 트리·커널보다 컸다. 27번 입력(블록 로그 RV)과 26c번 입력(15분봉)이 달라 순차 구조의 효과만 따로 말하기 어렵다(27c번이 통제를 시도). | Cho 외(2014), *Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation*, arXiv:1406.1078(arXiv 대조 완료). 게재처 EMNLP 2014는 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 저장소는 확인 못 함. 표준 구현은 PyTorch `torch.nn.GRU`(위와 같은 저장소, 확인(200)). | 사용 연구는 LSTM과 같다(1번, 2번, 3번, 5번, 8번, 23번, 24번, 26번, 26c번, 27c번). 구현 구분은 PyTorch 내장 모듈 위 자체 구성이다. **투자 활용 참고(외부 자료, 논문 아님)**: Qlib 시작 가이드(https://github.com/gameworkerkim/vibe-investing/blob/main/TechDoc/Quant_Qlib/Qlib-getting-started-KR.md)의 모델 동물원 절이 이 모델을 주식 종목 예측용 비교 모델로 나열한다. |
| DeepAR(장난감) | 원 설계는 LSTM이 매 시점 확률분포의 모수를 내고, 학습할 때는 정답을 입력하고(teacher forcing) 추론할 때는 뽑은 표본을 다시 입력해 미래 경로를 만든다. 이 연구의 `DeepARModel`은 1층 LSTM과 선형층뿐이라 `LSTMModel`과 코드가 같고, 확률분포 출력도 표본 생성도 없다. | 원 설계는 다중 시계열을 하나의 모델로 학습하고 확률 예측을 바로 낸다. | 이 연구의 근사본은 LSTM과 동일해 DeepAR의 장점이 없다. 원 설계도 모수형 분포를 가정해야 해서 변동성의 두꺼운 꼬리에 맞는 분포를 골라야 한다. | Salinas 외(2020), *DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks*, arXiv:1704.04110(arXiv 대조 완료). 게재처(*International Journal of Forecasting*)는 **게재처 미검증**. | 공식 구현은 AWS의 GluonTS(https://github.com/awslabs/gluonts, 확인(200)). neuralforecast에도 `DeepAR`가 있다(설치본 `neuralforecast/models/deepar.py`에서 확인). 이 연구가 실제 쓴 것은 자체 근사이다. | 사용 연구는 2번·3번의 `ModelZoo.DeepARModel`뿐이다. 16번은 DeepAR식 Student-t 음의 로그우도를 손실로만 빌렸고 모델은 쓰지 않았다. 구현 구분은 자체 근사이며 이후 연구에서 미사용이다. 재도입한다면 neuralforecast의 `DeepAR`를 써야 한다(기존 카탈로그 9절 결정). |
| ODE-RNN(장난감) | 원 설계는 관측 사이 구간에서 은닉 상태가 신경 미분방정식을 따라 연속적으로 변하고 적응형 ODE 솔버로 적분한다. 이 연구의 `ODERNNModel`은 `GRUCell`을 쓴 뒤 작은 MLP를 0.1 스텝 오일러 방식으로 한 번 더하는 것이 전부이다. | 원 설계는 불규칙하게 관측된 시계열을 자연스럽게 다룬다. | 이 연구의 근사본은 적응형 솔버가 없고 시점마다 반복문을 돌아 느리다. 원 설계의 이점은 불규칙 샘플링에 있는데, 이 연구는 이후 15분 시간 격자를 복원해 규칙 간격 입력으로 바꿨다(27번 보고서 변환 표). | Rubanova, Chen & Duvenaud(2019), *Latent ODEs for Irregularly-Sampled Time Series*, arXiv:1907.03907(arXiv 대조 완료). 게재처(NeurIPS 2019)는 기존 카탈로그 표기이며 **게재처 미검증**. | 공식 구현은 https://github.com/YuliaRubanova/latent_ode(확인(200)). 이 연구는 사용하지 않았다. | 사용 연구는 1번(간소화 컨셉이라고 코드 주석에 적음)과 2번·3번이다. 구현 구분은 자체 근사이며 이후 미사용이다. |

### 2. 합성곱·MLP·선형 계열

이 계열은 어텐션이나 순환 없이 합성곱, 완전연결층, 선형층만으로 창 전체를 한꺼번에 처리한다(병렬 처리). TCN, TimesNet, ModernTCN은 27번에서 neuralforecast 또는 저자 공식 코드로 처음 제대로 비교했고, 2~16번의 같은 이름 모델은 대부분 자체 근사이다. DLinear와 NLinear는 2026-10-05 결정으로 27번에서 제외됐고, N-BEATS와 MLP는 변동성 연구(23번 이후)에서 쓰이지 않았다.

| 모델 | 처리 방식·특징 | 장점 | 단점 | 참고문헌 | 코드 저장소 | 사용 연구(비고) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TCN | 팽창(dilated) 합성곱을 층마다 팽창률을 키워 쌓아 수용영역을 지수적으로 늘린다. 순환 없이 창 전체를 병렬로 처리한다. neuralforecast판은 커널 2, 팽창률 1·2·4·8·16, 인코더 은닉 128, MLP 디코더 구조이다(설치본 소스에서 확인). | 순차 병목이 없고 긴 수용영역을 얻는다. 원 논문은 단순한 합성곱 구조가 LSTM 같은 순환망을 여러 과제에서 앞선다고 보고한다. | 27번에서 4시간·12시간 구간 격차가 0.176·0.291(C등급)로 신규 모델 중 가장 컸고, 12시간 붕괴는 시드 5개 모두에서 재현됐다. process.md는 12시간 블록 표본 부족(약 1,500개)으로 해석한다. 자체 구현은 팽창 합성곱에 양쪽 대칭 패딩을 쓴 뒤 끝을 잘라 엄밀한 왼쪽 패딩 인과 합성곱은 아니다(코드에서 확인, 마지막 위치의 출력만 쓰므로 정답 누설은 없다). | Bai, Kolter & Koltun(2018), *An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling*, arXiv:1803.01271(arXiv 대조 완료). | 저자 공식 저장소는 https://github.com/locuslab/TCN(확인(200)). 이 연구는 27번에서 neuralforecast(https://github.com/Nixtla/neuralforecast, 확인(200), 설치본 3.2.2)의 `TCN`을 썼다. | 사용 연구는 2번·3번(장난감, 팽창 합성곱 1층), 5번(`TCNProbe`, 3층), 8번(파일 안 사본)·15번(`engine/models.py`의 `TCNForecaster`, 15번 기본 후보)인 자체 구현, 13번(코드 삭제, 복구 결과에 TCN 언급), 27번(neuralforecast 공식 라이브러리 구현, 시드 5개)이다. 기존 카탈로그가 `TCNForecaster`를 "미사용으로 남아 있던 것"이라고 적은 것은 15번 사용 사실과 맞지 않는다. **투자 활용 참고(외부 자료, 논문 아님)**: Qlib 시작 가이드(https://github.com/gameworkerkim/vibe-investing/blob/main/TechDoc/Quant_Qlib/Qlib-getting-started-KR.md)의 모델 동물원 절이 이 모델을 주식 종목 예측용 비교 모델로 나열한다. |
| ModernTCN | 입력을 패치로 나눈 뒤 큰 커널(51)과 작은 커널(5)을 병렬로 두는 재매개변수화 깊이별 합성곱, 그리고 합성곱 FFN 블록을 쌓는 순수 합성곱 구조이다(`ReparamLargeKernelConv`, `Block`을 소스에서 확인). 저자는 큰 수용영역과 변수 간 의존성 포착을 장점으로 든다(README). | 어텐션 없이 큰 수용영역을 얻어 학습이 가볍다. 27번에서 4시간 구간이 시드 5개 평균 격차 0.0103으로 합성곱 중 가장 좋았다. | 12시간에서 격차 0.192(C등급)로 붕괴해 TCN·TimesNet과 같은 패턴을 보였다. 시간 구간에 따라 순위 변동이 크다. 4시간 구간에서 최선과 구분되지 않는 모델 집합(MCS)에 드는 시드가 5개 중 4개뿐이었다(process.md의 시드 의존 칸). 저자 코드는 pip 패키지가 없어 `third_party`로 가져와 어댑터 함수를 직접 써야 했다. | Luo & Wang(2024), *ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis*, ICLR 2024 Spotlight. **미검증 인용**(arXiv에 논문이 검색되지 않아 도구 대조 불가, 저자·제목·게재처는 저장소 README 표기와만 일치). | 저자 공식 저장소는 https://github.com/luodhhh/ModernTCN(확인(200)). 이 연구는 `third_party/ModernTCN`(커밋 56a9a2c, `ModernTCN-Long-term-forecasting`)을 그대로 불러 썼다. | 사용 연구는 8번·15번(`ModernTCNLike`, 자체 근사: 1×1 합성곱, 커널 7 깊이별 합성곱 둘, 패치 없음)과 27번(저자 공식 코드, 저자 ETTh1 스크립트 설정, 변수 1개, 시드 5개)이다. 27번 어댑터는 별도 파일 `27_adapters.py`가 아니라 `27_model_expansion_test.py`의 `modern_tcn_predict` 함수이다(`27_adapters.py`는 저장소에 없음). |
| TimesNet | 1차원 시계열을 고속 푸리에 변환으로 찾은 주요 주기에 맞춰 2차원(주기 내 위치 x 주기 간 위치)으로 접고, 2D 합성곱으로 주기 내·주기 간 변동을 함께 포착한다. | 주기 구조가 뚜렷한 데이터에 강하고 예측·보간·분류 등 여러 과제를 한 구조로 처리한다. | 27번에서 전 구간 C등급(격차 0.081~0.195)이었다. 27번 입력은 (요일, 시) 주기를 제거한 수익률로 만든 로그 변동성이라 TimesNet이 찾을 주기가 줄었을 수 있으나 이 해석은 검증하지 않았다. 자체 근사 `TimesNetLike`는 푸리에 변환과 2D 변환이 없어 원 구조가 아니다. | Wu 외(2022), *TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis*, arXiv:2210.02186(arXiv 대조 완료). 게재처 ICLR 2023은 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 구현은 https://github.com/thuml/Time-Series-Library(확인(200)). 이 연구는 27번에서 neuralforecast `TimesNet`을 썼다. | 사용 연구는 8번·9번·10번(`TimesNetLike`, 자체 근사: 커널 3·7 1D 합성곱 둘의 시간 평균을 이어 붙여 MLP에 넣음, 10번에서는 실패 통제 모델)과 27번(neuralforecast 라이브러리 구현, 시드 5개)이다. |
| N-BEATS(장난감) | 원 설계는 역방향 잔차(backcast)와 순방향 예측(forecast)을 이중으로 쌓은 완전연결 블록 스택이고, 기저 확장으로 추세·계절을 분해한다. 이 연구의 `NBeatsModel`은 `Linear, ReLU, Linear` 2층 MLP뿐이라 잔차 스택도 기저 확장도 없다. | 원 설계는 순수 완전연결층만으로 단변량 점예측에서 높은 정확도를 보고한다. | 이 연구의 근사본은 단순 MLP라 N-BEATS의 성능을 대표하지 못한다. 27번 로스터에 넣지 않았고, 명시된 제외 사유는 확인 못 함(27번 설계 셀에는 S-Mamba, xLSTM, TimeGPT, DLinear, NLinear만 제외로 적혀 있음). | Oreshkin 외(2020), *N-BEATS: Neural basis expansion analysis for interpretable time series forecasting*, arXiv:1905.10437(arXiv 대조 완료). 게재처 ICLR 2020은 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 저장소는 https://github.com/ServiceNow/N-BEATS(확인(200), `ElementAI/N-BEATS` 주소는 이 주소로 이동됨). neuralforecast에도 `NBEATS`가 있다(설치본 소스에서 확인). 이 연구가 실제 쓴 것은 자체 근사이다. | 사용 연구는 2번·3번(`ModelZoo.NBeatsModel`)뿐이다. 자체 근사이며 이후 미사용이다. |
| DLinear | 입력을 이동평균으로 추세와 나머지(계절)로 분해하고 각각에 시간축 선형층을 하나씩 적용해 합한다. 원 논문의 핵심 주장은 복잡한 Transformer가 이런 단순 선형 모델을 못 이긴다는 것이다. | 파라미터가 적고 학습이 빠르며 강한 기준선이다. | 선형 예측기라 이 연구의 2026-09-07 선형성 기각 결정과 충돌해 27번에서 제외됐다. 원 결정문은 process.md의 "2026-10-05 결정: 27번에서 DLinear·NLinear도 제외(선형성 기각 결정과 HAR-RV 제외에 맞춤)"이다. | Zeng 외(2022), *Are Transformers Effective for Time Series Forecasting?*, arXiv:2205.13504(arXiv 대조 완료). 게재처 AAAI 2023은 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 구현은 https://github.com/cure-lab/LTSF-Linear(확인(200)). neuralforecast에도 `DLinear`가 있다(설치본 소스에서 확인). 이 연구가 실제 쓴 것은 자체 근사이다. | 사용 연구는 8번·15번(`DLinearLike`, 자체 근사: 이동평균 커널 5로 분해, 두 선형층과 특징 헤드를 더함)이다. 2번·3번의 `LinearDecompModel`은 이동평균 없이 입력을 두 선형층에 그대로 통과시켜 합한 것으로 DLinear의 전신이라 불리지만 실제 분해는 없다. 27번에서는 제외됐다. 제외 근거 문서는 `process.md`(2026-10-05 결정)이다. |
| NLinear | 입력 창의 마지막 값을 모든 값에서 빼서 정규화한 뒤 선형층 하나로 예측하고 마지막 값을 다시 더한다. 분포가 시간에 따라 이동하는 데이터에 강건하게 설계됐다. | DLinear와 같이 단순하고 빠르다. | 선형 예측기라 27번에서 제외됐다. 자체 근사 `NLinearLike`는 마지막 값을 다시 더하는 단계가 없고 특징 헤드만 있다(코드에서 확인). | DLinear와 같은 논문이다(arXiv:2205.13504, arXiv 대조 완료). | DLinear와 같다(공식 https://github.com/cure-lab/LTSF-Linear, neuralforecast `NLinear`). | 사용 연구는 8번·15번(`NLinearLike`, 자체 근사)이다. 27번에서 제외됐고 근거는 DLinear 행과 같다. |
| MLP(코드 이름 `Linear`) | 이름은 `Linear`이지만 `engine/models.py`의 `LinearForecaster`는 창을 펼쳐 `Linear`, `ReLU`, `Linear`를 통과시키는 은닉 1층 MLP이다. 2번·3번의 `ModelZoo.LinearModel`은 층 하나뿐인 진짜 선형 회귀이다. | 구조가 가장 단순한 통제군이다. | 방향 예측 시기(8~14번)에 출력 분산 폭주가 보고됐다(14번 보고서가 Linear 모델 고유 문제로 정리). 변동성 연구(23번 이후)에서는 쓰이지 않았다. | 특정 논문이 없는 표준 구조이다. 인용 해당 없음(원 논문 없음). | 표준 구현은 PyTorch 층(https://github.com/pytorch/pytorch, 확인(200)). neuralforecast에도 `MLP`가 있다(설치본 소스에서 확인). 이 연구가 쓴 것은 PyTorch 층 위 자체 구성이다. | 사용 연구는 2번·3번(진짜 선형), 5번(`LinearProbe`), 8번~12번·14번(`LinearForecaster`), 15번(선택 후보)이다. 27번은 이 모델을 쓰지 않았다(해당 없음, 27번 로스터에 MLP가 없음). 같은 이름 계열의 선형 모델은 2026-09-07 선형성 기각 결정에 따라 26번부터 순위·검정·결론에서 제외됐다(process.md). |

### 3. 어텐션(Transformer) 계열

어텐션 계열은 창 안의 모든 위치(또는 패치, 변수)가 서로를 한 번에 참조하며 처리한다. 이 연구에서 가장 혼란이 큰 계열이다. 2~16번의 `~Like`와 장난감 모델은 구조가 크게 단순화됐고, 24번에서 `PatchTSTLike`와 `ITransformerLike`가 일반화에 실패해 제외됐다. 27번에서 neuralforecast 공식 패키지판(PatchTST, iTransformer, Autoformer, TimeXer)을 새로 비교했으며, 신규 어텐션 중 가장 나은 PatchTST도 15분~1시간에서 최선과 격차가 컸다.

| 모델 | 처리 방식·특징 | 장점 | 단점 | 참고문헌 | 코드 저장소 | 사용 연구(비고) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Transformer | 입력 시점마다 토큰을 만들고 셀프어텐션으로 모든 위치 사이의 관계를 한 번에 계산한다. 이 연구의 `TransformerForecaster`는 `nn.TransformerEncoder` 2층에 학습형 위치 임베딩을 더하고, 마지막 위치의 출력을 선형 헤드에 넣는다(인코더만 사용하며 원 논문의 인코더-디코더 전체는 아님). | 먼 시점 사이의 관계를 직접 연결하고 병렬로 학습한다. | 연산이 창 길이의 제곱에 비례한다. 시계열에서는 순서 정보가 위치 임베딩에만 의존한다. 선형 모델이 시계열 Transformer를 이긴다는 비판이 있다(DLinear 행의 논문). 24번에서 같은 계열의 근사 모델이 일반화에 실패했다. | Vaswani 외(2017), *Attention Is All You Need*, arXiv:1706.03762(arXiv 대조 완료). 게재처 NeurIPS 2017은 기존 카탈로그 표기이며 **게재처 미검증**. | 논문 원 코드는 `tensor2tensor`로 알려져 있으나(https://github.com/tensorflow/tensor2tensor, 확인(200)) 논문 본문과 대조하지 못했다. 이 연구가 쓴 것은 PyTorch `nn.TransformerEncoder` 위 자체 구성(https://github.com/pytorch/pytorch)이다. neuralforecast에는 `VanillaTransformer`가 있다(설치본에서 확인). | 사용 연구는 1번(`SimpleTransformer`), 2번·3번(`ModelZoo.TransformerModel`), 4번(`TransformerRepresentative`), 5번(`TransformerProbe`), 8번 이후 `engine/models.py`의 `TransformerForecaster`(13번 복구 결과에도 언급)이다. 23번 이후 변동성 연구에는 없다. 27번 로스터에는 넣지 않았고 기존 카탈로그는 "GRU·LSTM·PatchTST가 이미 순환·어텐션 계열을 대표해 중복 비교 가치가 낮다"는 판단을 사유로 적는다. **투자 활용 참고(외부 자료, 논문 아님)**: Qlib 시작 가이드(https://github.com/gameworkerkim/vibe-investing/blob/main/TechDoc/Quant_Qlib/Qlib-getting-started-KR.md)의 모델 동물원 절이 이 모델을 주식 종목 예측용 비교 모델로 나열한다. |
| Informer | 원 설계는 ProbSparse 셀프어텐션으로 어텐션 비용을 줄이고 증류(distilling)로 입력 길이를 층마다 줄이며, 긴 예측 구간을 한 번에 내는 디코더를 쓴다. 이 연구의 `InformerModel`은 `TransformerModel`과 층 구성이 같고 기본 dropout 값(0.1)도 같아 사실상 같은 모델이다(2번·3번 코드에서 확인). | 원 설계는 긴 시퀀스 예측의 연산량을 줄인다. | 이 연구의 근사본에는 ProbSparse도 증류도 없어 Transformer와 구분되지 않는다. 기존 카탈로그는 "dropout만 추가"라고 적었으나 `nn.TransformerEncoderLayer`의 기본 dropout이 0.1이라 코드상 완전히 같다. | Zhou 외(2020), *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, arXiv:2012.07436(arXiv 대조 완료). 게재처 AAAI 2021은 기존 카탈로그 표기이며 **게재처 미검증**. 참고: Stefaniuk & Ślepaczuk(2025), *Informer in Algorithmic Investment Strategies on High Frequency Bitcoin Data*, arXiv:2503.18096(arXiv 대조 완료, 초록은 GMADL 손실 Informer가 매수후보유를 앞섰다고 보고). 16번 코드 주석은 이 논문을 "0-붕괴 실증"으로 적었으나 초록에서는 확인하지 못했다. | 저자 공식 저장소는 https://github.com/zhouhaoyi/Informer2020(확인(200)). neuralforecast에도 `Informer`가 있다(설치본에서 확인). 이 연구가 쓴 것은 자체 근사이다. | 사용 연구는 2번·3번(`ModelZoo.InformerModel`)뿐이다. 자체 근사이며 이후 미사용이다. |
| Non-stationary Transformer | 원 설계는 입력 창을 정규화(Series Stationarization)하고, 정규화로 지워진 평균·표준편차 정보를 어텐션에 다시 넣는 De-stationary Attention을 더한다. 이 연구의 `NonStatTFModel`은 창 평균을 빼 Transformer를 태운 뒤 평균을 더하는 것뿐이다. | 원 설계는 분포가 시간에 따라 변하는 데이터를 겨냥한다. | 이 연구의 근사본에는 De-stationary Attention이 없다. | Liu 외(2022), *Non-stationary Transformers: Exploring the Stationarity in Time Series Forecasting*, arXiv:2205.14415(arXiv 대조 완료). 게재처 NeurIPS 2022는 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 저장소는 https://github.com/thuml/Nonstationary_Transformers(확인(200)). 이 연구는 사용하지 않았다. | 사용 연구는 2번·3번(`ModelZoo.NonStatTFModel`)뿐이다. 자체 근사이며 이후 미사용이다. 16번은 같은 문제의식의 정규화(RevIN 등)를 시험하고 참고문헌에 이 논문을 올렸으나, 이 논문의 모델은 쓰지 않았다. |
| Autoformer | 시계열을 추세와 계절 성분으로 나누는 분해 블록을 모델 안에 반복해 넣고, 점곱 어텐션 대신 자기상관(auto-correlation)으로 주기적 의존성을 찾는다. | 주기 구조가 뚜렷한 장기 예측에 강하도록 설계됐다. | 27번에서 전 구간 C등급(격차 0.060~0.148)이었다. 이 연구의 근사본은 이동평균 분해와 1층 인코더뿐이고 자기상관 메커니즘이 없다. 입력이 (요일, 시) 주기를 제거한 값이라 자기상관이 찾을 주기가 줄었을 수 있으나 검증하지 않았다. | Wu 외(2021), *Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting*, arXiv:2106.13008(arXiv 대조 완료). 게재처 NeurIPS 2021은 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 저장소는 https://github.com/thuml/Autoformer(확인(200)). 이 연구는 27번에서 neuralforecast `Autoformer`를 썼다. | 사용 연구는 2번·3번(장난감: 이동평균 3 분해 뒤 마지막 위치 선형), 8번·9번·10번(`AutoformerLike`: 이동평균 9 분해, 1층 인코더, 10번에서는 실패 통제 모델), 27번(neuralforecast 라이브러리 구현, 시드 5개)이다. 27번에서 모델마다 기본 윈도 배치가 달라(Autoformer는 1,024) 모든 neuralforecast 모델의 `windows_batch_size`를 256으로 통일했다(27번 코드 주석, 메모리 부족 사고는 카탈로그 6-1절 기록). |
| PatchTST | 시계열을 겹치는 패치로 잘라 토큰으로 만들고 Transformer 인코더에 넣는다. 원 설계의 핵심은 채널독립으로, 다변량의 각 채널을 따로 처리한다. | 패치로 어텐션 비용을 줄이고 긴 입력을 읽는다. 27번 신규 어텐션 중 가장 나았다(4시간 격차 0.004로 A등급, 12시간 0.025로 B등급, 시드 5개 평균). | 15분·30분에서는 격차 0.083·0.054(C등급)였다. 자체 근사 `PatchTSTLike`는 패치마다 (수익률, 절댓값) 두 채널을 한 토큰에 합치는 채널혼합 구조라 원 설계와 반대이며, 24번에서 학습손실은 줄어도 검증손실이 개선되지 않아 제외됐다. 2번·3번의 장난감은 마지막 5개 값을 선형층 하나에 넣는 것이 전부이다. | Nie 외(2022), *A Time Series is Worth 64 Words: Long-term Forecasting with Transformers*, arXiv:2211.14730(arXiv 대조 완료). 게재처 ICLR 2023은 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 저장소는 https://github.com/yuqinie98/PatchTST(확인(200)). 이 연구는 27번에서 neuralforecast `PatchTST`를 썼다. | 사용 연구는 2번·3번(장난감), 8번 이후(`PatchTSTLike`, 24번 진단 뒤 `engine/models.py`에 패치 위치 임베딩 추가) 9번~12번·15번·16번·23번·24번(자체 근사, 24번 20종목 결과에서 제외 결정, 근거 `test/models/24_volatility_maxscale_refit_test.py`의 `EXCLUDED_MODELS`), 27번(neuralforecast 라이브러리 구현, 시드 5개)이다. 11번에는 분포 출력 변형(`DistributionalPatchTST`)도 있다. 기존 카탈로그 4절의 "채널독립이 원 논문의 핵심" 서술은 맞고, 27번의 단변량 입력에서는 채널독립이 자연스럽게 지켜진다. |
| iTransformer | 어텐션의 토큰을 시점이 아니라 변수(채널)로 뒤집는다. 각 변수의 전체 시계열을 하나의 토큰으로 만들고, 어텐션은 변수 사이 관계를, 피드포워드는 한 변수의 시간 패턴을 처리한다. | 변수 수가 많은 다변량 예측에서 변수 간 상관을 직접 학습한다. 초록은 임의 길이의 입력창을 더 잘 활용하고 변수 사이 일반화 능력이 향상됐다고 보고한다. | 이 연구는 변수가 2개(로그 RV, 부호 있는 블록 수익률)뿐이라 변수 어텐션이 사실상 토큰 2개 사이에서만 일어난다. 27번에서 15분~1시간 격차가 0.197~0.214(C등급)로 어텐션 4종 중 가장 나빴다. 24번에서는 자체 근사 `ITransformerLike`가 20종목 17개 모델 중 16위로 naive보다 못해 제외됐다. | Liu 외(2023), *iTransformer: Inverted Transformers Are Effective for Time Series Forecasting*, arXiv:2310.06625(arXiv 대조 완료). 게재처 ICLR 2024는 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 저장소는 https://github.com/thuml/iTransformer(확인(200)). 이 연구는 27번에서 neuralforecast `iTransformer`(`n_series=2`)를 썼다. 외부 가이드가 안내하는 `pip install iTransformer` 패키지는 Phil Wang(lucidrains)의 제3자 재구현이다(PyPI 메타데이터에서 확인). | 사용 연구는 4번(`ITransformerLite` 프로토타입), 8번·15번·16번·23번·24번(`ITransformerLike`, 자체 근사, 24번 결과 16/17위로 제외, 근거 `24_volatility_maxscale_refit_test.py`), 25번(24번 결과에서 예측 분산이 실제의 몇 배로 폭주한 진단 분석), 27번(neuralforecast 라이브러리 구현, 두 변수 입력, 시드 5개)이다. **투자 활용 참고(외부 자료, 논문 아님)**: https://github.com/gameworkerkim/vibe-investing/blob/main/TechDoc/TimesFM/iTransformer_%EA%B0%80%EC%9D%B4%EB%93%9C.md |
| TimeXer | 예측 대상(내생) 변수는 패치 토큰과 전역 토큰으로, 외생 변수는 변수 하나당 토큰 하나로 만든다. 내생 패치 사이에는 셀프어텐션을, 전역 토큰과 외생 토큰 사이에는 교차 어텐션을 함께 쓰며, 전역 토큰이 외생 정보를 내생 패치에 전달하는 다리 역할을 한다. | 외생 변수가 있는 예측에서 대상 변수와 외생 정보를 구분해 쓴다. | 27번에서 외생 변수를 넣지 않아(`n_series=1`, `hist_exog_list` 미지정, 입력 열은 `y`와 `available_mask`뿐) 설계 취지의 외생 경로가 이 연구에서는 쓰이지 않았다. 격차는 15분 0.148(C), 4시간 0.019(B), 12시간 0.057(C)이었다. 4번의 `TimeXerLite`와 8번의 `TimeXerLike`는 구조가 크게 다른 근사이다. | Wang 외(2024), *TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables*, arXiv:2402.19072(arXiv 대조 완료). 게재처 NeurIPS 2024는 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 저장소는 https://github.com/thuml/TimeXer(확인(200)). 이 연구는 27번에서 neuralforecast `TimeXer`를 썼다. | 사용 연구는 4번(`TimeXerLite`: 시장 변수가 질의, 텍스트 변수가 키·값인 교차 어텐션 프로토타입), 8번(`TimeXerLike`: 표준 Transformer 출력에 (1 + 0.1 x 외생 게이트)를 곱함, 자체 근사), 27번(neuralforecast 라이브러리 구현, 외생 변수 없이 단변량으로 사용, 시드 5개)이다. |
| mTAND | 원 설계는 학습한 연속시간 임베딩으로 여러 개의 시간 어텐션을 만들어 불규칙하게 관측된 시계열을 고정된 격자 위로 재구성한다. 이 연구의 `mTANDModel`은 선형 시간 임베딩과 `nn.MultiheadAttention` 한 층이 전부이다. | 원 설계는 불규칙 샘플링 시계열에서 결측 보간 없이 직접 학습할 수 있다. | 이 연구의 근사본에는 참조 시점으로의 재구성이 없다. 이후 연구가 15분 시간 격자를 복원해 규칙 간격 입력으로 바꿔 원 설계의 이점이 필요하지 않게 됐다. | Shukla & Marlin(2021), *Multi-Time Attention Networks for Irregularly Sampled Time Series*, arXiv:2101.10318(arXiv 대조 완료). 게재처 ICLR 2021은 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 저장소는 https://github.com/reml-lab/mTAN(확인(200), 저장소 이름이 `mTAN`). 이 연구는 사용하지 않았다. | 사용 연구는 2번·3번(`ModelZoo.mTANDModel`)뿐이다. 자체 근사이며 이후 미사용이다. |

### 4. 상태공간·선형어텐션·확장 LSTM 계열

이 계열은 창 전체를 어텐션으로 비교하는 대신 상태를 갱신하는 선형 비용의 시퀀스 구조이다. 이 연구에서는 Mamba를 이름만 딴 근사(`MambaLike` 등)로 8번 등에서 쓴 뒤, 27번에서 저자 공식 코드(S-Mamba)를 격리 환경으로 실행했다. xLSTM은 CUDA 컴파일러(`nvcc`)가 없어 실행하지 못했고, Gated DeltaNet은 검토만 하고 채택하지 않았다.

| 모델 | 처리 방식·특징 | 장점 | 단점 | 참고문헌 | 코드 저장소 | 사용 연구(비고) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Mamba(자체 근사) | 원 설계는 입력에 따라 상태 갱신 규칙이 달라지는 선택적 상태공간 모델이고, 학습은 병렬 스캔으로, 추론은 재귀로 한다. 이 연구의 `MambaLike`·`MambaModel`·`MambaLite`는 입력을 선형 변환하고 깊이별 인과 합성곱 한 층과 게이트(시그모이드)를 곱하는 구조뿐이라 상태공간 재귀가 없다(코드 주석이 "full Mamba implementation이 아니다"라고 적음). | 원 설계는 시퀀스 길이에 선형 비용으로 긴 문맥을 처리한다. | 이 연구의 근사본은 Mamba의 핵심(선택적 상태공간)을 구현하지 않아 Mamba의 성능을 대표하지 못한다. 원 설계도 언어 모델링용 범용 시퀀스 구조이며 시계열·비정상성 특화 구조는 아니다(기존 카탈로그 서술). | Gu & Dao(2023), *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752(arXiv 대조 완료). | 저자 공식 저장소는 https://github.com/state-spaces/mamba(확인(200)). 이 연구가 근사본에서 쓴 것은 PyTorch 층 위 자체 구성이다. | 사용 연구는 2번·3번(`ModelZoo.MambaModel`), 4번(`MambaLite`), 8번(`MambaLike`, 자체 근사)이다. 7번은 실행 계획만 만들었고 학습하지 않았다. 구현 구분은 자체 근사이며 이후 미사용이다. 27번에서는 공식 코드판인 S-Mamba로 대체됐다. |
| S-Mamba | 각 변수의 전체 시계열을 선형층으로 하나의 토큰으로 만들고, 양방향 Mamba 층으로 변수 사이 상관을, 피드포워드로 시간 의존성을 학습한다. 이 연구는 변수 2개(로그 RV, 표준화한 블록 수익률)를 넣어 변수 축이 길이 2의 시퀀스가 된다. | 긴 입력을 선형 비용으로 처리한다. 저자 코드가 공개돼 있고 저자 ETTh1 설정을 그대로 쓸 수 있다. | 27번에서 전 구간 C등급(격차 0.045~0.154)이었다. 변수가 2개뿐이라 변수 축에 쓰는 Mamba가 길이 2의 시퀀스만 처리한다(이 연구의 입력 구성이며 그 영향은 검증하지 않았다). 27d번은 S-Mamba를 시간 기준으로 병렬로 분류했다. 공식 요구 환경(`torch==2.0.1`, `mamba-ssm==1.2.0`)이 메인 환경과 달라 격리 환경이 필요하다. 저자 코드는 학습 설정을 저자 스크립트에 고정해 이 연구의 다른 neuralforecast 모델과 학습 절차가 다르다. | Wang 외(2024), *Is Mamba Effective for Time Series Forecasting?*, arXiv:2403.11144(arXiv 대조 완료). 게재처 *Neurocomputing*은 기존 카탈로그 표기이며 **게재처 미검증**. | 저자 공식 저장소는 https://github.com/wzhwzhwzh0921/S-D-Mamba(확인(200)). 이 연구는 `third_party/S-D-Mamba`(커밋 e7e8bf0)를 격리 환경 `.venvs/smamba_py310_20261006`(Python 3.10, torch 2.0.1+cu118, mamba-ssm 1.2.0, 미리 컴파일된 wheel)에서 러너 `27_smamba_run.py`로 실행했다. | 사용 연구는 27번(저자 공식 코드, 저자 ETTh1 설정: `e_layers` 2, `d_model` 256, `d_state` 2, `d_ff` 256, 학습률 0.00007, 10에폭, 인내 3, 시드 5개)이다. 27번 설계 셀에는 "S-Mamba는 `nvcc`가 없어 설치 실패로 제외"라고 남아 있으나 결과 보고서 0절이 격리 환경 구성으로 해결해 실제로 포함했다고 정정한다. 이 설계 셀의 제외 문장은 그 뒤 해결된 상태라 현재 사실이 아니다. |
| xLSTM | LSTM에 지수 게이트와 정규화·안정화 기법을 넣고 메모리 구조를 바꾼 확장 LSTM이다. 스칼라 메모리를 쓰는 sLSTM과, 행렬 메모리를 쓰며 완전히 병렬화할 수 있는 mLSTM을 잔차 블록으로 쌓는다. neuralforecast도 NX-AI의 `xlstm` 패키지를 감싸 `xLSTM` 모델을 제공한다(설치본 소스에서 확인). | 논문 초록은 언어 모델링에서 Transformer와 상태공간 모델에 견줄 만한 성능과 규모 확장성을 보고한다. 시계열 예측에서의 성능은 이 연구에서 확인하지 못했다. | 이 연구에서는 실행하지 못했다. CUDA 커널을 import 시점에 컴파일하는데 서버에 `nvcc`가 없고 `sudo`가 필요해 해결하지 못했다. | Beck 외(2024), *xLSTM: Extended Long Short-Term Memory*, arXiv:2405.04517(arXiv 대조 완료). | 저자 공식 저장소는 https://github.com/NX-AI/xlstm(확인(200)). 이 연구는 메인 환경에 설치했다가 `neuralforecast` import가 깨져 `uv remove xlstm`으로 되돌렸고, 격리 환경 `.venvs/xlstm_py313_20261004`에서도 같은 이유로 실패했다. | 제외됨(미채택). 사용한 연구 없음(실행 결과 없음). 제외 이유는 `nvcc` 부재이고 근거는 기존 카탈로그 8절과 26번 FM 카탈로그 보고서(커밋 2d25b0f의 `test/results/26_fm_catalog_20261004/26_fm_catalog_report.md`, 현재 작업 트리에는 없고 git 이력에만 있음)이다. 27번 설계 셀도 xLSTM을 제외로 적는다. 격리 환경 흔적은 남아 있으나 결과 산출물은 확인 못 함. |
| Gated DeltaNet(GDN, 검토만) | 선형어텐션의 한 종류로, 델타 규칙에 게이트를 더해 Mamba2를 개선한다고 논문이 주장한다. 언어 모델링용 빌딩 블록이다. | 논문은 Mamba2와 델타 규칙을 결합해 선형 비용으로 검색·긴 문맥 과제를 개선한다고 보고한다. | 시계열 예측 전용 공식 구현은 없다는 것이 기존 카탈로그 서술이다(이번에 별도로 검증하지 못함). | Yang, Kautz & Hatamizadeh(2024), *Gated Delta Networks: Improving Mamba2 with Delta Rule*, arXiv:2412.06464(arXiv 대조 완료). 게재처 ICLR 2025는 저장소 README에 표기돼 있다. | 저자 공식 저장소는 https://github.com/NVlabs/GatedDeltaNet(확인(200), README가 논문의 공식 PyTorch 구현이라고 밝힘). 이 연구는 사용하지 않았다. | 제외됨(미채택, 사용 연구 없음). 이유는 시계열 예측 전용 공식 구현을 확정하지 못했다는 것이고 근거는 기존 카탈로그 7절과 26번 FM 카탈로그 보고서(커밋 2d25b0f)이다. |

### 5. 시계열 파운데이션 모델(사전학습 + zero-shot)

파운데이션 모델은 대규모·다종 시계열로 미리 학습한 뒤, 이 연구의 데이터에는 재학습 없이(zero-shot) 직전 이력만 읽혀 다음 한 블록을 예측한다. 27번에서 7종(Chronos-Bolt, TimesFM, TTM, Moirai-2, Sundial, Time-MoE, Lag-Llama)을 20종목 x 5구간에서 비교했고, 기존 최선권(트리, GARCH 계열, GRU·LSTM, 커널)을 이기지 못했다(process.md의 27번 결론). 계열 단위 검정에서 파운데이션이 병렬 어텐션보다 30분~12시간에 유의하게 나았다고 보고됐다(구성원 평균 손실의 쌍별 검정이며 가설과 집단은 27b번 보고서에 있다). 사전학습 가중치가 평가 구간(2025-11-11 이후)에 갱신된 모델은 없었다(27번 보고서 2절, Hugging Face 저장소의 가중치 마지막 변경일 조회). 나머지 4종(TimeGPT, MOMENT, TimeFound, TSMamba)은 검토만 하고 쓰지 않았다.

공통 설정은 `27_fm_run.py`에 있다. 문맥은 직전 512블록(Lag-Llama는 1,124블록)이고, 분위수·표본을 내는 모델은 중앙값을, TimesFM·TTM·Time-MoE는 점예측을 쓴다. 표본 기반 모델은 `torch.manual_seed(0)`으로 고정해 반복하지 않았다(시드 1개). 모델마다 필요한 패키지 버전이 달라 메인 환경과 격리 환경 세 개를 나눠 서브프로세스로 실행했다.

| 모델 | 처리 방식·특징 | 장점 | 단점 | 참고문헌 | 코드 저장소 | 사용 연구(비고) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Chronos-Bolt | 원조 Chronos(값을 양자화한 토큰을 T5 계열 모델이 한 스텝씩 생성)를 개선한 패치 기반 변형이다. 과거 문맥을 여러 관측치씩 묶은 패치로 인코더에 넣고, 디코더가 여러 미래 스텝의 분위수를 한 번에 직접 낸다(패키지 README에서 확인). 이 연구는 `amazon/chronos-bolt-small`(README 표 기준 48M 파라미터)을 쓴다. | 원조보다 빠르고 메모리를 덜 쓴다(README는 최대 250배 빠르고 20배 적은 메모리라고 적음). 분위수를 바로 낸다. | 단변량 중심이다(후속 Chronos-2가 다변량·공변량을 지원한다고 README가 적으며 이 연구는 쓰지 않음). 27번에서 15분 구간 격차가 0.149(C등급)였고 4시간 0.012(B), 12시간 0.033(C)이었다. | Ansari 외(2024), *Chronos: Learning the Language of Time Series*, arXiv:2403.07815(원조 Chronos 논문, arXiv 대조 완료). Chronos-Bolt는 독립 논문이 없어 패키지 README와 AWS 블로그(README가 안내)로만 설명되므로 **미검증 인용**(Bolt 자체의 논문 대조 불가). | 공식 저장소는 https://github.com/amazon-science/chronos-forecasting(확인(200)), 가중치는 https://huggingface.co/amazon/chronos-bolt-small(확인(200)). 이 연구는 메인 환경의 패키지 `chronos-forecasting` 2.3.2를 썼다. | 사용 연구는 27번(공식 패키지, 문맥 512, 분위수 0.5, 시드 해당 없음)과 27b·27d번(저장된 예측 분석)이다. 가중치 마지막 변경일은 2024-11-13이다. **투자 활용 참고(외부 자료, 논문 아님)**: TimesFM 가이드의 경쟁 모델 비교 절에 짧게 언급됨(아래 링크). |
| TimesFM(2.5) | 패치 단위로 입력을 토큰화하는 디코더 스타일 어텐션 모델이다(논문 초록의 표현). 기존 카탈로그는 입력 패치보다 긴 출력 패치를 한 번에 생성해 반복 횟수를 줄인다고 설명하나, 출력 패치 길이 같은 세부 값은 논문 본문과 대조하지 못했다. 2.5판은 200M 파라미터, 최대 문맥 16k, 선택적 30M 분위수 헤드를 가진다(패키지 README). 이 연구는 점예측 헤드, 문맥 512, 최대 예측 8, 배치 256, 양수 가정 끔으로 설정했다. | 파운데이션 중 4시간·12시간에서 가장 나은 쪽이다(격차 0.006 A등급, 0.015 B등급). 파라미터가 작아 추론이 빠르다. | 15분·30분·1시간 격차는 0.150·0.089·0.041(C등급)이었다. 배치 기본값이 1이라 처음에 초당 약 6건으로 느렸다(카탈로그 6-1절 사고 기록). 2.5판 가중치는 평가 시작 41일 전(2025-10-01)까지 갱신돼 사전학습 기간과 가장 가깝다. 설치된 패키지는 최신판을 3.0이라고 적고(2026-08, 다변량·공변량 지원) 3.0 가중치는 비상업·비상용 라이선스이다. 2.5까지의 가중치는 Apache-2.0이다(패키지 README 확인). | Das 외(2023), *A decoder-only foundation model for time-series forecasting*, arXiv:2310.10688(arXiv 대조 완료, 게재처 ICML 2024는 패키지 README에 표기). 2.5판 자체의 논문은 없고 변경점은 패키지 README에만 있다. | 공식 저장소는 https://github.com/google-research/timesfm(확인(200)), 가중치는 https://huggingface.co/google/timesfm-2.5-200m-pytorch(확인(200)). 이 연구는 메인 환경의 패키지 `timesfm` 3.0.2에서 2.5 클래스 `TimesFM_2p5_200M_torch`를 썼다. | 사용 연구는 27번(공식 패키지, zero-shot)과 27b·27d번이다. 가중치 마지막 변경일은 2025-10-01이다. **투자 활용 참고(외부 자료, 논문 아님)**: https://github.com/gameworkerkim/vibe-investing/blob/main/TechDoc/TimesFM/TimesFM_%EB%B6%84%EC%84%9D_%EA%B0%80%EC%9D%B4%EB%93%9C.md (이 문서의 코드 예시는 1.x·2.0 API이고 2.5는 `ForecastConfig`를 쓴다). |
| TTM(Tiny Time Mixers, r2) | 어텐션 없이 완전연결층으로만 시간 방향·채널 방향 혼합을 하는 경량 TSMixer 구조이다. 논문은 적응형 패칭, 다양한 해상도 샘플링, 해상도 접두 튜닝으로 서로 다른 해상도의 데이터를 작은 용량으로 사전학습한다고 설명한다. 이 연구는 `ibm-granite/granite-timeseries-ttm-r2`의 기본 가지(카탈로그 6-1절 기준 문맥 512, 예측 96)에서 첫 예측 스텝을 쓴다. | 100만 파라미터부터 시작하는 작은 모델이라 CPU만으로도 실행할 수 있다고 논문이 보고한다. 추론이 빠르다. | 이 연구에서는 모델카드가 분·시간 해상도만 지원해 4시간·12시간 블록 결과를 결론에서 뺐다(40행). 지원하는 15분·30분·1시간에서도 격차가 0.200·0.120·0.059(C등급)로 Lag-Llama와 함께 파운데이션 중 가장 나쁜 쪽이었다. | Ekambaram 외(2024), *Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series*, arXiv:2401.03955(arXiv 대조 완료). 기존 카탈로그가 적은 제목(`...Fast Pretrained Models for Enterprise Time-series Forecasting`)은 현재 arXiv 제목과 달라 이번에 고쳐 적었다. 게재처 NeurIPS 2024는 **게재처 미검증**. | 공식 저장소는 https://github.com/ibm-granite/granite-tsfm(확인(200)), 가중치는 https://huggingface.co/ibm-granite/granite-timeseries-ttm-r2(확인(200)). 이 연구는 메인 환경의 패키지 `granite-tsfm` 0.3.10(`tsfm_public`)을 썼다. | 사용 연구는 27번(공식 패키지, zero-shot)이다. 4시간·12시간 40행은 `27_model_expansion_excluded_ttm_unsupported_resolution.csv`에 기록으로만 남기고 순위·검정에서 제외했다(근거 27번 보고서 0절과 카탈로그 6-1절). 가중치 마지막 변경일은 2024-10-09이다. |
| Moirai-2(2.0) | 디코더 전용 Transformer이고 분위수 손실로 학습하며 여러 토큰을 한 번에 예측한다(multi-token prediction). 1.0판의 마스크드 인코더, 다중 패치 입력, 혼합분포 출력을 단일 패치와 분위수 손실로 단순화했다. 이 연구는 `Salesforce/moirai-2.0-R-small`에서 아홉 분위수 중 가운데(중앙값)를 쓴다. | 논문은 1.0 최대판보다 2배 빠르고 30배 작으면서 성능이 더 좋다고 보고한다. | 27번에서 15분 구간 격차 0.180(C등급)으로 나빴고 4시간 0.013(B), 12시간 0.029(B)였다. 논문은 파라미터를 늘려도 성능이 정체하고 긴 예측 구간에서 하락한다고 적는다. 기존 카탈로그의 Moirai 행은 마스크드 인코더와 any-variate attention을 설명하는데 이는 1.0판 설명이라 2.0판 설명과 어긋난다(이번에 지적). | Woo 외(2024), *Unified Training of Universal Time Series Forecasting Transformers*, arXiv:2402.02592(1.0판, arXiv 대조 완료). 이 연구가 쓴 2.0판은 Liu 외(2025), *Moirai 2.0: When Less Is More for Time Series Forecasting*, arXiv:2511.11698(arXiv 대조 완료). | 공식 저장소는 https://github.com/SalesforceAIResearch/uni2ts(확인(200)), 가중치는 https://huggingface.co/Salesforce/moirai-2.0-R-small(확인(200)). 이 연구는 격리 환경 `.venvs/moirai_py312_20261004_192438`(`uni2ts` 2.0.0, torch 2.4.1+cu121)을 썼다. | 사용 연구는 27번(공식 패키지, zero-shot, 서브프로세스)이다. 가중치 마지막 변경일은 2025-08-06이다. **투자 활용 참고(외부 자료, 논문 아님)**: 위 TimesFM 가이드의 경쟁 모델 비교 절에 Moirai 1.0 기준으로 짧게 언급됨. |
| Sundial | 디코더 전용 Transformer가 다음 패치의 분포를 흐름 매칭(flow matching) 기반 TimeFlow 손실로 학습해, 이산화 없이 연속값에서 사전학습하고 여러 개의 가능한 미래 경로를 생성한다. 이 연구는 `thuml/sundial-base-128m`에서 표본 50개의 중앙값을 쓴다. | 논문은 1조 시점 규모의 데이터(TimeBench)로 사전학습했고, 점예측·확률예측 벤치마크에서 최고 수준이며 수 밀리초 안에 zero-shot 예측을 낸다고 보고한다. | 공식 코드가 `transformers==4.40.1`을 요구해 메인 환경(5.16.1)에서 `DynamicCache` 오류가 나 격리 환경이 필요했다. 27번에서 15분 0.155(C), 4시간 0.018(B), 12시간 0.024(B)였다. 표본 추출 방식이라 시드 고정으로 한 번만 돌려 표본 잡음이 남는다. | Liu 외(2025), *Sundial: A Family of Highly Capable Time Series Foundation Models*, arXiv:2502.00816(arXiv 대조 완료). | 공식 저장소는 https://github.com/thuml/Sundial(확인(200)), 가중치는 https://huggingface.co/thuml/sundial-base-128m(확인(200)). 이 연구는 격리 환경 `.venvs/legacy_hf_py312_20261004`(transformers 4.40.1)를 썼다. | 사용 연구는 27번(공식 코드, zero-shot)이다. 가중치 마지막 변경일은 2025-05-14이다. |
| Time-MoE | 디코더 전용 Transformer에 희소 전문가 혼합(MoE)을 넣어, 예측마다 일부 전문가만 활성화해 연산을 줄이면서 모델 용량을 키운다. 이 연구는 `Maple728/TimeMoE-50M`을 쓰고 문맥별 평균·표준편차로 정규화한 뒤 점예측을 낸다. | 논문은 3,000억 시점 규모의 데이터(Time-300B)로 최대 24억 파라미터까지 확장했고, 같은 활성 파라미터의 밀집 모델보다 낫다고 보고한다. 이 연구의 파운데이션 중 15분 구간 격차가 가장 작았다(0.145, C등급). | 공식 코드가 `transformers==4.40.1`을 요구한다(Sundial과 같은 격리 환경 공유). 배치 256은 메모리가 부족해 64로 낮췄다. 15분 구간도 C등급이라 기존 최선권에는 못 미쳤다. | Shi 외(2024), *Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts*, arXiv:2409.16040(arXiv 대조 완료). | 공식 저장소는 https://github.com/Time-MoE/Time-MoE(확인(200)), 가중치는 https://huggingface.co/Maple728/TimeMoE-50M(확인(200)). 이 연구는 격리 환경 `.venvs/legacy_hf_py312_20261004`를 썼다. | 사용 연구는 27번(공식 코드, zero-shot)이다. 가중치 마지막 변경일은 2024-09-21이다. **투자 활용 참고(외부 자료, 논문 아님)**: 위 TimesFM 가이드의 경쟁 모델 비교 절에 짧게 언급됨. |
| Lag-Llama | LLaMA 스타일 디코더 전용 Transformer가 값 대신 시차(lag) 특징을 공변량으로 받아 단변량 확률 예측을 한다. 매 시점 Student-t 분포의 모수를 내고, 미래는 한 스텝씩 순차 표본 추출로 만든다. 이 연구는 문맥 1,124블록(시차 최대 1,092 포함), 표본 100개의 중앙값을 쓴다. | 단변량 확률 예측을 위해 설계돼 표본 경로를 바로 얻을 수 있다. | 한 스텝씩 순차 표본을 뽑아 패치 단위 직접 예측 모델보다 1,000건당 처리 시간이 훨씬 길다(환경 로그 기준 9.89초, 배치 64). 27번에서 15분 0.212(C)로 가장 나빴고 4시간 0.037·12시간 0.032(C)로 두 장기 구간에서도 B등급에 못 들었다. 환경에서 세 가지 버전 충돌(`pkg_resources`, 판다스 `'Q'` 별칭, `weights_only`)을 차례로 해결해야 했다. 기존 카탈로그의 "최초의 공개 시계열 파운데이션 모델"이라는 서술은 이번 초록 대조에서 확인하지 못했다. | Rasul 외(2023), *Lag-Llama: Towards Foundation Models for Probabilistic Time Series Forecasting*, arXiv:2310.08278(arXiv 대조 완료). arXiv 최초 제출은 2023-10-12이다(기존 카탈로그는 연도를 2024로 적었으나 게재 연도인지 대조하지 못함). | 공식 저장소는 https://github.com/time-series-foundation-models/lag-llama(확인(200)), 체크포인트는 https://huggingface.co/time-series-foundation-models/Lag-Llama(확인(200)). 이 연구는 `third_party/lag-llama`(VERSION 0.1.0, git 관리 밖의 복사본이라 원본 커밋은 확인 못 함)와 격리 환경 `.venvs/lagllama_py312_20261004`(`gluonts` 0.14.4)를 썼다. | 사용 연구는 27번(공식 코드, zero-shot)이다. 가중치 마지막 변경일은 2024-02-07이다. **투자 활용 참고(외부 자료, 논문 아님)**: 위 TimesFM 가이드의 경쟁 모델 비교 절에 짧게 언급됨. |
| TimeGPT(검토만, 제외) | Nixtla가 API로만 제공하는 상용 시계열 파운데이션 모델이다. 내부 구조는 공개되지 않았다. | 설치 없이 API 호출만으로 쓰고 미세조정과 이상 탐지를 지원한다고 외부 가이드가 설명한다. | API 계정과 키가 필요하고 가중치가 공개돼 있지 않아 이 연구에서는 제외했다. 외부 가이드 기준 사용 요금이 든다. | Garza 외(2023), *TimeGPT-1*, arXiv:2310.03589(arXiv 대조 완료). | 모델은 비공개이고 파이썬 SDK 저장소만 https://github.com/Nixtla/nixtla(확인(200), README가 SDK 설치를 안내하고 제목을 TimeGPT-2.1로 적음)에 있다. 논문은 TimeGPT-1이다. 이 연구는 사용하지 않았다. | 제외됨(미채택, 사용 연구 없음). 이유는 API 계정·키 필요로 2026-10-04 사용자 지시에 따라 뺐다는 것이다. 근거는 기존 카탈로그 8절, 26번 FM 카탈로그 보고서(커밋 2d25b0f), 27번 설계 셀이다. **투자 활용 참고(외부 자료, 논문 아님)**: 위 TimesFM 가이드의 경쟁 모델 비교 절에 짧게 언급됨. |
| MOMENT(검토만, 제외) | 시계열 이해 전반(예측, 보간, 이상 탐지, 분류)을 위한 개방형 파운데이션 모델 가족이다.  | 하나의 모델로 여러 과제를 처리하고 마지막 선형층만 미세조정하는 선형 탐침으로 활용할 수 있다. | 저장소 README는 zero-shot 단기 예측이 일부 데이터에서 ARIMA보다 낫다고 적는 정도로, 예측에 특화된 모델이 아니다. 이 연구에서는 사용하지 않았다. | Goswami 외(2024), *MOMENT: A Family of Open Time-series Foundation Models*, arXiv:2402.03885(arXiv 대조 완료, README가 ICML 2024 채택을 표기). | 공식 저장소는 https://github.com/moment-timeseries-foundation-model/moment(확인(200), README가 위 논문과 가중치 `AutonLab/MOMENT-1-large`를 연결). 이 연구는 사용하지 않았다. | 제외됨(미채택, 사용 연구 없음). 기존 카탈로그와 26번 FM 카탈로그 보고서(커밋 2d25b0f)는 "공식 저장소를 특정하지 못해" 제외했다고 적었으나, 이번 조회에서는 공식 저장소가 확인되므로 그 사유는 현재 사실과 다르다. 재검토 여부는 결정 사항이다. **투자 활용 참고(외부 자료, 논문 아님)**: 위 TimesFM 가이드의 경쟁 모델 비교 절에 짧게 언급됨. |
| TimeFound(검토만, 제외) | 인코더-디코더 Transformer 기반 시계열 파운데이션 모델이다. 다중 해상도 패칭으로 여러 규모의 시간 패턴을 잡는다고 논문이 설명한다. 200M·710M 두 크기로 사전학습했다. | 초록은 보지 못한 여러 데이터에서 최신 파운데이션 모델과 대등하거나 나은 zero-shot 예측 성능을 보인다고 보고한다. | 공식 저장소를 확인하지 못해 이 연구에서는 쓰지 못했다. | Xiao 외(2025), *TimeFound: A Foundation Model for Time Series Forecasting*, arXiv:2503.04118(arXiv 대조 완료). | **확인 못 함**. GitHub 검색에서 나온 `makamoa/timefound`는 확인(200)이나 마지막 갱신이 2024-11-14로 논문(2025-03)보다 앞서 이 논문의 공식 저장소인지 판단하지 못했다. | 제외됨(미채택, 사용 연구 없음). 이유는 공식 저장소 미확정이고 근거는 기존 카탈로그 7절과 26번 FM 카탈로그 보고서(커밋 2d25b0f)이다. |
| TSMamba(검토만, 제외) | 순방향·역방향 Mamba 인코더로 구성한 선형 비용의 시계열 파운데이션 모델이다. 사전학습된 Mamba 언어 모델을 활용한 2단계 전이 학습을 쓴다. | 초록은 적은 학습 데이터로 최신 파운데이션 모델에 견줄 zero-shot 성능을 보인다고 보고한다. | 초록이 "코드를 공개할 예정"이라고 적으며 공식 저장소를 확인하지 못했다. | Ma 외(2024), *A Mamba Foundation Model for Time Series Forecasting*, arXiv:2411.02941(arXiv 대조 완료). | **확인 못 함**. GitHub 저장소 검색에서 해당 이름의 저장소가 나오지 않았다. | 제외됨(미채택, 사용 연구 없음). 이유는 공식 저장소 미확정이고 근거는 기존 카탈로그 7절과 26번 FM 카탈로그 보고서(커밋 2d25b0f)이다. |


---

### 6. 구현 구분 한눈에 보기

같은 이름의 모델이 연구 시기마다 다른 구현이었으므로 한 표로 모았다. "자체 근사"가 붙은 칸은 그 모델의 성능으로 읽으면 안 된다.

| 모델 | 1~16번(방향·초기 실험) | 23~26c번(변동성, GRU·LSTM 중심) | 27번 |
| :--- | :--- | :--- | :--- |
| LSTM, GRU | PyTorch 내장 모듈 위 자체 구성 | PyTorch 내장 모듈 위 자체 구성(`engine/models.py`) | 사용 안 함(27번의 순환 기준은 26c 결과를 가져옴), 27c번에서 블록 입력으로 재학습 |
| TCN | 자체 근사(2·3번 1층, 5번 3층, 8·15번 `TCNForecaster`) | 사용 안 함 | 라이브러리 구현(neuralforecast) |
| ModernTCN | 자체 근사(`ModernTCNLike`, 8·15번) | 사용 안 함 | 공식 구현(저자 저장소 커밋 56a9a2c) |
| TimesNet | 자체 근사(`TimesNetLike`, 8·9·10번) | 사용 안 함 | 라이브러리 구현(neuralforecast) |
| DLinear, NLinear | 자체 근사(`DLinearLike`, `NLinearLike`, 8·15번) | 사용 안 함 | 제외(2026-10-05 결정) |
| N-BEATS, DeepAR, ODE-RNN, mTAND, Informer, Non-stationary Transformer | 장난감 자체 근사(2·3번, ODE-RNN은 1번도) | 사용 안 함 | 사용 안 함 |
| MLP(`Linear`) | PyTorch 층 위 자체 구성(5·8~12·14·15번) | 사용 안 함 | 사용 안 함 |
| Transformer | PyTorch 내장 모듈 위 자체 구성(1~5·8번) | 사용 안 함 | 사용 안 함 |
| Autoformer | 자체 근사(2·3번 장난감, 8·9·10번 `AutoformerLike`) | 사용 안 함 | 라이브러리 구현(neuralforecast) |
| PatchTST | 자체 근사(2·3번 장난감, 8~12·15·16번 `PatchTSTLike`) | 자체 근사(23·24번, 24번 결과로 제외) | 라이브러리 구현(neuralforecast) |
| iTransformer | 자체 근사(4번 `ITransformerLite`, 8·15·16번 `ITransformerLike`) | 자체 근사(23·24번, 24번 결과로 제외) | 라이브러리 구현(neuralforecast, 두 변수) |
| TimeXer | 자체 근사(4번 `TimeXerLite`, 8번 `TimeXerLike`) | 사용 안 함 | 라이브러리 구현(neuralforecast, 외생 변수 없이) |
| Mamba, S-Mamba | 자체 근사(2·3·4·8번) | 사용 안 함 | 공식 구현(S-Mamba 저자 저장소 커밋 e7e8bf0, 격리 환경) |
| xLSTM | 해당 없음(실행 못 함) | 해당 없음 | 제외(`nvcc` 부재로 실행 실패) |
| Chronos-Bolt, TimesFM, TTM | 해당 없음(파운데이션은 27번에서 처음 도입) | 해당 없음 | 공식 패키지(메인 환경) |
| Moirai-2, Sundial, Time-MoE, Lag-Llama | 해당 없음 | 해당 없음 | 공식 패키지·코드(격리 환경) |
| TimeGPT, MOMENT, TimeFound, TSMamba, GDN | 해당 없음 | 해당 없음 | 검토만, 제외 |

---

### 7. 퀀트 트레이딩 활용 참고 자료(외부 자료, 논문 아님)

연구 결과를 최종적으로 퀀트 트레이딩에 쓰려는 목적이 있으므로, 이전에 받은 TimesFM 분석 가이드와 같은 저장소(https://github.com/gameworkerkim/vibe-investing)의 `TechDoc` 아래 문서를 다시 조사했다. 이 자료들은 논문이 아니라 개인 저장소가 정리한 안내 문서이며 동료 심사를 거치지 않았다. 문서가 적은 수치와 사실 관계는 이번에 검증하지 않았다.

#### 7.1 조사 방법과 찾은 문서 목록

`curl -s https://api.github.com/repos/gameworkerkim/vibe-investing/contents/TechDoc`로 `TechDoc` 아래 폴더 목록을 확인하고, 저장소 트리에서 언어별 변형(`_CN`, `_EN`, `_JA`)을 뺀 마크다운 110개를 내려받아 이 연구의 모델 이름(Chronos, Moirai, TimesFM, Time-MoE, Lag-Llama, TimeGPT, Sundial, TTM, PatchTST, iTransformer, TimesNet, Mamba, xLSTM, N-BEATS, TCN, LSTM, GRU, Informer, Autoformer, DLinear)으로 검색했다. 모델을 직접 다룬 문서는 아래 셋이다. 다른 문서(LLM·에이전트 안내 등)는 "transformer"를 일반 단어로만 쓰며 이 연구의 모델을 다루지 않는다.

| 문서 | 주소 | 관련 모델 |
| :--- | :--- | :--- |
| TimesFM 분석 가이드(2026-06-19 등재, 문서 하단은 2025년 6월 갱신으로 표기) | https://github.com/gameworkerkim/vibe-investing/blob/main/TechDoc/TimesFM/TimesFM_%EB%B6%84%EC%84%9D_%EA%B0%80%EC%9D%B4%EB%93%9C.md | TimesFM. 경쟁 모델 비교 절에서 Chronos-Bolt, TimeGPT, Moirai, Time-MoE, MOMENT, Lag-Llama, TimesNet, PatchTST, iTransformer를 한 단락씩 언급한다. |
| iTransformer 시작 가이드 | https://github.com/gameworkerkim/vibe-investing/blob/main/TechDoc/TimesFM/iTransformer_%EA%B0%80%EC%9D%B4%EB%93%9C.md | iTransformer, TimesFM과의 비교. |
| Qlib 시작 가이드(한국어) | https://github.com/gameworkerkim/vibe-investing/blob/main/TechDoc/Quant_Qlib/Qlib-getting-started-KR.md | 모델 동물원 절이 LSTM, GRU, Transformer, TCN 등 25종 이상을 나열한다. 이 연구의 LSTM·GRU·TCN·Transformer 행에 연결했다. |

찾지 못한 것: Chronos-Bolt, Moirai, Time-MoE, Lag-Llama, TimeGPT, MOMENT, Sundial, TTM, Mamba, 그 밖의 모델 전용 문서는 `TechDoc` 아래에 없다(해당 없음, 위 검색에서 이름이 나온 문서가 위 셋뿐이기 때문이다). 같은 저장소의 `GS_Quant` 안내는 "시계열"을 언급하지만 이 연구의 모델을 다루지 않는다.

#### 7.2 문서별 요지

**TimesFM 분석 가이드.** 용도는 파운데이션 모델의 zero-shot 예측을 거시경제 지표, 기업 매출·수요 예측, 주가 추세와 변동성 예측에 쓰는 것이다. 문서는 변동성 예측이 분위수 예측과 함께 쓰일 때 효과적(별 4개)이고 주가 방향 예측은 정확도가 제한적(별 3개)이며, 주가보다 거시지표·매출·수요에 더 적합하다고 평가하고 LLM 뉴스 감성 분석과 결합할 때 시너지가 크다고 적는다. 주의점은 시장 구조 변화(regime change)에 취약할 수 있고, 거래 전략에는 리스크 관리와 포트폴리오 최적화 계층이 따로 필요하며, 예측 근거를 설명하기 어려운 블랙박스이고, 공개 체크포인트가 단변량 중심이며, 단기(1~5일)보다 중장기에 더 맞을 수 있고, 도입 전 대상 도메인에서 검증이 필수라는 것이다. 이 연구와 견줄 때의 유의점: 이 문서가 더 맞다고 보는 범위(중장기, 거시지표)와 달리 이 연구는 15분~12시간 암호화폐 변동성이다. 문서 안에서 체크포인트 이름(`2.0-500m`과 `2.0-200m`)과 최신 버전 표기(2.5)가 서로 어긋나고 코드 예시는 구 API이다. 설치된 `timesfm` 패키지 README는 최신판이 3.0이며 3.0 가중치는 비상업·비상용 라이선스라고 적으므로(2.5까지는 Apache-2.0), 실전 운용 때는 라이선스를 먼저 확인해야 한다.

**iTransformer 시작 가이드.** 용도는 종목 사이 상관(커플링) 신호를 잡아 섹터 로테이션 포착, 종목 간 공분산 구조 예측으로 리밸런싱 트리거 생성, 페어 트레이딩 스프레드 예측, 여러 경제 지표를 함께 본 예측 시장 확률 추정이다. 주의점은 변수 수 N에 대해 연산이 N의 제곱으로 늘고, 시간 방향의 국소 정보가 약해져 단변량이나 시간 패턴이 중요한 경우 불리할 수 있으며, 계절성 노이즈에 민감하고, 새 도메인마다 학습이 필요해 zero-shot이 안 된다는 것이다. 문서 저자도 "손이 많이 가고 TimesFM과 섞어 쓰면 유용하다"고 적는다. 이 연구와 견줄 때의 유의점: 문서가 강점으로 삼는 많은 변수 사이 상관을 이 연구는 쓰지 못했다. 27번에서 변수가 로그 RV와 블록 수익률 둘뿐이었고 15분~1시간에서 어텐션 4종 중 가장 나빴다. 문서가 안내하는 `pip install iTransformer`는 저자 공식 코드가 아니라 Phil Wang의 제3자 재구현이다(PyPI 메타데이터에서 확인).

**Qlib 시작 가이드(한국어).** 용도는 Microsoft Qlib으로 데이터 처리, 모델 훈련, 백테스트, 리포트 분석을 한 틀에서 하는 것이며, 표 형태 팩터(Alpha158)에는 GBDT 계열, 원시 가격·거래량 시계열(Alpha360)에는 GRU 같은 딥러닝 계열을 권한다. 주의점은 거래비용을 반영하면 연 수익률이 약 5%p 낮아지고, 미래 참조(look-ahead)를 피해야 하며, 검증 구간에 맞춘 반복 튜닝은 시험 성능을 부풀리고, 무작위성이 있는 딥러닝은 최소 20회 반복 평균을 보고 단일 실행으로 모델 우열을 판단하면 안 되며, 일간 IC 0.03~0.05는 비용과 용량 제약을 넘지 못하는 경우가 많고, 백테스트가 실거래 성과를 보장하지 않는다는 것이다. 이 연구와 견줄 때의 유의점: 이 연구는 예측 손실(QLIKE) 비교까지만 했고 거래 비용이 반영된 수익 평가는 하지 않았다. 신경망 시드는 5개로 문서의 권장 20회보다 적다.

---

### 8. 인용 검증 요약

arXiv 대조 완료(제목, 저자 확인): 1406.1078(GRU), 1706.03762(Transformer), 2012.07436(Informer), 2205.14415(Non-stationary Transformer), 2106.13008(Autoformer), 2211.14730(PatchTST), 2310.06625(iTransformer), 2402.19072(TimeXer), 2101.10318(mTAND), 2205.13504(DLinear·NLinear), 2210.02186(TimesNet), 1905.10437(N-BEATS), 1803.01271(TCN), 1704.04110(DeepAR), 1907.03907(ODE-RNN), 2312.00752(Mamba), 2403.11144(S-Mamba), 2405.04517(xLSTM), 2412.06464(Gated DeltaNet), 2403.07815(Chronos), 2310.10688(TimesFM), 2401.03955(TTM), 2402.02592(Moirai 1.0), 2511.11698(Moirai 2.0), 2502.00816(Sundial), 2409.16040(Time-MoE), 2310.08278(Lag-Llama), 2310.03589(TimeGPT-1), 2402.03885(MOMENT), 2503.04118(TimeFound), 2411.02941(TSMamba), 2503.18096(Informer 비트코인 적용 사례).

**미검증 인용**: (1) Hochreiter & Schmidhuber(1997) LSTM 논문은 arXiv에 없어 대조하지 못했다. (2) Luo & Wang(2024) ModernTCN은 arXiv에서 검색되지 않아 저장소 README의 서지와만 일치를 확인했다. (3) Chronos-Bolt와 TimesFM 2.5판은 독립 논문이 없어 패키지 README로만 설명했다. (4) 위 논문들의 학회·학술지 게재처(NeurIPS, ICLR, AAAI, EMNLP, IJF, Neurocomputing 등)는 arXiv 메타데이터에 없어 대조하지 못했고 기존 카탈로그 표기를 따랐다(TimesFM, MOMENT, Gated DeltaNet, ModernTCN의 게재처는 저장소 또는 패키지 README에 적혀 있어 그 표기를 따름).

저장소 주소 확인 결과: 이 문서에 적은 GitHub 주소는 모두 `curl`로 200을 확인했다. 처음 시도한 `Nixtra/neuralforecast`(철자 오류)는 404였고 `Nixtla/neuralforecast`는 200이다. 알려진 옛 이름 `ElementAI/N-BEATS`는 301로 `ServiceNow/N-BEATS`(200)로 이동한다. `makamoa/timefound`는 200이지만 공식 저장소인지 판단하지 못했고, TSMamba는 저장소를 찾지 못했다(확인 못 함).

---

## 부록 A. 실행 설정 점검표(실행 전 대조 필수)

공식 문서·저자 스크립트의 실행 예시 값과 우리 설정을 나란히 둔다. **새 모델을 돌리거나 설정을 바꾸기 전에 이 표와 대조한다.**
배치 관련 기본값을 확인하지 않아 계산이 수십 배 느려지거나(TimesFM `per_core_batch_size` 기본 1, 2026-10-06) 메모리 부족으로
실패한(Time-MoE, Autoformer 기본 윈도 배치 1,024) 일이 있었다. 러너(`test/models/27_fm_run.py`)는 시작할 때 적용된 배치 값을 출력하고,
배치가 1이면 멈춘다. 확인 근거는 각 모델 카드·저장소 README(2026-10-06 조회)다.

| 모델 | 공식 예시 설정 | 우리 설정 | 차이와 이유 |
| :--- | :--- | :--- | :--- |
| TimesFM 2.5 | `ForecastConfig(max_context=1024, max_horizon=256, normalize_inputs=True, ...)`. `per_core_batch_size` 기본값 1(README 예시에 없음) | max_context 512, max_horizon 8, **per_core_batch_size 256**, infer_is_positive False, force_flip_invariance False | 로그 RV는 음수라 양수 가정을 끈다. 배치 1은 초당 약 6건이라 256으로 지정(4,096건 6초) |
| Chronos-Bolt | `predict_quantiles(context, prediction_length=12)`, 문맥은 1D 텐서 목록 또는 왼쪽 패딩 2D 텐서 | 문맥 512, 한 번에 256개, 분위수 0.5 | 공식 속도 비교도 문맥 512 기준 |
| TTM r2 | 문맥·예측 길이별 가지(`get_model()` 권장), **분·시간 해상도만 지원(10분·15분·1시간)**, 0 패딩으로 문맥 늘리기 비권장 | main 가지(512-96), 첫 예측 스텝 | **4시간·12시간 블록은 공식 지원 해상도 밖** → 이 두 구간의 TTM은 결론에서 뺀다 |
| Moirai-2.0 | 문맥 길이 지정, 예측 분위수 출력 | 문맥 512, 한 번에 256개, 중앙 분위수 | - |
| Sundial | `generate(seqs, max_new_tokens, num_samples=20)`, 문맥 최대 2,880 | 문맥 512, 표본 50개 중앙값, 한 번에 256개 | 표본 잡음을 줄이려 표본 수를 늘림 |
| Time-MoE | `generate(정규화한 문맥, max_new_tokens)`, 사용자가 직접 정규화, 문맥 최대 4,096 | 문맥 512, 문맥별 평균·표준편차 정규화, 배치 64(메모리 부족 시 자동 절반) | 배치 256은 다른 GPU 작업과 겹칠 때 메모리 부족 |
| Lag-Llama | 체크포인트의 하이퍼파라미터 사용, context_length 32 + 최대 lag 1,092, 표본 100개 | 같음, gluonts 배치 256 | - |
| neuralforecast 6종 | 모델마다 기본 `windows_batch_size`가 다름(Autoformer 1,024 등), max_steps 1,000~5,000 | **windows_batch_size 256 통일**, max_steps 1,000, 조기종료 인내 3 | 모델 간 학습 예산을 같게 맞춤 |
| ModernTCN | 저자 ETTh1 스크립트: batch 512, lr 1e-4, patch 8/4, 큰 커널 51 | 같음 | - |
| S-Mamba | 저자 ETTh1 스크립트: batch 32, lr 7e-5, d_model 256, d_state 2, 10에폭 | 같음 | - |
| GRU·LSTM | (자체 학습 루프) | 배치 2,048, bf16 혼합정밀도, 메모리 부족 시 배치 절반 | - |

**28번 변경분(2026-10-15, `test/models/28_seed_expansion_test.py`).** 위 표의 27번 값에서 아래만 바꾼다. 나머지는 같다.

| 모델 | 공식 기본값(확인 근거) | 27번 | 28번 | 이유 |
| :--- | :--- | :--- | :--- | :--- |
| neuralforecast 6종 | `max_steps`: PatchTST·Autoformer 5,000, iTransformer·TCN·TimesNet·TimeXer 1,000. `learning_rate`: PatchTST·Autoformer·TimesNet 1e-4, 나머지 1e-3(설치본 3.2.2 생성자 기본값, 2026-10-15 `inspect.signature`로 확인). 조기 종료는 `EarlyStopping`만 쓰고 체크포인트를 끈다(`_base_model.py`) | max_steps 1,000, 기본 학습률, 인내 3, 최선 가중치 복원 없음 | **max_steps 5,000**(라이브러리 기본값 중 큰 값), 검증 100스텝마다, **인내 5**, **최선 가중치 복원 콜백**, **학습률 3점(기본값 × 0.25·1·4)** | 27번에서 상한 도달 14~48%, 멈춘 적합은 최선보다 300스텝 뒤 가중치로 예측했다 |
| ModernTCN | 저자 lr 1e-4 | 저자 값 | 학습률 3점(1e-4 × 0.25·1·4), 나머지 저자 값 | 모든 학습 신경망에 같은 탐색 기회 |
| S-Mamba | 저자 lr 7e-5 | 저자 값 | 학습률 3점(7e-5 × 0.25·1·4), 나머지 저자 값(러너 파일은 그대로 두고 모듈 상수만 바꿔 실행) | 같음 |
| GRU·LSTM | (자체) | 학습률 2e-3·5e-4·1e-4 | **학습률 8e-3·2e-3·5e-4**(중심 2e-3 × 4·1·0.25), 옵티마이저 상태 깊은 복사 | 27번에서 상단 2e-3 선택 71%(범위가 위에서 잘렸을 가능성) |
| 전 학습 모델·정지 분류기 | - | 보정은 내부 모델, 평가 예측은 재적합 모델 | **무재적합**: 내부 모델 하나로 보정·평가 예측 | 보정 계수와 예측 모델 불일치(15분 QLIKE +0.09 악화) 제거 |

학습률은 시드 0에서 (종목, 구간, 모델)마다 내부검증 구간의 척도 보정 QLIKE로 고르고 모든 시드에 고정한다.

**속도 측정(2026-10-15, KRW-BTC, 시드 0, 학습률 3점, GPU 워커 2).** 작업 하나(3점 적합) 소요: GRU·LSTM 13~17초, PatchTST 15~47초, iTransformer 67~95초, TCN 7~65초, Autoformer 135~139초, TimesNet 546~703초, TimeXer 64~97초, ModernTCN 1~12초, S-Mamba 6~105초. 상한 5,000 도달 0건(최장 3,400스텝, Autoformer 12시간). 이로 추정한 GPU 작업량은 시드 0이 약 30작업시간, 시드 1 이상이 시드당 약 10작업시간이며 TimesNet이 약 60%다.

활용 참고(논문 외): TimesFM 개요·장단점·투자 활용·경쟁 모델 비교 정리 — https://github.com/gameworkerkim/vibe-investing/blob/main/TechDoc/TimesFM/TimesFM_%EB%B6%84%EC%84%9D_%EA%B0%80%EC%9D%B4%EB%93%9C.md
(이 문서의 코드 예시 `TimesFmHparams(per_core_batch_size=32, ...)`는 1.x·2.0 API이고, 2.5는 `ForecastConfig`를 쓴다)

---

## 부록 B. 이번 개정에서 바로잡은 기존 카탈로그의 서술

개정 전 `model_catalog.md`(git 이력 `7efe5e6` 이전 판)와 코드·보고서를 대조해 나온 차이다. 코드나 문서로 직접 확인한 항목만 본문에 반영했다.

### B-1. 고전 통계·계량·기계학습 부분

기존 카탈로그의 1~2절(통계·계량, 선형·커널·트리)을 출발점으로 코드와 대조했다. 다른 점과 오류는 다음과 같다.

1. **TAR-GARCH 임계변수 설명이 부정확하다.** 카탈로그는 임계변수를 "과거 수익률 부호·크기 등"이라 적었으나 구현은 수익률 부호를 쓰지 않고 직전 1시간(4봉) 실현변동성 수준 하나만 쓴다(`engine/regime_garch.py` 서두의 임계변수 주석과 `test/models/24_volatility_maxscale_refit_test.py`의 `switch_lagged = pas`를 1칸 민 값, 26번 이후 `tar_switch`). 수익률 부호를 쓰는 것은 Zakoian(1994)의 TGARCH다.
2. **제외된 모델이 "기존 유지"로 적혀 있다.** Linear, Ridge, HAR-RV 행이 모두 "기존 유지 · 적용 연구 24, 25, 26"이다. 실제로는 Linear·Ridge가 2026-09-07 선형성 기각 결정으로, HAR-RV가 2026-10-05 사용자 결정으로 결론에서 제외됐고 26c번 코드에서 삭제됐다. 카탈로그에 제외 사실이 나오는 곳은 4절 DLinear 행의 비고 한 줄뿐이다.
3. **코드가 카탈로그를 가리키지만 카탈로그에 없는 내용이 있다.** `test/models/26c_recent_twopart_test.py`의 주석은 "제외 이유와 복구 방법은 `model_catalog.md`와 이 변경의 커밋 메시지에 있다"고 하고, 27번 보고서도 Linear·Ridge·HAR-RV의 제외 이유가 카탈로그에 있다고 한다. 카탈로그에는 Linear·Ridge·HAR-RV의 제외 결정 원문과 복구 방법이 적혀 있지 않다.
4. **적용 연구 번호가 빠져 있다.** 1~2절의 거의 모든 행이 "24, 25, 26"이다. 코드로 확인한 실제 번호는 GARCH-t가 19, 19c, 23~27, MS-GARCH와 TAR-GARCH가 23(예비검증), 24~27, 선형·커널·트리가 23~27(선형 계열은 26c에서 삭제, 25는 저장 예측만 읽음), HAR-RV가 18, 19, 19c, 23, 24, 26이다.
5. **카탈로그에 없는 모델.** naive(직전 구간 RV), persistence, momentum naive, 상수분산, 롤링분산, EWMA, 수익률 AR(1), GARCH 정규, HAR-RV 분산 수준판과 HAR-RV+, EGARCH·GJR·APARCH·FIGARCH·HARCH(2026-09-07 탐색), 정지 확률 분류기(LightGBM), 유사 국면 검색(K-Means, DTW), 보정 배율과 insanity filter, 시드 평균 앙상블과 gate 융합이 이 문서에 새로 들어갔다.
6. **구현 설명의 차이.** 카탈로그는 GARCH-t를 `arch` 패키지라고만 적었는데 19·19c번은 scipy로 직접 구현한 `fit_garch11_t`를 썼다. 카탈로그의 MS-GARCH 행은 모수화가 둘(원본, 2026-10-06 제약판)이고 24·25번이 분산 척도 오류를 안고 있다는 사실을 적지 않았다.
7. **"분류 문제는 20번 LogisticRegression이 유일하다"는 서술이 낡았다.** 카탈로그 9절의 이 서술은 26c번에서 LightGBM 정지 확률 분류기가 추가되어 맞지 않는다.
8. **LightGBM 저장소 소속.** 카탈로그는 "공식 `lightgbm` 패키지(Microsoft)"라 적었는데 2026-10-14 현재 `microsoft/LightGBM` 주소는 `lightgbm-org/LightGBM`으로 이동(301)한다.
9. **HAR-RV 문구.** 카탈로그는 HAR-RV를 "이기기 어려운 표준 벤치마크"라 소개한다. 문헌상의 평가는 맞지만 이 연구의 결론에서는 기준점으로 쓰지 않으므로(2절), 소개 문구에 이 연구에서의 지위를 함께 적는 편이 안전하다.
10. **환경 정보.** `uv.lock`에 xgboost 3.2.0과 3.4.1, scipy 1.15.3과 1.17.1이 함께 있어 실행 환경이 어느 버전인지 이 문서는 확인하지 못했다.

### B-2. 딥러닝·파운데이션 부분

아래는 이번 조사에서 코드, 패키지 문서, arXiv 초록과 대조해 확인한 사항이다. 저장소 파일은 수정하지 않았다.

| 번호 | 위치 | 내용 | 근거 |
| :--- | :--- | :--- | :--- |
| 1 | 4·5·6절 비고 | 27번이 끝났는데도 PatchTST, iTransformer, TCN, Autoformer, TimesNet, TimeXer, ModernTCN과 파운데이션 7종의 "적용 연구 27(예정)" 표기가 남아 있다. | 27번 결과 보고서와 `process.md`가 실행 완료를 기록한다. |
| 2 | 4절 TCN·Autoformer·TimesNet·TimeXer, 5절 ModernTCN | 자체 구현이 "미사용으로 남아 있던 것"이라고 적었으나 8번(전부), 9·10번(`AutoformerLike`, `TimesNetLike`), 15번(`TCN`, `ModernTCNLike`)에서 실제 사용됐다. | 각 드라이버의 모델 목록(`DEFAULT_MODELS`, `EXPANDED_MODELS`)과 `engine/models.py`의 `make_model`. |
| 3 | 6절 Moirai / Moirai-2.0 행 | 설명(마스크드 인코더, any-variate attention, 다중 패치 투영)과 인용(Woo 외 2024)이 1.0판이다. 이 연구가 쓴 `moirai-2.0-R-small`은 디코더 전용, 단일 패치, 분위수 손실이다. | 2.0판 논문 arXiv:2511.11698 초록과 `27_fm_run.py`. |
| 4 | 6절 TTM 행 | 논문 제목이 현재 arXiv 제목과 다르다(`Fast Pretrained Models for Enterprise Time-series Forecasting` 대 `Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series`). "다른 파운데이션 모델 대비 수백 분의 1" 파라미터라는 수치는 초록에서 확인하지 못했다(초록은 "100만 파라미터부터"). | arXiv:2401.03955 초록 대조. |
| 5 | 7절 MOMENT 행 | "공식 저장소를 특정하지 못함"이라고 적었으나 `moment-timeseries-foundation-model/moment`가 확인(200)이고 README가 논문 arXiv:2402.03885를 연결한다. 미채택 사유가 현재 사실과 다르다. | 저장소 README 확인. |
| 6 | 9절 Informer 행 | "`TransformerModel`과 동일 코드에 dropout만 추가"라고 적었으나 `nn.TransformerEncoderLayer`의 기본 dropout이 0.1이라 2번·3번의 `InformerModel`은 코드상 완전히 같다. | 2번·3번 `ModelZoo` 코드. |
| 7 | 3절 GRU·LSTM 비고 | 적용 연구를 "24, 25, 26"으로 적었다. 25번은 저장된 예측만 분석하고 재학습하지 않았으며(25번 설계 "재적합 없이"), 학습이 있었던 1·2·3·4·5·8번(모두 자체 구성 또는 근사), 23번, 26c번, 27c번이 빠져 있다. | 각 드라이버의 학습 코드와 25번 설계 셀. |
| 8 | 6절 Lag-Llama 행 | "최초의 공개 시계열 파운데이션 모델"이라는 서술을 초록에서 확인하지 못했다. 인용 연도 2024는 arXiv 최초 제출(2023-10-12)과 다르다. | arXiv:2310.08278 초록과 제출일. |
| 9 | 6절 Chronos-Bolt 행 | 원조보다 "수십 배 빠름"이라고 적었으나 패키지 README는 최대 250배 빠르고 20배 적은 메모리라고 적는다(표현 정밀도 문제). | 설치본 `chronos-forecasting` 2.3.2 README. |
| 10 | 6절 TimesFM 행 | 최신판 정보가 없다. 설치된 패키지 README는 TimesFM 3.0(2026-08, 다변량·공변량 지원)이 최신이고 3.0 가중치가 비상업 라이선스라고 적는다. 설명의 "출력 패치 128스텝"은 2.5판에서 대조하지 못했다. | 설치본 `timesfm` 3.0.2 README. |
| 11 | 5절 Mamba / S-Mamba 행 | 한 행에 원형 Mamba와 S-Mamba를 합쳐 적어, 이 연구가 8번 등에서 쓴 `MambaLike`(상태공간 재귀 없는 자체 근사)와 27번의 공식 S-Mamba가 구분되지 않는다. | `engine/models.py`, 27번 코드. |
| 12 | 4절 TimeXer 행 | 설계 취지(외생 교차 어텐션)를 적었으나 27번이 외생 변수 없이 단변량으로 썼다는 사실이 비고에 없다. | `27_model_expansion_test.py`의 `make_nf_model`(`n_series=1`)과 `nf_frame`. |
| 13 | 카탈로그 밖(27번 설계 셀, 보고서 설계 절, `27_model_expansion_test.py` 머리말) | 설계 셀이 ModernTCN 어댑터를 `27_adapters.py`라고 적었으나 이 파일은 저장소에 없고, 어댑터는 `27_model_expansion_test.py`의 `modern_tcn_predict`이다. 설계 셀의 "S-Mamba 설치 실패로 제외" 문장은 결과 보고서 0절이 정정한 대로 현재 사실이 아니다. | `git ls-files`에 없음. 보고서 0절. |

---
