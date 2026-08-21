# 후속 — 변동성(B) 방향에 대한 문헌 근거 보강

작성일 2026-08-21 · 브랜치 `stock`
· 이 문서는 [7월 브리프](professor_brief_publication_case_20260722.md)의 후속입니다.
방향/변동성 논의와 차분 관련 사실은 그 문서에서 이미 전해드렸고, 여기서는 **그 이후
새로 확인한 것만** 다룹니다.

---

## 1. 현재 상태 — 알고리즘 선정을 멈추고 EDA로 되돌아간 이유

지금 1단계(실시간 시계열 분석)의 본격적인 알고리즘·모델 선정 작업을 잠시 멈추고,
다시 초반의 EDA(데이터 탐색) 단계로 돌아와 점검을 수행하고 있습니다.

이유는 7월 브리프에서 이미 말씀드린 그 지점입니다 — **예측 대상을 방향(A)으로 할지
변동성(B)으로 할지가 아직 결정되지 않은 상태**였고, 그 결정이 나기 전에 한쪽(B) 갈래의
세부 모델 선택까지 계속 파고드는 것은 순서가 맞지 않다고 판단했습니다. 그래서 결정이
나올 때까지 세부 모델링은 멈추고, 지금까지 확인한 내용을 재점검·문헌 보강하는 작업만
하고 있습니다.

---

## 2. 그런데 문헌을 찾아보니 — 방향보다 변동성이 학계 정설에 부합

7월 브리프에서 이미 데이터로 확인한 결과(방향 자기상관≈0.05, 정확도 0.48~0.54 /
변동성 자기상관 0.36, GARCH로 0.46)가, **금융계량경제학에서 이미 정설로 굳어진
현상과 정확히 일치**한다는 것을 문헌으로 재확인했습니다. 이전 브리프에는 이 문헌적
뒷받침이 없었어서, 이번에 10편을 확인해 보강했습니다.

### 2.1 고전 4편 — 이 분야 모든 후속 연구의 토대

원문 전문 접근이 제한돼 있어 핵심 주장의 요지만 적습니다(직접 인용 아님).

**[1] Fama, E. F. (1970). "Efficient Capital Markets: A Review of Theory and Empirical
Work."** *Journal of Finance*, 25(2), 383-417.
> 요지: 가격에는 이용 가능한 정보가 이미 반영돼 있어 다음 방향을 미리 알 수 없다는
> **효율적 시장 가설**을 정립.

**[2] Mandelbrot, B. (1963). "The Variation of Certain Speculative Prices."**
*Journal of Business*, 36(4), 394-419.
> 요지: 큰 변동 뒤에 큰 변동이 이어지는 **변동성 군집**을 최초로 관측·보고.

**[3] Engle, R. F. (1982). ARCH 원 논문.** *Econometrica*, 50(4), 987-1007.
> 요지: 변동성이 시간에 따라 예측 가능하게 변한다는 조건부 이분산 모형을 최초 수식화.
> 2003년 노벨 경제학상.

**[4] Cont, R. (2001). "Empirical Properties of Asset Returns: Stylized Facts and
Statistical Issues."** *Quantitative Finance*, 1(2), 223-236.
> 요지: "방향에는 선형 상관이 거의 없고 크기에는 강한 상관·군집이 있다"를 **정형화된
> 사실**로 종합 정리한 표준 레퍼런스.

### 2.2 최신 실증 6편 — 원문 직접 인용

**[5] Yi, E., Yang, B., Jeong, M., Sohn, S., & Ahn, K. (2023). "Market efficiency of
cryptocurrency: evidence from the Bitcoin market."** *Scientific Reports* (Nature), 13, 4789.
> *"Our results show that the Bitcoin market is close to being weakly efficient."*

**[6] Singha, M. (2025). "Hidden Order in Trades Predicts the Size of Price Moves."**
arXiv:2512.15720.
> *"while directional price movements remain largely unpredictable — consistent with
> weak-form efficiency — the magnitude of price changes displays systematic
> structure... directional accuracy remains at 45.0% — statistically indistinguishable
> from chance (p = 0.12)."* (SPY 3,850만 건 체결 데이터)

**[7] Barjašić, I., & Antulov-Fantulin, N. (2020). "Time-varying volatility in Bitcoin
market and information flow at minute-level frequency."** arXiv:2004.00550.
> *"the simplest GARCH(1,1) reacts the best to the addition of external signal to model
> volatility process on out-of-sample data."*

**[8] Hansen, P. R., Kim, C., & Kimbrough, W. (2021). "Periodicity in Cryptocurrency
Volatility and Liquidity."** arXiv:2109.12142.
> *"We find systematic patterns in both volatility and volume across day-of-the-week,
> hour-of-the-day, and within the hour... These patterns have grown stronger over the
> years."*

**[9] Kaur, E. (2026). "The Limits of Conditional Volatility: Assessing Cryptocurrency
VaR under EWMA and IGARCH Models."** arXiv:2601.13757.
> *"The EWMA/IGARCH baseline, characterized by infinite volatility persistence
> (alpha + beta = 1), provided the only robust conditional volatility estimate."*
> (고베타 알트코인 XRP·SOL·ADA 대상)

**[10] Hassler, U., & Pohle, M.-O. (2019). "Forecasting under Long Memory and
Nonstationarity."** arXiv:1910.08202.
> *"Long memory in the sense of slowly decaying autocorrelations is a stylized fact in
> many time series from economics and finance... methods based on fractional
> integration clearly are superior to alternative methods not accounting for long
> memory."*

**요약**: 10편 모두 같은 방향입니다 — 가격 방향은 이론([1])·실증([5][6]) 양쪽에서
예측이 어렵고, 변동의 크기는 이론([2][3])·실증([6][7][8][9][10]) 양쪽에서 예측
가능한 구조를 갖습니다. 7월 브리프의 데이터 결과와 정확히 같은 결론입니다.

---

## 3. 추가 제안 — "변동성 예측"으로 목표를 잡으면 두 활용처 모두에 맞습니다

이 연구의 활용 방향은 두 갈래로 나뉠 수 있습니다.

| | (a) 연구 목표: 비정상성/불확실성 자체를 다루는 연구 | (b) 실용 목표: 트레이딩 보조 도구 |
|---|---|---|
| 무엇을 다루나 | 비정상적인 데이터 자체의 구조를 설명 | 실제 매매 판단에 넣을 수 있는 신호 |

**"변동성을 맞춘다"로 목표를 설정하면 이 둘을 나누지 않고 하나로 갈 수 있습니다.**
(b)로는 변동성 예측이 그대로 리스크 관리 신호가 되고, (a)로는 "변동의 크기가 시간에
따라 변한다"는 비정상성의 구체적인 형태를 다루는 것 자체가 연구가 됩니다.

---

## 4. 정리

- 7월 브리프에서 드린 A(방향)/B(변동성) 결정 요청이 여전히 유효하며, 이번엔 문헌
  10편으로 그 판단에 근거를 보강했습니다.
- 결정이 나기 전까지는 세부 모델링을 멈추고 EDA·문헌 점검만 진행하고 있습니다.
- "변동성 예측"으로 목표를 잡으면 비정상성 연구 목적과 트레이딩 도구 목적 모두에
  쓸 수 있다는 점을 추가로 제안드립니다.
