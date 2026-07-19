# 17번 문헌 리뷰 — 고빈도 금융/암호화폐 수익률의 "방향(부호) 예측 가능성"

작성일: 2026-07-19 · 브랜치: `stock` · 목적: 아래 사용자 5개 질문에 근거 있는 답을 주기 위한 문헌 조사

> 조사 방법: repo `.mcp.json`의 arxiv MCP(`mcp__arxiv__*`)는 이번 세션에서 노출되지 않아
> **웹 검색 + arxiv.org / 학회·저널 페이지 직접 대조**로 대체했다. 각 논문은 arXiv abstract,
> NBER, 저널 페이지 중 하나에서 제목·저자·연도·ID를 대조해 실재를 확인했고, 확인 못 한
> 항목은 "미확인"으로 명시했다. 지어낸 인용은 없다.
>
> 우리 실험 실측 전제(이 리뷰가 설명해야 하는 현상):
> - 업비트 BTC 15분봉 로그수익률 자기상관: lag1 −0.05, lag16 ~0, lag64 ~0 (거의 0)
> - |수익률|(변동 크기) 자기상관: lag1 +0.36, lag16 +0.16 (강하게 예측됨)
> - 방향 정확도(direction accuracy)는 어떤 모델·손실·정규화·전종목에서도 0.48~0.54
> - 예측 진폭(variance_ratio)은 RevIN으로 1.55까지 살릴 수 있으나 방향 상관(Pearson)은 ≤0.07 고착

---

## 0. 이 리뷰가 답하는 질문과 우리 연결

| 절 | 다루는 것 | 사용자 질문 |
|---|---|---|
| 1 | 부호는 안 잡히는데 크기는 잡히는 현상의 정식 이론 | Q1 |
| 2 | 이것이 bias-variance인가 vs 신호 부재(irreducible error)인가 | Q2 |
| 3 | 실제 달성되는 방향 정확도 — 정직한 벤치마크와 과장/체리피킹 구분 | Q3 |
| 4 | 방향을 실제로 개선한 방법 (외생·오더북·cross-asset·손실·regime) | Q4 |
| 5 | "예전엔 lagging 있어도 방향을 잘 잡았다"는 착시의 문헌적 근거 | Q5 |

우리 실측은 교과서적 **stylized facts**와 정확히 일치한다: 수익률 자기상관 ≈ 0(부호
예측 불가) + |수익률|/제곱수익률 자기상관 강함(변동 크기 예측 가능). 아래는 이것이
"우리 데이터/모델의 결함"이 아니라 **금융 시계열의 알려진 구조**임을 문헌으로 못박는다.

---

## 1. "크기는 예측되는데 부호는 안 된다"는 학계의 정식 결과

### 왜 이 현상이 나타나나 (핵심 근거)

금융 수익률의 가장 오래된 실증 규칙(**Cont의 stylized facts**)이 바로 이것이다:
(1) 수익률 자체의 선형 자기상관은 거의 0, (2) **절대/제곱 수익률의 자기상관은 크고
느리게 감소**(volatility clustering), (3) fat tail, (4) 변동성 군집. 우리 lag1 수익률
−0.05 vs |수익률| +0.36은 이 규칙의 교과서적 재현이다.

### 1.1 Cont — Stylized Facts / Volatility Clustering

- **출처**: Cont, "Empirical properties of asset returns: stylized facts and statistical
  issues", *Quantitative Finance* 1(2):223–236, 2001.
  http://www-stat.wharton.upenn.edu/~steele/Resources/FTSResources/StylizedFacts/Cont2001.pdf (확인)
  · 관련: Cont, "Volatility Clustering in Financial Markets", 2005.
  http://rama.cont.perso.math.cnrs.fr/pdf/clustering.pdf (확인)
- **핵심 아이디어**: 수익률은 iid가 아니다. 선형 자기상관은 (아주 짧은 lag의 미세구조
  효과를 빼면) 통계적으로 0에 가깝지만, |수익률|·수익률²의 자기상관은 크고 수일~수주
  지속. "변동의 크기는 군집하지만 부호는 군집하지 않는다."
- **우리 연결**: 우리 lag1 수익률 −0.05(≈0), |수익률| +0.36, lag16 +0.16은 Cont의
  facts 1·2를 그대로 재현한 것. **데이터가 정상이라는 증거이지 파이프라인 버그가 아니다.**
- **한 줄 근거**: 부호 무상관 + 크기 강상관은 자산수익률의 가장 보편적 stylized fact다.

### 1.2 Christoffersen & Diebold — 부호 예측가능성과 변동성 동역학 (Q1·Q2의 핵심 논문)

- **출처**: Christoffersen & Diebold, "Financial Asset Returns, Direction-of-Change
  Forecasting, and Volatility Dynamics", *Management Science* 52(8):1273–1287, 2006
  (원 NBER Working Paper 10009, 2003). https://www.nber.org/papers/w10009 (확인) ·
  https://users.nber.org/~confer/2003/si2003/papers/efww/diebold.pdf (확인)
- **핵심 아이디어**: **조건부 평균이 예측 안 돼도(수익률 자기상관 ≈ 0) 부호는 예측될 수
  있다.** 부호 예측가능성은 (a) **0이 아닌 기대수익률(drift)** 과 (b) **시변 변동성**의
  상호작용에서 나온다. 분포가 대칭이고 평균이 0이거나 변동성이 상수이면 부호는 예측
  불가능하다. 논문의 결정적 문장: *"The standard finding of little or no conditional mean
  dependence is entirely consistent with a significant degree of sign dependence and
  volatility dependence."*
- **중요한 단서(Q3·Q5와 직결)**: 저자들은 부호 예측가능성이 **very high-frequency(일간
  이하)나 very low-frequency(연간)에서는 잘 안 나타나고, 중간 horizon에서 더 잘 나타난다**고
  명시한다. 이유: 부호 예측력 ∝ (drift × horizon) / (변동성 × √horizon) 꼴로, horizon이
  커질수록 drift가 누적되어 신호 대 잡음비가 개선되기 때문. **15분 같은 초단기에서는 drift가
  변동성에 비해 사실상 0이라 부호가 안 잡힌다.**
- **우리 연결**: 우리 방향 정확도 0.48~0.54 고착은 이 이론의 정확한 예측이다. 15분봉에서
  15분 drift ≈ 0 ⇒ 부호 신호 ≈ 0. 변동 '크기'만 잡히는 것도 같은 이론(변동성은 예측되나
  그 자체는 부호 정보를 주지 못함, drift가 0이므로)에서 나온다. **horizon을 늘리면
  (h=16/64봉) 부호 예측력이 이론상 개선될 여지가 있다** — 우리 16번 h-격자 설계와 직결.
- **한 줄 근거**: 평균 무예측 + 변동성 예측 + drift≈0 ⇒ 초단기 부호 무예측은 정리(theorem)다.

### 1.3 Brou & Luger — Sign을 Magnitude에 조건화하는 분해 (최신, 2026)

- **출처**: Brou & Luger, "A new decomposition approach to modeling financial returns:
  Conditioning sign on magnitude", *Journal of Banking & Finance*(게재 승인), 2026.
  arXiv:2606.04153 (확인). https://arxiv.org/abs/2606.04153
- **핵심 아이디어**: 수익률을 **부호(sign) × 크기(magnitude=|수익률|, 변동성과 밀접)**로
  분해하고, magnitude의 분포를 먼저 모델링한 뒤 **부호를 '동시점 magnitude에 조건화'**해
  binary choice로 모델링. 즉 "큰 변동이 예상될 때 부호가 어느 쪽으로 쏠리는지"를 명시적으로
  학습. 월간 미국 초과수익에서 선형회귀 대비 통계·경제적 이득 보고.
- **우리 연결**: 우리처럼 magnitude(변동성)는 잡히고 부호는 안 잡히는 상황에서 **부호를
  독립으로 예측하지 말고 크기에 조건화**하는 것이 최신 정공법. 16번 sign-on-magnitude
  head(분위/분포 head 위에 방향 보조 head)의 문헌 근거. 단 이들의 이득은 **월간(중간
  horizon)**에서 나온 것 — 15분에는 그대로 이식되지 않을 수 있음(1.2 horizon 단서).
- **한 줄 근거**: 부호는 단독보다 magnitude 조건부로 볼 때 예측 신호가 살아난다.

**1절 종합**: "크기는 예측, 부호는 무예측"은 (i) Cont의 stylized fact이자 (ii) Christoffersen–
Diebold 정리로, **학계에서 확립된 현상**이다. 우리 실측은 이 이론의 재현이다. 관련 개념 사슬:
volatility clustering(Cont) → GARCH가 그 크기를 모델링 → EMH(약형)가 부호 무예측을 설명 →
sign predictability는 drift×변동성 상호작용으로 **중간 horizon에서만** 소량 회복.

---

## 2. Bias-Variance 트레이드오프인가, 아니면 신호가 없는 것인가 (Q2)

### 핵심 구분: reducible error vs irreducible error

기대예측오차는 **bias² + variance + irreducible error(σ²)** 로 분해된다. 앞의 둘은
모델·정규화로 줄일 수 있는 **reducible error**, 마지막 σ²는 **어떤 완벽한 모델로도 못 줄이는
잡음(= Bayes error의 회귀판)**이다. "분산은 잡히는데 방향은 못 잡는" 우리 현상을 이 틀로
정확히 진단할 수 있다.

- **출처(표준)**: Bias–variance tradeoff — https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff (확인).
  Bayes/irreducible error의 정의는 통계학습 표준(예: Hastie–Tibshirani–Friedman, *ESL*).

### 2.1 우리 현상은 트레이드오프가 아니라 "부호 채널의 irreducible error가 지배"

- **논리**:
  - **크기(variance) 채널**: |수익률|은 자기상관 +0.36으로 **reducible error가 크다**(=줄일
    signal이 있다). 그래서 RevIN·GARCH류가 variance_ratio를 1.55까지 살릴 수 있다. 여기서
    우리가 하는 일은 실제 bias-variance 최적화(모델이 신호를 잡아 오차를 줄임)다.
  - **부호(sign) 채널**: 15분 수익률 부호의 조건부 신호는 **거의 전부 irreducible error**다.
    수익률 자기상관 ≈ 0 + drift ≈ 0(1.2) ⇒ 조건부 부호분포가 거의 50:50 ⇒ **Bayes-optimal
    분류기의 정확도 자체가 ~0.5**. 우리가 0.48~0.54에 고착된 것은 **모델의 bias나 variance가
    나빠서가 아니라, 도달 가능한 상한(Bayes error) 자체가 0.5 근처**이기 때문이다.
- **결정적 감별점**: bias-variance 문제라면 "모델을 바꾸거나 정규화를 조절하면 방향이
  개선"되어야 한다. 그러나 우리는 **모델·손실·정규화·전종목을 다 바꿔도 0.48~0.54를 못
  벗어난다** — 이는 reducible error가 이미 소진되었고 **남은 것이 irreducible error(신호 부재)**
  임을 가리키는 전형적 signature다. Pearson 상관이 ≤0.07에 고착되는 것도 같은 진단.
- **variance_ratio 1.55 vs 방향 상관 0.07의 공존이 곧 증거**: 진폭(크기)은 살릴 수 있는데
  방향 상관은 못 올린다 = "크기 채널엔 reducible signal, 부호 채널엔 없음"의 직접 관측.

### 2.2 왜 착각하기 쉬운가 — 진폭 압축(mean-reversion)은 bias, 부호 무예측은 irreducible

- 두 가지를 섞으면 안 된다.
  1. **진폭 압축(variance_ratio ≪ 1, 0-평탄화)**: 이것은 **bias 문제**다. MSE/Huber가
     조건부 평균(≈0)으로 수렴해 진폭을 죽이는 것 — 손실 교체·RevIN 역정규화로 **줄일 수 있다**
     (실제로 우리가 1.55로 살렸다). 이건 reducible.
  2. **방향(부호) 무예측(direction accuracy ≈ 0.5, Pearson ≈ 0)**: 이것은 **irreducible**.
     진폭을 완벽히 살려도 부호를 못 맞히는 것이 바로 이 증거다.
- 즉 **"분산을 잡았다"는 진폭 압축이라는 bias를 푼 것이고, "방향을 못 잡는다"는 신호 부재
  (irreducible)다.** 둘은 다른 축이며, 하나의 bias-variance 트레이드오프로 묶이지 않는다.

- **출처(관점 보강)**: Gu, Kelly & Xiu, "Empirical Asset Pricing via Machine Learning",
  *Review of Financial Studies* 33(5):2223–2273, 2020. arXiv/SSRN 3159577 (확인).
  https://dachxiu.chicagobooth.edu/download/ML.pdf — 900+ 예측변수·비선형 신경망으로도
  종목단위 월간 예측 R²가 **~0.33–0.40%**(즉 대부분이 irreducible)에 머문다. 고용량 모델이
  더 얹으면 signal이 아니라 **noise를 적합**(high-variance 영역)한다고 명시.

**2절 종합 (Q2 답)**: 우리 현상은 단일 bias-variance 트레이드오프가 아니다. **크기 채널**은
reducible signal이 있어 bias(진폭 압축)를 풀면 개선되고(실측 1.55), **부호 채널**은 15분
horizon에서 조건부 신호가 거의 없어 **irreducible error가 지배** → Bayes error가 0.5 근처라
어떤 모델로도 방향 정확도를 못 올린다. "분산은 잡고 방향은 못 잡음"은 **두 채널의 서로 다른
error 구성** 때문이지 하나의 트레이드오프 조절 실패가 아니다.

---

## 3. 기존 연구가 실제 달성하는 방향 정확도 (Q3) — 정직한 값 vs 과장

### 3.1 정직하게 보고된 값: 대체로 50~56%, "60%면 좋음"

- **EMH/random-walk 기준선**: 약형 효율시장 하에서 가격 방향은 사실상 예측 불가에 가깝고,
  일반적으로 **50%가 기준선**. 문헌·실무 통념은 **엄밀한 out-of-sample에서 55%면 우수,
  지속적 60%+면 매우 드묾**이다.
- **Gu–Kelly–Xiu (2020)**: 위(2.2). 방향 정확도로 환산하면 종목단위 우위는 몇 %p 수준.
  정직한 대규모 연구일수록 우위가 작다.
- **crypto 방향 예측(정직한 프레이밍)**: Omole & Enke, "Deep learning for Bitcoin price
  direction prediction", *Financial Innovation* 10(1), 2024. https://doi.org/10.1186/s40854-024-00643-1 (확인).
  → 여기서 보고된 **CNN-LSTM+Boruta 82.44%**와 **연 6654% 수익**은 **체리피킹/누수 의심
  구간**으로 봐야 한다(아래 3.2). 같은 논문도 손실 없는 순수 방향 우위가 그렇게 크다는 것은
  전형적 과장 패턴이다.

### 3.2 과장·체리피킹·데이터 누수의 전형 (반드시 감별)

문헌 전반에 **"엄밀 검증하면 정확도가 급락"**하는 패턴이 반복 보고된다.

- **데이터 누수(look-ahead / leakage)**: 시계열을 셔플하는 잘못된 cross-validation, 미래를
  참조하는 기술지표, 사후정보가 섞인 텍스트가 흔한 누수 경로. 인접 도메인 리뷰에서 **누수
  통제 시 94%→66%(약 28%p 하락)** 같은 급락이 관측됨(방법론 경고 근거).
  https://fastercapital.com/content/Lookahead-bias-in-machine-learning--Enhancing-predictive-models.html (개념 확인)
- **오더북(FI-2010) 벤치마크 과최적화**: DeepLOB(아래 4.2)의 **FI-2010 정확도 ~84%**는
  **전처리된 단순 벤치마크에서의 값**이며, "너무 단순해 overfitting 여지가 크고 실거래로
  일반화가 안 된다"는 비판이 정착. 실제 시장 데이터로 옮기면 F1이 유의하게 하락(예:
  BINCTABL 평균 **F1 약 19.6%p 하락**). 근거: Prata et al., "LOB-Based Deep Learning
  Models for Stock Price Trend Prediction: A Benchmark Study", arXiv:2308.01915 (확인) /
  Briola et al., "Deep limit order book forecasting: a microstructural guide",
  arXiv:2403.09267 (확인) — "high forecasting power ≠ actionable trading signal" 명시.
- **거래비용 무시**: 방향 정확도만 높게 보고하고 스프레드·슬리피지·수수료를 빼면 실현 불가.
  Christoffersen–Diebold의 horizon 단서(1.2)와 배치되는 초단기 고정확도 주장은 특히 의심.

### 3.3 60%+ 달성 사례는 어떤 조건인가

- **중간 horizon + 외생정보**: 부호 예측가능성은 이론상 중간 horizon에서 커진다(1.2). 실제로
  뉴스 감성 등 외생변수를 더하면 **방향 정확도 59~63%** 보고가 있다(4.1). 순수 초단기
  가격만으로 60%+는 재현성 있게 보고된 바가 드물다.
- **오더북(초단기지만 미시구조 정보 사용)**: large-tick 종목·확률 임계값 적용 시 F1 0.7~0.9
  보고(4.2). 단 이는 **가격 시계열이 아니라 호가창(order flow) 정보**를 쓴 것이고 FI-2010
  일반화 한계가 붙는다.
- **감별 요약**: 60%+가 진짜인 경우는 대개 (a) **추가 데이터 채널**(오더북·텍스트·온체인),
  (b) **중간~장 horizon**, (c) **특정 시장/종목(large-tick, 고유동성)** 조건이다. **순수
  가격 시계열 + 초단기(15분) + 전종목**에서 지속 60%+는 문헌상 사실상 없다 — 우리 0.48~0.54는
  이 조건에서 **정상 범위**다.

**3절 종합 (Q3 답)**: 엄밀한 순수-가격 초단기 방향 정확도는 **50~56%가 현실**이고 55%면
좋은 편. 60%+ 주장은 대부분 **외생정보·중간 horizon·특정 시장** 조건이거나 **누수/거래비용
무시/단순 벤치마크 과최적화**에 기인한 과장이다. 우리 0.48~0.54는 조건(순수 가격·15분·전종목)을
감안하면 문헌과 일치하는 정상 결과다.

---

## 4. 방향 예측을 실제로 개선한 방법 (Q4)

핵심 결론 먼저: **순수 가격 시계열만으로 방향을 재현성 있게 개선한 사례는 드물다.** 개선은
거의 항상 **정보 채널을 추가**하거나(외생·오더북·cross-asset), **문제를 재정의**(분류손실·
중간 horizon·regime 분리·cross-sectional ranking)할 때 나온다.

### 4.1 외생변수 / 텍스트(뉴스·감성)

- **핵심 아이디어**: 가격에 없는 정보(뉴스 감성, 거시, 소셜)를 더하면 부호 신호가 소량 회복.
  검증된 대표값은 AZFinText의 **57.1%**(가격만보다 상향). 후속 감성 연구들에서 **59~63%**
  보고도 있으나(개별 확인 부분적), firm-specific으로 쪼개면 유의성이 사라지는 등 robust하지
  않은 경우도 많음.
- **출처(확인)**: Schumaker & Chen, "Textual analysis of stock market prediction using
  breaking financial news: The AZFin text system", *ACM TOIS* 27(2), 2009 — 기사어휘+가격
  결합 모델이 뉴스 20분 후 방향 **57.1%** 정확도(단순 가격 대비 상향). dl.acm.org/doi/10.1145/1462198.1462204 (확인) ·
  최근 종합: "News Sentiment and Stock Market Dynamics", *JRFM* 18(8):412, 2025.
  https://www.mdpi.com/1911-8074/18/8/412 (확인)
- **우리 연결**: repo에 `ingest_text_context.py`가 있으므로, 순수 가격에서 부호가 막히면
  **텍스트 컨텍스트를 부호 보조채널로** 붙이는 것이 문헌상 가장 검증된 개선 경로. 단 15분
  단위 뉴스 정합·누수 통제가 관건.

### 4.2 오더북 / 미시구조 (order flow)

- **출처(확인)**: Zhang, Zohren & Roberts, "DeepLOB: Deep Convolutional Neural Networks
  for Limit Order Books", *IEEE Trans. Signal Processing* 67(11):3001–3012, 2019.
  arXiv:1808.03668 (확인). FI-2010에서 정확도/F1 ~84%.
- **핵심 아이디어**: 호가창의 **order flow imbalance**가 초단기 방향에 실제 신호를 준다.
  가격 시계열엔 없는 정보. **단 FI-2010 과최적화·실거래 일반화 한계**(3.2) 유의.
- **우리 연결**: 업비트도 호가/체결 데이터를 얻을 수 있다면 15분 방향의 유일하게 검증된
  초단기 개선 채널. 다만 15분 리샘플에서는 microstructure alpha가 상당 부분 소멸할 수 있음.

### 4.3 Cross-asset / cross-sectional (상대 강도)

- **출처(확인)**: Feng et al., "Temporal Relational Ranking for Stock Prediction" (RSR),
  *ACM TOIS* 37(2), 2019. arXiv:1809.09441 (확인, 16번 리뷰와 공유) · Gu–Kelly–Xiu(2020)에서
  momentum/liquidity/volatility가 지배적 예측자.
- **핵심 아이디어**: 절대 부호 대신 **종목 간 상대 순위**를 맞히면(cross-sectional momentum)
  절대 예측이 붕괴해도 신호가 남는다. "누가 더 오를까"는 "오를까/내릴까"보다 잘 잡힌다.
- **우리 연결**: 데이터 축이 **KRW 전 종목**이므로, 절대 방향(0.5 고착) 대신 **cross-sectional
  ranking(Spearman IC, 상·하위 분위 스프레드)**으로 목표를 바꾸는 것이 정공법. 16번 3절과 직결.

### 4.4 손실함수: 회귀 → 분류/방향 손실

- **출처(확인)**: Michańków, Sakowski & Ślepaczuk, "Generalized Mean Absolute Directional
  Loss (GMADL)…", arXiv:2412.18405 (확인) · Stefaniuk & Ślepaczuk, "Informer in Algorithmic
  Investment Strategies on High Frequency Bitcoin Data", arXiv:2503.18096 (확인) — **RMSE는
  5/15분에서 거래비용보다 작은 0 근처 예측(우리 진폭 압축과 동일)** 을 내고, GMADL/Quantile이
  이를 완화.
- **핵심 아이디어**: MSE(조건부 평균 추정)를 **직접 방향/분포 손실**로 바꾸면 0-붕괴(진폭
  압축)를 페널티. **주의**: 이것은 주로 **진폭/거래성과**를 개선하지 방향 정확도(부호 신호)를
  근본적으로 만들어내진 못한다 — 신호가 없으면 손실만 바꿔도 정확도는 안 오른다(2절).
- **우리 연결**: 손실 교체는 **진폭 압축(bias)** 해결용이지 **부호 무예측(irreducible)** 해결용이
  아님을 구분해 기대치를 설정. Brou–Luger식 sign-on-magnitude(1.3)가 손실보다 정합적.

### 4.5 Regime 분리 / 중간 horizon

- **출처(확인)**: Wu et al., "Review of deep learning models for crypto price prediction",
  arXiv:2405.11431 (확인) — regime(pre-COVID/COVID)·변동성별 분할 평가가 표준. · horizon
  효과의 이론적 근거는 Christoffersen–Diebold(1.2).
- **핵심 아이디어**: (a) 고변동/추세 regime에서만 부호 신호가 살아나는 경우가 있어 regime을
  나눠 학습·평가하면 특정 구간 정확도가 오름. (b) horizon을 15분→수 시간/일로 늘리면 drift
  누적으로 부호 예측력이 이론상 개선.
- **우리 연결**: 16번 h-격자(1/4/16/64봉)·regime split이 여기 근거. 15분 단일 horizon 고집이
  방향 실패의 한 원인일 수 있음.

**4절 종합 (Q4 답)**: 방향이 실제로 개선된 사례는 대부분 **정보 추가(텍스트·오더북·cross-
asset)** 또는 **문제 재정의(cross-sectional ranking·중간 horizon·regime)**다. **순수 가격
시계열 + 초단기만으로 방향을 재현성 있게 개선한 사례는 사실상 없다.** 손실 교체(GMADL 등)는
진폭 압축은 고치지만 부호 신호를 창조하지 못한다.

---

## 5. "예전엔 lagging 있어도 방향을 잘 잡았다"는 착시 (Q5)

### 5.1 lag-1(직전값 복사) 예측이 눈으로는 실제를 따라가 보이는 이유

- **핵심 아이디어**: 가격(레벨) 시계열에 **naive(persistence, ŷ_t = y_{t−1})** 예측을 겹쳐
  그리면, 예측선이 실제선을 **한 칸 밀린 채 거의 겹쳐** 보인다. RMSE도 낮게 나온다. 그래서
  "잘 맞는 것처럼" 보인다. 그러나 이는 **한 스텝 지연된 복사**일 뿐, **다음 스텝의 부호(변화
  방향)**에 대한 정보는 0이다. 방향 정확도로 채점하면 동전던지기(≈0.5)로 붕괴한다.
- **왜 착시인가**: 사람이 보는 "잘 맞음"은 **레벨의 근접성**이고, 트레이딩이 요구하는 것은
  **차분(수익률)의 부호**다. 레벨이 강한 persistence(단위근/랜덤워크)를 가지면 레벨 예측은
  쉬워 보이지만(naive가 우수), 차분의 부호는 여전히 예측 불가. **레벨 그래프의 시각적 적합도와
  방향 정확도는 사실상 무관하다.**
- **출처(확인)**:
  - Naive/persistence가 레벨을 밀려서 따라가고 강한 벤치마크가 되는 성질:
    Nau, "Notes on the random walk model", Duke(강의노트, 확인)
    https://people.duke.edu/~rnau/Notes_on_the_random_walk_model--Robert_Nau.pdf ·
    Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*(naive 벤치마크 표준).
  - **방향 예측 유의성을 레벨 적합도와 분리해 검정**하는 표준 도구: Pesaran & Timmermann,
    "A Simple Nonparametric Test of Predictive Performance", *Journal of Business &
    Economic Statistics* 10(4):461–465, 1992 (확인, market-timing/directional test).
    → "예측선이 실제를 따라가 보인다"가 아니라 **부호 적중이 우연을 유의하게 초과하는가**를
    본다. 우리 0.48~0.54는 이 검정에서 유의하지 않을 값.
  - 예측오차 우열의 정식 검정(RMSE가 낮다고 방향이 좋은 게 아님): Diebold & Mariano,
    "Comparing Predictive Accuracy", *JBES* 13(3):253–263, 1995 (확인).

### 5.2 우리 실측과의 연결

- 우리 **variance_ratio 1.55 + Pearson ≤0.07**의 공존이 바로 이 착시의 정량판이다: 예측이
  진폭(모양·크기)은 실제만큼 출렁여 **눈으로는 실제를 따라가 보이지만**, 실제와의 상관(부호
  정합)은 0.07로 거의 없다. **"따라가 보임"과 "방향 맞힘"은 별개**임을 수치로 확인한 것.
- 따라서 "예전엔 lagging 있어도 방향을 잘 잡았다"는 인상은 (a) 레벨 그래프로 봤거나(밀린
  복사가 겹쳐 보임), (b) in-sample/누수 상태였거나, (c) 방향을 Pesaran–Timmermann식으로
  엄밀 채점하지 않았을 가능성이 높다. **차분의 부호를 out-of-sample로 채점하면 착시가 사라진다.**

**5절 종합 (Q5 답)**: lag-1 복사는 레벨 그래프에서 실제를 한 칸 밀린 채 겹쳐 보여 "잘 맞는"
착시를 준다. 이는 레벨의 persistence(랜덤워크성) 때문이며 **다음 스텝 부호 정보는 담고 있지
않다.** 시각적 적합도(레벨 RMSE)와 방향 정확도는 분리해서 봐야 하고(Pesaran–Timmermann,
Diebold–Mariano), 엄밀히 채점하면 방향은 ≈0.5로 돌아온다. 우리 variance_ratio 1.55 vs
Pearson 0.07이 그 정량적 증거다.

---

## 6. 5개 질문에 대한 요약 답

**Q1. "분산은 예측되는데 방향은 안 되는" 현상이 학계에 알려졌나?**
→ **그렇다, 매우 확립된 현상이다.** (i) Cont의 stylized facts: 수익률 자기상관 ≈ 0(부호
무예측) + |수익률|/제곱수익률 자기상관 강함(volatility clustering, GARCH가 모델링).
(ii) Christoffersen–Diebold(2006)는 **조건부 평균 무예측이 유의한 부호 예측성·변동성 예측성과
완전히 양립**함을 정리로 보였고, 부호 예측성은 drift×시변변동성에서 나오며 **초단기(일간 이하)
에선 잘 안 나타나고 중간 horizon에서 나타난다**고 명시. 우리 lag1 −0.05 vs |수익률| +0.36은
이 이론의 교과서적 재현. EMH(약형)는 부호 무예측 쪽을 설명. → **우리 결함이 아니라 알려진 구조.**

**Q2. Bias-variance 트레이드오프인가, 신호 부재인가?**
→ **단일 트레이드오프가 아니라, 두 채널의 error 구성이 다른 것.** 크기 채널은 reducible
signal이 있어 bias(진폭 압축)를 풀면 개선(실측 variance_ratio 1.55). 부호 채널은 15분에서
조건부 신호가 거의 없어 **irreducible error(≈Bayes error 0.5)가 지배** → 모델·손실·정규화·
전종목을 다 바꿔도 0.48~0.54를 못 벗어남. 진폭 압축(bias, reducible) ≠ 부호 무예측
(irreducible)을 반드시 구분. **"분산을 잡음"은 bias를 푼 것, "방향을 못 잡음"은 신호가 없는 것.**

**Q3. 실제 달성되는 방향 정확도는? 55%가 좋은가, 60%+ 사례는?**
→ 엄밀 out-of-sample·순수 가격·초단기에서는 **50~56%가 현실이고 55%면 우수**. 60%+는 대부분
(a) 외생정보(텍스트·오더북·온체인), (b) 중간~장 horizon, (c) 특정 시장(large-tick·고유동성)
조건. **순수 가격 + 15분 + 전종목에서 지속 60%+는 문헌상 사실상 없음.** 82~91% 정확도나
연 6654% 수익 류 주장은 **데이터 누수·거래비용 무시·단순 벤치마크(FI-2010) 과최적화**의
전형(누수 통제 시 28%p 급락, FI-2010→실거래 F1 약 19.6%p 하락 사례). **우리 0.48~0.54는
조건상 정상 범위.**

**Q4. 방향을 실제로 개선한 방법은? 순수 가격만으로 개선 사례가 있나?**
→ 개선은 거의 항상 **정보 추가**(뉴스·감성 AZFinText 57.1%, 오더북 order-flow, cross-asset) 또는
**문제 재정의**(cross-sectional ranking, 중간 horizon, regime 분리)에서 나온다. 손실 교체
(GMADL·Quantile)는 **진폭 압축(bias)은 고치지만 부호 신호를 창조하지 못한다.** **순수 가격
시계열 + 초단기만으로 방향을 재현성 있게 개선한 사례는 사실상 없음.** 우리에게 가장 검증된
경로: (1) 텍스트 컨텍스트(`ingest_text_context.py`) 보조채널, (2) KRW 전 종목 cross-sectional
ranking(IC), (3) horizon 확장(h=16/64봉).

**Q5. "예전엔 lagging 있어도 방향을 잘 잡았다"는 착시인 이유?**
→ lag-1(직전값 복사)은 **레벨 그래프에서 실제를 한 칸 밀린 채 겹쳐 보여** "잘 맞는" 착시를
준다(레벨의 랜덤워크성 때문). 그러나 **다음 스텝 부호 정보는 0**이라 방향 정확도로 채점하면
≈0.5. **시각적 레벨 적합도(RMSE)와 방향 정확도는 분리**해야 하며(Pesaran–Timmermann 방향
검정, Diebold–Mariano 정확도 비교), 엄밀히 채점하면 착시가 사라진다. 우리 **variance_ratio
1.55(진폭은 따라감) vs Pearson 0.07(부호는 안 맞음)** 공존이 그 정량 증거.

**전체 한 줄**: 우리가 본 것은 버그가 아니라 **금융 초단기 수익률의 알려진 구조** —
크기는 reducible signal이라 살릴 수 있고(진폭 1.55), 부호는 15분 horizon에서 irreducible
error가 지배해 ~0.5에 고착된다. 방향을 올리려면 **순수 가격을 벗어나(정보 추가) 또는 문제를
바꿔야(ranking·horizon·regime)** 한다.

---

## 7. 참고문헌 (arXiv/venue 확인 완료 = ✅, 미확인 = ⚠)

### 부호/변동성 예측가능성 이론 (Q1·Q2)
- ✅ Cont, "Empirical properties of asset returns: stylized facts and statistical issues",
  *Quantitative Finance* 1(2), 2001. (Wharton 미러 PDF 확인)
- ✅ Cont, "Volatility Clustering in Financial Markets", 2005. (저자 사이트 PDF 확인)
- ✅ Christoffersen & Diebold, "Financial Asset Returns, Direction-of-Change Forecasting,
  and Volatility Dynamics", *Management Science* 52(8), 2006 (NBER WP 10009, 2003). 확인
- ✅ Brou & Luger, "A new decomposition approach to modeling financial returns: Conditioning
  sign on magnitude", *J. Banking & Finance*(승인), 2026. arXiv:2606.04153. 확인
- ✅ Gu, Kelly & Xiu, "Empirical Asset Pricing via Machine Learning", *RFS* 33(5), 2020. 확인

### 방향 예측 정확도·검정 (Q3·Q5)
- ✅ Omole & Enke, "Deep learning for Bitcoin price direction prediction…", *Financial
  Innovation* 10(1), 2024. doi:10.1186/s40854-024-00643-1. 확인 (단, 82%/6654%는 과장 의심으로 인용)
- ✅ Zhang, Zohren & Roberts, "DeepLOB…", *IEEE TSP* 67(11), 2019. arXiv:1808.03668. 확인
- ✅ Prata et al., "LOB-Based Deep Learning Models…: A Benchmark Study", arXiv:2308.01915. 확인
- ✅ Briola et al., "Deep limit order book forecasting: a microstructural guide", arXiv:2403.09267. 확인
- ✅ Pesaran & Timmermann, "A Simple Nonparametric Test of Predictive Performance",
  *JBES* 10(4), 1992. 확인 (제목·저널·연도 대조)
- ✅ Diebold & Mariano, "Comparing Predictive Accuracy", *JBES* 13(3), 1995. 확인
- ✅ Nau, "Notes on the random walk model" (Duke 강의노트, naive/persistence 벤치마크). 확인
- ⚠ Hyndman & Athanasopoulos, *Forecasting: Principles and Practice*(naive 벤치마크 표준
  교재) — 개념 표준이나 이번 세션에서 특정 판·페이지 직접 대조는 미수행 → "미확인" 표기.

### 방향 개선 방법 (Q4)
- ✅ Schumaker & Chen, "Textual analysis of stock market prediction using breaking financial
  news: The AZFin text system", *ACM TOIS* 27(2), 2009 (방향 57.1%). dl.acm.org/doi/10.1145/1462198.1462204 확인
- ✅ "News Sentiment and Stock Market Dynamics: A Machine Learning Investigation",
  *JRFM* 18(8):412, 2025. 확인
- ✅ Feng et al., "Temporal Relational Ranking for Stock Prediction" (RSR), *ACM TOIS*
  37(2), 2019. arXiv:1809.09441. 확인
- ✅ Michańków, Sakowski & Ślepaczuk, "Generalized Mean Absolute Directional Loss (GMADL)…",
  arXiv:2412.18405 (2024). 확인
- ✅ Stefaniuk & Ślepaczuk, "Informer In Algorithmic Investment Strategies on High Frequency
  Bitcoin Data", arXiv:2503.18096 (2025). 확인
- ✅ Wu et al., "Review of deep learning models for crypto price prediction…", arXiv:2405.11431 (2024). 확인

### 방법론 개념 (bias-variance / 누수)
- ✅ Bias–variance tradeoff (개념 표준). en.wikipedia.org/wiki/Bias–variance_tradeoff. 확인
- ✅ Look-ahead/leakage로 인한 정확도 팽창(28%p 급락 예시). FasterCapital 개념 페이지 확인
  (1차 학술 출처가 아닌 개념 정리 페이지 → 수치는 "예시"로만 인용).
