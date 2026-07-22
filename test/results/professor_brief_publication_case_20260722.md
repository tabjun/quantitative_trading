# 교수님 공유용 — 연구 정리 및 논문화 근거

작성일: 2026-07-22 · 브랜치: `stock` · 작성: 연구 진행 중간 정리(초안)

> 목적: 지금까지의 연구를 교수님께 설명드리고 "이걸 논문으로 쓸 수 있는가"를 논의하기 위한
> 정리 문서입니다. 연구 목적 → 주요 초점 → 분석 과정 → 결과 → 방법론(사용 데이터) 순으로
> 정리하고, 마지막에 baseline·유사 지표를 쓴 참고문헌을 붙였습니다.

---

## 1. 연구 목적

업비트 원화(KRW) 마켓 암호화폐 15분봉을 대상으로, **비정상(non-stationary) 시계열의 추세를
딥러닝으로 얼마나 잡을 수 있는가**를 연구합니다. 세부 목적은 세 가지의 결합입니다.

1. **학습 건전성**: 최적화·손실함수가 쉬운 해(0 수렴·직전값 복사)로 붕괴하지 않고, 과적합이
   관리되며, 비정상성 가정을 충족하는 모델을 만든다.
2. **전체 변동 폭의 예측**: 작은 폭만이 아니라 큰 변동까지 잡는다(정확도 단독이 아니라 큰 변동
   구간의 정밀도·재현율로 평가).
3. **하방 방어(MDD)는 생존 제약**: 예측을 매매로 옮길 때 최대낙폭을 함께 지키되, 그 자체가
   목적을 대체하지는 않는다.

## 2. 주요 초점 (무엇을 새로 밝히려 하는가)

- 가격 수준(레벨)이 아니라 **수익률/추세**를 대상으로, 고빈도(15분) 암호화폐에서 딥러닝
  예측의 **실제 한계와 그 구조적 원인**을 체계적으로 규명한다.
- 특히 **"방향(부호)은 예측 안 되지만 변동성(크기)은 예측된다"**는 구조를 다모델·다손실·
  전종목 규모로 실증하고, 그 위에서 어떤 재정의(변동성 타깃·외생정보·cross-sectional)가
  실제로 도움이 되는지 가른다.

## 3. 분석 과정 (실험 1~17번의 흐름)

| 단계 | 실험 | 한 일 | 결과 |
|---|---|---|---|
| 기초 | 1~3 | RNN/LSTM/GRU/Transformer/Autoformer 등 벤치마크, 다종목 | 가격 레벨 예측은 lag-copy 착시로 좋아 보임 |
| 진단 | 5~9 | 최적화·손실·전처리·정규화 격자, 붕괴 진단 | 두 붕괴 모드 규명: **진폭 압축**(0으로 평탄화) vs **분산 폭주** |
| 목적함수 | 10~11 | objective/앙상블 재설계, 위험 분포 예측 | 붕괴 완화하나 방향은 여전히 persistence 미달 |
| 변수·융합 | 12~14 | feature group ablation, fusion 결함 교정 | multi-timeframe이 상대적으로 유리하나 신호 약함 |
| target 재정의 | 15 | "다음 15분" → "h시간 누적 추세"로 변경 | 진폭은 부분 회복(variance_ratio 0.36), 방향 상관은 여전히 ~0.07 |
| 비정상성·손실·cross-sec | 16 | RevIN·분위/분포/tail 손실·전종목 | 진폭 회복, 방향은 종목 가로질러 일반화 안 됨 |
| **탐색적 분석** | **17** | **정상성 검정·분해·자기상관·상호정보량 전면 EDA** | **방향 무신호·변동성 군집을 데이터로 확정** |

## 4. 결과 (핵심)

1. **가격 레벨은 비정상**(ADF p=0.65, KPSS p=0.01 일치), **수익률은 평균-정상이나 분산은
   비정상**(ARCH-LM p=0, 변동성 군집).
2. **방향(부호) 자기상관 ≈ 0**(lag1 -0.05, lag16 0.00) → 순수 가격으로 방향 예측 불가.
   **크기(|수익률|) 자기상관은 0.36→0.12로 느리게 감소** → 변동성은 예측 가능.
3. 이 구조 때문에 다모델·다손실·전종목 어디서도 방향 정확도는 0.48~0.54(동전던지기)에 머문다.
   이는 우리 모델의 결함이 아니라 **효율시장 하 고빈도 수익률의 알려진 성질**이다(6절 문헌).
4. RevIN·분위 손실로 예측 **진폭**은 실제 수준까지 회복 가능하나, 그 진폭이 올바른 **방향**을
   갖게 만들지는 못한다(진폭≠타이밍).

**논문 기여로서의 의미**: 성능이 높아야 논문이 되는 게 아니라 **기여가 있으면** 된다. 본 연구는
(a) 고빈도 crypto 방향 예측의 한계를 KRW 전종목·다모델·다손실로 체계적으로 실증한 negative
result, (b) 진폭 압축/분산 폭주 두 붕괴 모드의 진단과 교정(RevIN·분위 손실), (c) 평가 프레임의
자기기만(변동 없는 예측이 방어 우수로 오인되는 MDD 단독 지표) 문서화 — 세 가지가 기여점이다.

## 5. 방법론 & 사용 데이터

- **데이터**: 업비트 KRW 마켓 15분봉. 단일 종목 심층 분석은 KRW-BTC 104,734행(2023-07~2026-07),
  전종목 축은 `upbit_krw_candle` 269종목 16,081,696행. DuckDB 저장.
- **전처리·정규화**: 로그수익률/차분/윈저라이즈, RevIN·Dish-TS 계열 instance 정규화.
- **모델**: Linear/PatchTST/DLinear/NLinear/TCN/ModernTCN/(i)Transformer/Autoformer/Mamba/
  TimesNet 계열("Shallow but Wide": 1~2층, width 64~128).
- **손실**: Huber/balanced-composite(평균형) → 분위(pinball)·분포(Student-t)·tail-weighted(큰 변동용).
- **평가 지표**: persistence 대비 copy_risk·MASE, variance_ratio(진폭), 방향 정확도·large-move
  방향정확도·tail F1(큰 변동 포착), cross-sectional IC, 그리고 거래비용 반영 MDD·수익(생존 제약).
- **탐색적 분석(17)**: ADF/KPSS/ARCH-LM, STL 분해, ACF/PACF, 상호정보량.

## 6. Baseline·유사 지표를 쓴 참고문헌

(실재 확인된 것만. 상세·URL은 `test/research_materials/16_..._20260719.md`,
`17_direction_predictability_literature_review_20260719.md`)

**방향/부호 예측성의 이론적 기반**
- Christoffersen & Diebold (2006), "Financial Asset Returns, Direction-of-Change Forecasting,
  and Volatility Dynamics", *Management Science* / NBER WP 10009 — 조건부 평균 무예측과 부호·
  변동성 예측성이 양립함을 증명. **우리 방향 지표(direction accuracy) 해석의 근거.**
- Cont (2001), "Empirical properties of asset returns: stylized facts", *Quantitative Finance*
  1(2) — 수익률 자기상관 ≈0 vs |수익률| 강한 자기상관(변동성 군집). **우리 S3 결과의 근거.**
- Pesaran & Timmermann (1992), *JBES* 10(4) — 방향 예측 정확도 검정. **평가 지표 근거.**

**딥러닝 시계열 예측 baseline (우리가 구현·비교한 모델)**
- PatchTST (arXiv:2211.14730), iTransformer (arXiv:2310.06625), DLinear/NLinear,
  Autoformer, TimesNet — 우리 모델군의 원 논문.
- RevIN (ICLR 2022), Non-stationary Transformer (arXiv:2205.14415), Dish-TS (arXiv:2302.14829),
  FAN (arXiv:2409.20371) — 비정상성 정규화, 우리 16번의 근거.

**손실·평가 지표**
- Koenker & Bassett (1978) 분위 회귀(pinball) · DeepAR (arXiv:1704.04110) Student-t 분포 예측
  · tail-weighted loss (arXiv:2112.00825) — 우리 큰 변동 손실의 근거.
- MASE (Hyndman & Koehler 2006), CRPS (Gneiting & Raftery 2007) — 우리 평가 지표.

**암호화폐·고빈도 방향 예측 (유사 조건 선행연구)**
- Stefaniuk & Ślepaczuk (arXiv:2503.18096) — 고빈도 Bitcoin, 순수 MSE의 0-붕괴와 손실 교체
  효과. **우리 진폭 압축과 같은 실패 모드 실증.**
- Gu, Kelly & Xiu (2020), *RFS* — 자산 수익률 ML 예측의 현실적 R²(~0.4%). **우리 낮은 R²가
  분야 표준임을 보이는 근거.**
- (오더북/FI-2010 계열 DeepLOB arXiv:1808.03668 등은 호가창 정보라 우리와 조건이 다름 — 비교
  불가로 명시.)

## 7. 교수님께 여쭐 논의점

1. 위 negative-result + 진단/교정을 **방법론 논문**으로 정리하는 방향이 적절한지.
2. 다음 확장(변동성 예측 GARCH 베이스라인 vs 딥러닝 / 외생정보 도입 / cross-sectional)을
   본 논문에 포함할지, 후속 논문으로 분리할지.
3. 평가 프레임(방향+크기+생존제약 3축)과 baseline 선정이 심사에서 방어 가능한지.
