# 교정 축 재실행 검증 노트 (2026-08-14)

목적: 2026-08-14에 실험 드라이버 10개의 기본 테이블을 `btc_15m_advance` → `upbit_krw_candle`
(+ `--ticker KRW-BTC`)로 교정했다. **그 교정이 기존 결론을 바꾸는지** 확인한 기록이다.

- 사용자 승인: "교정 + 핵심 실험 재실행"(2026-08-13)
- 실행 환경: 서버, RTX 4090 24GB(`nvidia-smi`는 NVML 버전 불일치로 실패하지만 torch 연산 정상)
- 원본 7월 산출물은 각 실험의 날짜 디렉터리에 그대로 복원해 두었다(아래 5절 참조)

---

## 1. 결론 요약

**세 실험 모두 축 교정으로 결론이 바뀌지 않는다.** `btc_15m_advance`와 `upbit_krw_candle`의
해당 종목 데이터가 사실상 동일하기 때문이며(행수 차이 0.03% 수준, 양끝 2일 차이), 이는
"교정 때문에 과거 결과가 무효가 되지는 않는다"는 확인이다.

부수적으로 **두 가지 방법론 결함**을 발견했다(3·4절). 이쪽이 축 교정보다 중요하다.

---

## 2. 실험별 결과

### 2.1 17번 EDA (CPU, 학습 없음)

`--table upbit_krw_candle --ticker KRW-BTC`, 104,752행.

| 지표 | 기존(7월) | 재실행 | 판정 |
| :--- | ---: | ---: | :--- |
| 수익률 ACF(1) | −0.051299 | −0.05136 | 동일 |
| \|수익률\| ACF(1) | 0.363992 | 0.363948 | 동일 |
| 레벨 ADF p | 0.652007 | 0.581288 | 둘 다 비정상(≫0.05) |
| 레벨 KPSS p | 0.01 | 0.01 | 동일(비정상) |
| ARCH-LM p | 0 | 0 | 동일 |

"방향 무신호 / 크기 신호 있음", "레벨 비정상 / 수익률 평균-정상·분산-비정상" 모두 유지.

### 2.2 16번 t3 횡단면 (GPU, 30종목 학습)

`--suite t3_crosssection --normalization revin --loss pinball --n-tickers 30`. 30종목 완주, 오류 0.

| 지표 | 기존(7월) | 재실행 |
| :--- | ---: | ---: |
| trend_corr 평균 | +0.0027 | +0.0048 |
| trend_corr 중앙값 | −0.0009 | −0.0127 |
| trend_corr 양수 종목 | **14/30** | **14/30** |
| trend_corr 범위 | −0.0664~+0.0822 | −0.0754~+0.0950 |
| tail_f1 평균 | 0.251 | 0.256 |

재실행 추가 지표: `cross_sectional_ic = 0.0136`, direction_accuracy 평균 0.4968
(0.5 초과 13/30), variance_ratio 평균 0.461.

**판정 유지**: 횡단면에서 추세 상관이 0 근방이고 양수 종목이 정확히 절반. 상대강도 신호 없음.

### 2.3 15번 t8 다자산 (GPU, 10종목 학습)

7월은 4종목(BTC/ETH/XRP/SOL)을 **종목별 개별 테이블**(`eth_15m_advance` 등)로 돌렸다.
재실행은 유동성 상위 10종목을 **통합 테이블 하나**로 돌렸다(스테이블코인 제외, 모두 ~10.4만 행).

구성: `--models ITransformerLike --objective huber`(7월 승계 구성과 일치), seed 42.

**겹치는 4종목 — 같은 구성, 테이블만 다름(축 효과 분리):**

| 종목 | trend_corr 7월 | 재실행 | 차이 | large_move_da 7월 | 재실행 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| KRW-SOL | −0.0242 | −0.0237 | +0.0005 | 0.4678 | 0.4600 |
| KRW-XRP | 0.0046 | 0.0126 | +0.0080 | 0.4867 | 0.5117 |
| KRW-BTC | 0.0410 | 0.0514 | +0.0104 | 0.4922 | 0.5267 |
| KRW-ETH | 0.0006 | 0.0361 | **+0.0355** | 0.5044 | 0.5133 |

**10종목 전체(7월 구성):** trend_corr 평균 +0.0167, 중앙값 +0.0216, 양수 7/10,
범위 −0.0237~+0.0514. large_move_da 평균 0.5042(0.5 초과 6/10).
variance_ratio 평균 0.194, 범위 0.058~0.338.

**판정 유지**: 종목을 4개에서 10개로 넓혀도 신호가 드러나지 않는다. 추세 상관 최대 0.0514,
대변동 방향정확도 평균 0.5042(동전던지기).

---

## 3. 발견 ① 15번은 승계 구성을 CLI로 명시해야 한다 (재실행자 주의)

**처음 t8을 기본값으로 돌린 것은 오류였다.** 15번은 suite 간 우승 구성을 CLI로 승계하는
구조라(`--objective`, `--normalization`, `--preprocessing`, `--horizon`, `--models`),
기본값으로 돌리면 7월과 다른 실험이 된다.

| | 모델 | objective |
| :--- | :--- | :--- |
| 기존(7월) | ITransformerLike | huber |
| 기본값 실행(오류) | PatchTSTLike | balanced_composite |

이 차이로 같은 KRW-ETH의 variance_ratio가 **0.093 → 1.378(15배)**, mase_momentum이
0.691 → 1.272로 벌어졌다. 축 교정 효과로 오해할 수 있는 크기다. 구성을 7월과 맞추자
variance_ratio가 0.058~0.338로 돌아왔다.

- 잘못된 구성 실행 결과도 증거로 보존: `15_t8_multiasset_rerun_DEFAULTcfg_10tickers.csv`
- **교훈**: 15번 재실행 시 `--models`·`--objective`를 반드시 명시한다. 리더보드 CSV의
  `model`/`objective` 컬럼으로 구성 일치를 먼저 확인한 뒤 비교한다.
- 참고: 7월 승계 구성의 `huber`는 이후 "큰 변동·최적화에 애매하다"는 지적으로 방향이탈로
  분류된 손실이다(`known_pitfalls.md`, `select_champion` 코드게이트). 여기서는 **7월과의
  비교 가능성**을 위해 그대로 썼을 뿐, 이 구성을 권장하는 것이 아니다.

---

## 4. 발견 ② t8의 trend_corr는 실행 간 잡음이 신호보다 크다 (중요)

겹치는 4종목은 **같은 구성·같은 시드(42, `run_trend_case`에서 종목마다 `set_seed` 적용)·
거의 같은 데이터**인데 trend_corr가 +0.0005 ~ **+0.0355** 움직였다. 남은 차이 요인은
행수 0.03%(37행)·양끝 2일 차이와 GPU 비결정성뿐이다.

그런데 10종목 trend_corr 평균은 **+0.0167**이다 — **실행 간 잡음(최대 0.0355)이 측정하려는
신호(0.017)보다 크다.**

**함의**: t8의 원래 판독 기준("여러 종목에서 trend_corr가 함께 양수면 신호가 구조적,
한 종목만이면 우연·과적합 의심")은 이 잡음 수준에서 성립하지 않는다. 양수/음수 부호가
잡음으로 뒤집힐 수 있으므로, **부호 세기(7/10 양수)를 근거로 삼아선 안 된다.**

**후속 조치 제안**(미착수):
- t8을 다중 시드(≥5)로 돌려 종목별 trend_corr의 표준오차를 먼저 구한다.
- 그 표준오차보다 작은 종목 간 차이는 해석하지 않는다.
- `torch.backends.cudnn.deterministic = True` + `benchmark = False`로 GPU 비결정성을
  제거한 뒤 잡음의 순수 데이터 성분만 남긴다.
- 이 조치 전까지 t8 결과는 "신호 없음"의 근거로만 쓰고, 종목 간 순위 비교에는 쓰지 않는다.

---

## 5. 산출물 출처 관리

세 실험 모두 `EXPERIMENT_TAG`가 7월 날짜로 고정돼 있어, 재실행이 **7월 날짜 디렉터리를
덮어썼다**. 처리 방식:

- **7월 디렉터리는 git에서 원상 복원**했다(`17_eda_direction_signal_20260722`,
  `16_nonstationary_crosssectional_20260719`, `15_trend_capture_defense_20260716`).
  날짜 디렉터리는 그 날짜의 기록이어야 한다.
- **재실행 산출물은 이 디렉터리에 보존**한다(리더보드 CSV 3종 + 이 노트).
- 재실행 그림은 7월 그림과 사실상 동일해 별도 보존하지 않았다(필요하면 아래 명령으로 재생성).

**재실행 명령(그대로 재현 가능):**

```bash
# 17번 EDA
uv run test/models/17_eda_direction_signal_test.py

# 16번 횡단면 30종목
uv run test/models/16_nonstationary_crosssectional_test.py \
    --suite t3_crosssection --normalization revin --loss pinball --n-tickers 30

# 15번 t8 다자산 10종목 (승계 구성 명시 필수 — 3절)
uv run test/models/15_trend_capture_defense_test.py --suite t8_multiasset \
    --models ITransformerLike --objective huber \
    --tickers KRW-XRP,KRW-BTC,KRW-DOGE,KRW-ETH,KRW-SOL,KRW-SHIB,KRW-SEI,KRW-XLM,KRW-SUI,KRW-ETC
```

주의: 위 명령을 다시 실행하면 7월 디렉터리를 또 덮어쓴다. 실행 후
`git checkout -- test/results/<7월디렉터리>/ test/images/<7월디렉터리>/`로 복원하거나,
드라이버의 `EXPERIMENT_TAG`를 새 날짜로 바꿔서 돌린다.
