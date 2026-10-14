# 19c 모델 선택 전종목(20) 확인 원시 수치

- 대상: 유동성 상위 20종목(16번 t3·18번 D6와 같은 표본): KRW-XRP, KRW-BTC, KRW-DOGE, KRW-ETH, KRW-SOL, KRW-SHIB, KRW-SEI, KRW-XLM, KRW-SUI, KRW-ETC, KRW-ADA, KRW-STX, KRW-HBAR, KRW-GAS, KRW-NEAR, KRW-BCH, KRW-SAND, KRW-ARK, KRW-POLYX, KRW-LINK
- 조건: 주기 성분 제거(19번에서 18/18셀 전부 우세이므로 이 조건만 사용)
- 19번의 `run_condition`을 그대로 재사용 — 수치 일관성 보장

## 실험 설계

> 드라이버 서두 설계 셀 원문(`test/models/19c_model_selection_crosssection.py`). 실행 전에 적은 설계이며, 결과를 본 뒤 바꾼 내용은 결과 절에 따로 적는다.

### 19c: B 갈래 모델 선택을 전종목으로 확인 — BTC 하나로 정한 게 아니다

#### 왜 (Why)
사용자 지적(2026-08-19): "모든 종목에 대한 분석을 다 원하는데, `preflight.ensure_data`
예시가 KRW-BTC만 있어서 다시 BTC 단일종목 문제로 돌아간 거 아닌가?"

확인해보니 실제로 절반은 맞는 지적이었다. 18d(전처리 사다리)·18e(설명용 그림)·18f(지표
유효성)는 **개념 설명용**이라 BTC 하나로 충분하다(269종목에 같은 메커니즘 그림을 269번
그려도 새 정보가 없다). 그런데 **19번(B 갈래 모델 선택 — GARCH-t vs HAR-RV)은 BTC 하나로만
결론 냈다.** "4시간이면 GARCH-t, 16시간이면 HAR-RV"가 BTC 특정 현상인지 구조적인지 확인이
안 됐다 — 정확히 지적받은 지점이다.

#### 무엇을 (What)
19번의 삼원 격자(모델×주기제거×horizon)를 **유동성 상위 20종목**(16번 t3·18번 D6와 같은
표본)에 반복한다. 결과는 "종목마다 최우수 모델"의 분포로 집계한다 — 개별 QLIKE 수치가
아니라 **승자의 일관성**이 이 실험의 산출물이다.

#### 어떻게 (How)
19번의 `run_condition`을 그대로 재사용한다(재구현하면 두 실험의 수치가 갈린다). 주기 제거
조건만 쓴다(19번에서 18/18셀 전부 이겼으므로 이미 결정됐다). 비용 통제를 위해 GARCH 시작점을
19번보다 줄인다.

#### 기대 결과 / 반영 (Expected)
"target horizon별 최우수 모델"이 BTC 특정이 아니라 몇 종목에서 재현되는지 확정한다.

---

## 종목 × horizon 최우수 모델

| 종목 | h=1(15분) | h=16(4시간) | h=64(16시간) |
| :--- | :--- | :--- | :--- |
| KRW-XRP | GARCH-t | GARCH-t | GARCH-t |
| KRW-BTC | GARCH-t | GARCH-t | HAR-RV(일·주·월) |
| KRW-DOGE | GARCH-정규 | HAR-RV(일·주·월) | HAR-RV(일·주·월) |
| KRW-ETH | GARCH-정규 | GARCH-정규 | GARCH-t |
| KRW-SOL | GARCH-t | GARCH-t | HAR-RV(일·주·월) |
| KRW-SHIB | GARCH-정규 | 롤링분산(96) | 롤링분산(96) |
| KRW-SEI | GARCH-t | 롤링분산(96) | 롤링분산(96) |
| KRW-XLM | GARCH-정규 | HAR-RV(일·주·월) | HAR-RV(일·주·월) |
| KRW-SUI | GARCH-t | GARCH-t | GARCH-t |
| KRW-ETC | GARCH-t | GARCH-t | 롤링분산(96) |
| KRW-ADA | GARCH-t | GARCH-t | GARCH-t |
| KRW-STX | GARCH-t | GARCH-정규 | GARCH-t |
| KRW-HBAR | GARCH-t | HAR-RV+(분기추가) | HAR-RV+(분기추가) |
| KRW-GAS | GARCH-t | GARCH-t | HAR-RV(일·주·월) |
| KRW-NEAR | GARCH-t | GARCH-t | GARCH-t |
| KRW-BCH | GARCH-t | GARCH-t | HAR-RV(일·주·월) |
| KRW-SAND | GARCH-t | GARCH-정규 | HAR-RV+(분기추가) |
| KRW-ARK | GARCH-t | GARCH-t | 롤링분산(96) |
| KRW-POLYX | GARCH-t | GARCH-정규 | HAR-RV(일·주·월) |
| KRW-LINK | GARCH-t | GARCH-t | GARCH-t |

## horizon별 승자 분포 — 이것이 실제 산출물이다

### h=1 (15분)

- 유효 종목 수: 20/20
  - **GARCH-t**: 16종목 (80%)
  - **GARCH-정규**: 4종목 (20%)

### h=16 (240분=4시간)

- 유효 종목 수: 20/20
  - **GARCH-t**: 11종목 (55%)
  - **GARCH-정규**: 4종목 (20%)
  - **HAR-RV(일·주·월)**: 2종목 (10%)
  - **롤링분산(96)**: 2종목 (10%)
  - **HAR-RV+(분기추가)**: 1종목 (5%)

### h=64 (960분=16시간)

- 유효 종목 수: 20/20
  - **GARCH-t**: 7종목 (35%)
  - **HAR-RV(일·주·월)**: 7종목 (35%)
  - **롤링분산(96)**: 4종목 (20%)
  - **HAR-RV+(분기추가)**: 2종목 (10%)

## 판정 (동률 처리 수정판 — 최초 집계는 tie-break를 명시하지 않아 재작성했다)

- h=1: **GARCH-t**가 16/20종목(80%)에서 최다 → **구조적**
- h=16: **GARCH-t**가 11/20종목(55%)에서 최다 → **혼재**(정규 4, HAR-RV 2, 롤링분산 2, HAR-RV+ 1로 분산)
- h=64: **동률** — GARCH-t와 HAR-RV(일·주·월)가 각 7/20종목(35%)으로 공동 최다
  → **단일 승자를 정할 수 없다.** 16시간 target에서는 두 모델을 함께 후보로 두고
  종목별 성격(유동성·거래 목적)에 따라 고르는 것이 맞다.

- BTC 단독 결과(19번: h=1 GARCH-t·h=16 GARCH-t·h=64 HAR-RV)와 전종목 다수결의 일치:
  h=1 일치, h=16 일치, h=64는 BTC가 고른 HAR-RV가 **동률의 절반**이라 완전한 불일치는 아니다.
  → **BTC 단일종목 결론은 h=1·16(15분~4시간)에서는 전종목 대표성이 있다.** h=64(16시간)는
  BTC만으로 HAR-RV 단독 승자라 결론 내린 것이 과했다 — 실제로는 GARCH-t와 동률이다.
  target이 16시간이면 **두 모델을 모두 후보로 표본외 검증**해야 한다.

### 참고: 최초 자동 집계(수정 전) 오류

최초 코드는 동률을 처리하지 않고 `value_counts().index[0]`로 임의의 하나(GARCH-t)를 골라
"h=64: BTC 특정 의심"이라는 오해의 소지가 있는 라벨을 냈다. **실제로는 BTC 특정 문제가 아니라
순수한 동률**이었다 — GARCH-t와 HAR-RV가 정확히 7종목씩 나뉜다. 코드는
`test/models/19c_model_selection_crosssection.py`에서 동률을 명시하도록 수정했다.

