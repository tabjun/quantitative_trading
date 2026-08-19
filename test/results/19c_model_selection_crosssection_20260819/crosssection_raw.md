# 19c 모델 선택 전종목(20) 확인 원시 수치

- 대상: 유동성 상위 20종목(16번 t3·18번 D6와 같은 표본): KRW-XRP, KRW-BTC, KRW-DOGE, KRW-ETH, KRW-SOL, KRW-SHIB, KRW-SEI, KRW-XLM, KRW-SUI, KRW-ETC, KRW-ADA, KRW-STX, KRW-HBAR, KRW-GAS, KRW-NEAR, KRW-BCH, KRW-SAND, KRW-ARK, KRW-POLYX, KRW-LINK
- 조건: 주기 성분 제거(19번에서 18/18셀 전부 우세이므로 이 조건만 사용)
- 19번의 `run_condition`을 그대로 재사용 — 수치 일관성 보장

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

