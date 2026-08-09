# 13번 노트북 live recovery dump

## 목적

이 문서는 저장에 실패한 `test/models/13_feature_algorithm_resource_test.ipynb`의 **열려 있는 VSCode 화면**에서
즉시 대피시킨 출력 화면 캡처와, 그 화면에서 직접 읽어낸 최소한의 구조 정보를 모아 둔 복구 덤프다.

중요한 점:

- 이것은 저장된 `.ipynb` output을 파싱한 결과가 아니다.
- 열려 있던 VSCode 화면을 구간별로 캡처한 **live recovery**다.
- 따라서 일부 표/그래프는 부분 화면만 남아 있고, 전체 수치는 아직 완전하지 않다.

## 현재까지 확보한 캡처

캡처 폴더:

- `test/images/13_output_recovery_20260627/`

확보 파일:

1. `capture_001.png`
2. `capture_002.png`
3. `capture_003.png`
4. `capture_004.png`

## 캡처 1

파일:

- [capture_001.png](/C:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/test/images/13_output_recovery_20260627/capture_001.png)

화면 내용:

- 상단에 작은 결과 표 일부가 보인다.
- 표 아래에는 `5 rows x 8 cols`가 보인다.
- 이어서 `[case 28/1440] mtf_returns_only / seasonal_diff16 / MambaLike + PatchTSTLike / seed42 / seq64 / ...` 학습 로그가 보인다.
- 그 아래에는 해당 케이스의 대표 진단 그래프 일부가 보인다.

읽힌 핵심 정보:

- 총 케이스 수는 화면 기준 `1440`으로 보인다.
- 적어도 `mtf_returns_only` feature set, `seasonal_diff16` 전처리, `MambaLike + PatchTSTLike`, `seed42`, `seq64` 조합이 실행되었다.
- `epoch=001`부터 `epoch=014`까지 로그가 보이고, 마지막 줄은 `early-stop epoch=14 best_val=0.002502`로 읽힌다.

해석 메모:

- 13번은 실제로 large matrix를 돌고 있었고, 단순 dry plan이 아니라 케이스별 학습 로그와 대표 그래프까지 출력하고 있었다.
- 하지만 이 방식은 notebook output volume을 크게 키워 저장 실패 리스크를 높였을 가능성이 크다.

## 캡처 2

파일:

- [capture_002.png](/C:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/test/images/13_output_recovery_20260627/capture_002.png)

화면 내용:

- risk branch 계열 그래프 3개가 선명하게 보인다.
  - `Precision-recall curve`
  - `Selective risk-coverage`
  - `Probability versus realized risk`
- 아래에는 다음 표가 보인다:
  - `[PatchTSTLike__mtf_returns_only__winsor_025__absolute_move__horizon16__seed777 event preview]`
- 표 하단에는 에러/traceback 일부가 보인다.

읽힌 표 컬럼:

- `timestamp`
- `prev_close`
- `future_end_close`
- `future_return`
- `event_score`
- `event_label`

읽힌 예시 행:

- `2026-05-20 00:15:00`
- `2026-05-20 00:30:00`
- `2026-05-20 00:45:00`

읽힌 핵심 정보:

- risk event 쪽에서는 `PatchTSTLike + mtf_returns_only + winsor_025 + absolute_move + horizon16 + seed777` 조합이 적어도 실행되었다.
- `absolute_move` risk event와 `horizon16` 설정이 실제 output에 포함되어 있음을 확인했다.
- event preview table이 실제로 생성되고 있었다.
- 하단 traceback 일부에는 `test/models/9_preprocessing_uncertainty_diagnostics...` 경로와 `fig.tight_layout()` / `fig.canvas.print_figure(...)` 같은 matplotlib 출력 관련 줄이 보인다.

해석 메모:

- 13번은 단순 point branch만이 아니라 risk branch preview와 calibration/risk 그래프까지 한 노트북에 계속 누적 출력하고 있었다.
- 이게 `Notebook too large to backup` 문제를 더 악화시켰을 가능성이 높다.

## 캡처 3

파일:

- [capture_003.png](/C:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/test/images/13_output_recovery_20260627/capture_003.png)

화면 내용:

- point prediction 계열 6개 진단 그래프가 보인다.
  - `Total objective`
  - `Objective components`
  - `Gradient norm`
  - `Return prediction`
  - `Next-candle comparison`
  - `Calibration scatter / Pearson=0.097`
- 아래에는
  - `[MambaLike__seasonal_diff16__balanced_composite__seed42 prediction preview]`
  표가 보인다.

읽힌 표 컬럼:

- `timestamp`
- `actual_open`
- `actual_high`
- `actual_low`
- `actual_close`
- `pred_open_...` 이후 예측 OHLC 계열 컬럼 일부

읽힌 핵심 정보:

- point branch에서는 `MambaLike + seasonal_diff16 + balanced_composite + seed42` 조합이 실행되었다.
- prediction preview table이 실제로 생성되고 있었다.
- candle proxy 기반 preview 출력이 살아 있음을 확인했다.

해석 메모:

- 13번은 12번의 보고서 구조를 계승해 point branch 진단 그래프와 candle preview를 포함한 상태로 확장 실행 중이었다.
- 즉 사용자 의도였던 “변수군 + 알고리즘 + 전처리 + seed + risk gate 동시 확인” 구조는 실제로 구현되어 돌아가고 있었다.

## 캡처 4

파일:

- [capture_004.png](/C:/Users/jun99/OneDrive/바탕%20화면/Analysis/toy_agent_project/quantitative_trading/test/images/13_output_recovery_20260627/capture_004.png)

화면 내용:

- 상단에는 학습 로그가 길게 이어지고,
- 하단에는 다음 케이스의 point branch 진단 그래프가 보인다:
  - `PatchTSTLike__winsor_025__balanced_composite__seed2026`

읽힌 핵심 정보:

- `PatchTSTLike + winsor_025 + balanced_composite + seed2026` 조합이 실제로 실행되었다.
- `Gradient norm` 패널에서 `max`가 매우 크게 튀는 구간이 보인다. 화면상 대략 `500+` 수준 스파이크가 관찰된다.
- `Calibration scatter / Pearson=0.094`가 보인다.

해석 메모:

- 단순히 에러만 난 것이 아니라, 에러 전에도 일부 케이스에서는 gradient spike가 큰 상태가 관찰됐다.
- 즉 13번은 자원 문제뿐 아니라, 일부 조합에서 optimization stability 자체도 같이 봐야 한다.

## 현재까지 화면에서 확정적으로 말할 수 있는 것

1. 13번 실험은 실제로 large matrix를 돌고 있었다.
2. 화면상 총 케이스 수는 `1440`으로 읽힌다.
3. 적어도 다음 축이 실제 실행에 포함되었다.
   - feature set: `mtf_returns_only`
   - preprocessing: `seasonal_diff16`, `winsor_025`
   - point models: `MambaLike`, `PatchTSTLike`
   - risk target: `absolute_move`
   - horizon: `horizon16`
   - objectives: `balanced_composite`
   - seeds: `42`, `777`, `2026`
4. point branch 진단 그래프, risk branch 그래프, prediction preview table, event preview table이 실제로 출력되고 있었다.
5. 실행 후반에는 저장 실패와 별개로 shared memory 계열 중단 문제가 있었다.

## 아직 확보하지 못한 것

다음은 현재 캡처만으로는 부족하다.

- top leaderboard 전체
- feature group average summary 전체
- model family average summary 전체
- final decision / top candidates 요약표
- 마지막 full traceback 전체
- 마지막까지 완료된 case 범위

## 왜 이 dump가 의미가 있나

비록 전체 결과를 완전 복원하지는 못했지만, 이 dump만으로도 아래는 증명된다.

1. 13번은 빈 노트북이 아니었다.
2. 실제 대규모 조합 실험이 돌았다.
3. point/risk/fusion 확장 구조가 살아 있었다.
4. 저장 실패는 “결과가 없어서”가 아니라 “결과가 너무 커서 저장이 안 된 것”에 가깝다.

## 다음 복구 우선순위

1. leaderboard 화면
2. feature group average 화면
3. model family average 화면
4. final decision 요약 화면
5. 마지막 traceback 전체 화면

이번 덤프는 그 전에 먼저 잡아 둔 **중간 복구본**이다.
