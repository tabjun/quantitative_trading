# 13번 feature-algorithm-resource 실험 — 복원 정본 결과 (notebook backup 파싱)

> 작성일: 2026-06-28
> 원본: 저장 실패한 `13_feature_algorithm_resource_test.ipynb`의 **VSCode 마지막 성공 백업본**
> (`%APPDATA%\Code\Backups\030fa7eb8939ca69ef0c9b0c3b7e69de\file\7394394e`, 13.3 MB, mtime 2026-06-25 02:41)
> 이 백업본을 JSON 파싱해 stdout 47,115자 + display 표 + 이미지 24장을 전량 복원했다.

---

## 0. 이 문서가 이전 캡처 보고서를 대체한다

디스크의 `.ipynb`(40,881 bytes)는 `# [FOR COMMIT TRACKING ONLY - DO NOT EXECUTE]` 마커가 붙은
**커밋 추적용 스텁**이라 output이 0개였다. 그래서 앞선 보고서들은 화면 스크린샷 4장만으로 작성됐다.

그러나 VSCode가 저장 실패 직전 마지막으로 성공시킨 **hot-exit 백업(13.3 MB)**이 남아 있었고,
이 안에는 실제 실행 output이 전부 직렬화되어 있었다. 본 문서는 그 백업을 파싱한 **정본**이며,
스크린샷 기반 추정치(예: fusion 표 컬럼 미상)를 정확한 수치로 교체한다.

### 0.1 복원된 것 (백업 파싱 = 정확한 수치)

- 실행 환경 메타(JSON), RAW/파생변수 설명표, 13개 feature group 가용성표, missingness audit
- suite 실행 계획: `feature_algorithm_matrix`, **executable_cases=1440**
- case별 분배: feature group당 90케이스(16그룹), point model당 144케이스(10모델)
- **case 1~8 전체 결과**(학습 로그 + prediction preview + risk 로그 + event preview + **policy 성능표**)
- **case 9 학습 로그**(여기서 백업이 끊김)
- 이미지 24장 (`test/images/13_recovered_from_backup_20260628/img_01~24_*.png`)

### 0.2 끝내 존재하지 않는 것 (생성 자체가 안 됨)

실행은 **1440 케이스 중 9개**까지만 진행됐고, 그 직후 backup이 "too large"로 실패하기 시작했으며
(화면상 case 28+까지 더 갔지만 백업 안 됨), 최종적으로 PyTorch shared memory 에러로 중단됐다. 따라서:

- 전체 leaderboard, feature group 평균, model family 평균, feature×model heatmap, final summary
  → **애초에 그 단계까지 도달하지 못해 생성된 적이 없다.** 재실행만이 유일한 복원 경로다.

> 즉 이전에 "영구 소실"이라 적었던 표들은 "소실"이 아니라 **"아직 만들어지지 않음"**이 정확하다.
> 9/1440 = 0.6%만 완주했으므로, 이번 회차는 성능 결론 실험이 아니라 **8케이스 파일럿 + 자원 장애**다.

---

## 1. 실행 환경 (백업 stdout 원문)

```json
{
  "python_version": "3.12.13 (main, May 10 2026) [Clang 22.1.3]",
  "platform": "Linux-5.15.153.1-microsoft-standard-WSL2-x86_64 (glibc2.35)",
  "cpu_logical": 32, "cpu_physical": 16,
  "memory_total_gb": 15.53, "memory_available_gb": 6.93,
  "torch_version": "2.10.0+cu126", "torch_cuda_version": "12.6",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4090", "gpu_memory_gb": 23.99,
  "selected_resource_profile": "school_4090_15gb:exclusive",
  "applied_num_workers": 4, "applied_batch_size": 48,
  "applied_torch_threads": 16, "applied_torch_interop_threads": 4
}
```
- batch policy: `requested=1024 oom_retry=2048->1024->512->256->...` (실제 적용 batch=48).
- **주목**: `memory_total_gb=15.53` (WSL2 RAM 15.5GB)인데 `num_workers=4` + 대량 output이 겹쳐
  shared memory(`/dev/shm`)가 고갈됐다. GPU(24GB)는 여유 있었으나 **시스템 RAM/shm이 병목**이었다.

---

## 2. 실험 설계 (백업 stdout 원문)

### 2.1 suite 계획
- suite=`feature_algorithm_matrix`, **executable_cases=1440**
- 축: feature_group × preprocessing × point_model × seed (risk_model=PatchTSTLike 고정,
  seq_len=64, risk_horizon=16, risk_event_kind=absolute_move, risk_allow_quantile=0.55)

### 2.2 case 분배 (실제 출력)

feature group별 90케이스 (16개 그룹 × 90 = 1440):
```
mtf_returns_only, mtf_volatility_only, mtf_trend_position_only, mtf_volume_range_only,
mtf_returns_volatility, mtf_returns_trend, mtf_volatility_trend, mtf_full,
mtf_plus_momentum_reversal, mtf_plus_calendar_cycle, mtf_plus_shock_event,
mtf_plus_attention_proxy, mtf_plus_liquidity_micro, mtf_plus_orderflow_proxy,
mtf_plus_full_available, mtf_plus_text_context  → 각 90 케이스
```
point model별 144케이스 (10개 모델 × 144 = 1440):
```
Linear, PatchTSTLike, TCNLike, ModernTCNLike, DLinearLike, NLinearLike,
TransformerLike, ITransformerLike, AutoformerLike, MambaLike  → 각 144 케이스
```
preprocessing 3종: `seasonal_diff16`, `winsor_025`, `linear_detrend+asinh_robust`
seeds 3종: `42`, `2026`, `777`

### 2.3 13개 feature group 가용성 (included 16 / excluded 4)
- **included (16)**: 위 2.2 목록 (각 n_features: mtf_returns_only=5 … mtf_plus_full_available=60)
- **excluded (4)**: `mtf_plus_cross_market`, `mtf_plus_macro_proxy`, `mtf_plus_onchain_proxy`,
  `mtf_plus_derivatives_proxy` — 사유: "usable columns too few or optional mart columns missing"
- missingness: raw OHLC missing=0, inferred gaps=15, make_features dropna 제거행=114

---

## 3. ★ 완료된 case 1~8 성능표 (policy preview) — 이번 실험의 실질 결과

각 case는 `cumulative_return / mdd / sortino_proxy`를 4개 정책으로 비교한다.
buy_hold는 모든 케이스 동일(테스트 구간 자체가 하락장: ret=-0.1481, mdd=-0.3757).
**Defense First 관점에서 핵심 지표는 MDD(최대낙폭, 0에 가까울수록 방어 우수)다.**

| case | point_model | seed | point_only ret / mdd | risk_gate_only ret / mdd | point_plus_risk_gate ret / mdd |
|---|---|---|---|---|---|
| 1 | Linear | 42 | -0.2763 / -0.2763 | -0.9573 / -0.9576 | -0.2081 / -0.2081 |
| 2 | Linear | 2026 | -0.9575 / -0.9575 | -0.8840 / -0.8853 | -0.8789 / -0.8789 |
| 3 | Linear | 777 | -0.6832 / -0.6832 | -0.9728 / -0.9729 | -0.5648 / -0.5648 |
| 4 | PatchTSTLike | 42 | **-0.0243 / -0.0302** | -0.9573 / -0.9576 | **-0.0209 / -0.0268** |
| 5 | PatchTSTLike | 2026 | -0.1982 / -0.1982 | -0.8840 / -0.8853 | -0.1692 / -0.1692 |
| 6 | PatchTSTLike | 777 | -0.2675 / -0.2675 | -0.9728 / -0.9729 | -0.2155 / -0.2155 |
| 7 | TCNLike | 42 | 0.0000 / 0.0000 (무거래) | -0.9573 / -0.9576 | 0.0000 / 0.0000 (무거래) |
| 8 | TCNLike | 2026 | 0.0000 / 0.0000 (무거래) | -0.8840 / -0.8853 | 0.0000 / 0.0000 (무거래) |
| 9 | TCNLike | 777 | (policy 미생성 — 백업 여기서 끊김) | — | — |

(no_trade 정책은 전 case에서 0.0000 / 0.0000, 기준선이다.)

### 3.1 이 8케이스에서 읽히는 사실 (정확한 수치 기반)

1. **case 4 (PatchTSTLike, seed42)가 이 파일럿에서 유일하게 방어에 성공**:
   point_plus_risk_gate가 ret -2.09%, **MDD -2.68%**로 buy_hold(MDD -37.6%) 대비 압도적으로 낙폭을 줄였다.
2. **risk_gate_only는 모든 case에서 재앙적**(ret/mdd -88~-97%): risk gate 단독은 과매매(turnover 1500~2200)로
   파멸. risk gate는 단독이 아니라 point 신호와 결합(point_plus_risk_gate)할 때만 의미가 있다.
3. **point_plus_risk_gate가 point_only보다 거의 항상 MDD를 줄였다**(case 1·3·4·5·6) →
   risk gate를 방어 보조로 쓰는 설계 방향 자체는 옳다.
4. **TCNLike(case 7·8)는 무거래(trade_count=0)** → 신호가 임계를 못 넘겨 거래를 안 함. 0% 방어이긴 하나
   "기회 상실"이며, 임계/스케일 점검 필요.
5. **seed 분산이 매우 크다**(Linear: seed42 -27.6% vs seed2026 -95.8%) → 단일 seed 결론은 위험,
   seed 평균이 필수인데 이번엔 seed가 다 안 돌아 평균을 낼 수 없다.

---

## 4. 대표 학습 로그 (백업 stdout 원문)

### 4.1 case 1 point branch (Linear, seed42) — 24 epoch 전량
```text
epoch=001 train=0.045851 val=0.022673 grad=0.5075 lr=9.96e-04
epoch=005 train=0.007606 val=0.006849 grad=2.0280 lr=8.97e-04
epoch=010 train=0.002849 val=0.001410 grad=1.0604 lr=6.29e-04
epoch=013 train=0.001636 val=0.001264 grad=0.6252 lr=4.35e-04   ← best 근방
epoch=018 train=0.001567 val=0.001189 grad=2.8261 lr=1.46e-04
epoch=024 train=0.001440 val=0.001178 grad=2.3054 lr=0.00e+00
```
(전체 24 epoch이 백업에 있음. val 0.0227 → 0.00118로 안정 수렴, grad 0.5~4.6 범위.)

### 4.2 case 1 risk branch (PatchTSTLike risk) — early-stop
```text
epoch=001 train=0.945247 val=0.882157 grad=0.2791
epoch=009 train=0.810605 val=1.009309 grad=0.6104
[early-stop] epoch=9 best_val=0.861249
```
risk branch는 train은 줄지만 val이 0.86~1.01에서 진동 → risk 분류기가 일반화에 약함(§3.2의
risk_gate_only 참사와 일치). best_val=0.861은 거의 무작위(이벤트 base rate 부근).

### 4.3 case 9 point branch (TCNLike, seed777) — 백업이 끊긴 지점
```text
epoch=001 train=0.003604 val=0.000878 grad=0.5045 lr=9.96e-04
epoch=008 train=0.001605 val=0.001238 grad=0.4394 lr=7.50e-04
[early-stop] epoch=8 best_val=0.000878
```
→ 이 직후 policy preview를 만들기 전에 백업이 끊겼다(이후 화면상 case 28+까지 진행되다 shm 크래시).

---

## 5. prediction / event preview 형식 (백업 원문, case 1 예시)

### 5.1 point prediction preview (Linear seed42) — pred 컬럼까지 전량
실제 OHLC + `pred_open_proxy / pred_close_proxy / persistence_close / actual_return / predicted_return`.
대표 행:

| timestamp | actual_close | pred_close_proxy | persistence_close | actual_return | predicted_return |
|---|---|---|---|---|---|
| 2026-05-20 00:30 | 114,061,000 | 114,086,768 | 114,106,000 | -0.000394 | -0.000169 |
| 2026-05-20 01:00 | 113,968,000 | 114,152,848 | 114,093,000 | -0.001096 | 0.000524 |
| 2026-05-20 01:30 | 114,342,000 | 114,208,504 | 114,162,000 | 0.001575 | 0.000407 |

> predicted_return(±0.0005)이 actual_return(±0.0017)보다 **진폭이 작다 = 분산 압축(collapse)**.
> pred_close_proxy가 persistence_close에 매우 가까움 → persistence를 크게 이기지 못함.

### 5.2 risk event preview (PatchTSTLike risk, seed42) — threshold/확률 포함
| timestamp | future_return | event_score | event_label | predicted_probability | threshold |
|---|---|---|---|---|---|
| 2026-05-20 00:15 | -0.003918 | 0.006165 | 0 | 0.439287 | 0.42 |
| 2026-05-20 02:00 | 0.001738 | 0.004994 | 0 | 0.353292 | 0.42 |
| 2026-05-20 02:15 | 0.003498 | 0.004223 | 0 | 0.402773 | 0.42 |

> 이 구간 event_label은 전부 0(이벤트 없음)인데 predicted_probability는 0.35~0.48로 threshold(0.42)
> 부근에서 흔들림 → risk 분류기가 비이벤트 구간에서도 확률을 높게 줘 과매매 유발(→ risk_gate_only 참사).

---

## 6. 진단 그래프 (백업에서 추출한 이미지 24장)

`test/images/13_recovered_from_backup_20260628/img_01~24_*.png` (case 1~8, case별 약 3장).
각 case 묶음: ① point branch 6분할 진단(Total objective / Objective components / Gradient norm /
Return prediction / Next-candle comparison / Calibration scatter), ② risk branch 3분할
(Precision-recall / Selective risk-coverage / Probability vs realized risk), ③ policy/누적수익 곡선.

공통 관찰(case 1~8):
- Calibration scatter Pearson이 일관되게 매우 낮음(0.09 부근) → point 방향 예측력 거의 없음.
- Return prediction 진폭 압축(collapse) 반복.
- Gradient norm은 대체로 안정(0.3~6)이나 일부 epoch에서 스파이크.
- risk Precision-recall이 base rate 부근으로 붕괴 → risk 분류기 판별력 약함.

> 그래프 해석 지침(데이터·모델→축→목적→관찰→좋음/나쁨→다음)은 이미지별로
> 후속 보고서에서 case 4(유일 성공)와 case 2(최악)를 대비해 상세화할 수 있다.

---

## 7. 중단 원인 (재확인)

1. **실행**: `RuntimeError: unable to allocate shared memory(shm) ... Resource temporarily unavailable (11)`
   — WSL2 RAM 15.5GB + `num_workers=4` + case마다 큰 표/이미지 누적 → /dev/shm 고갈.
2. **저장**: `Notebook too large to backup` — output 누적으로 VSCode snapshot 실패.
   본 백업(13.3MB, case 9 시점)이 **마지막으로 성공한 snapshot**이고 이후 backup은 계속 실패했다.
- traceback 경로: `9_preprocessing_uncertainty_diagnostics_test.py:126 fig.tight_layout()` →
  `pylabtools.py:170 print_figure` (matplotlib PNG 렌더 중 폰트 경고 다수: Glyph 48264 '번' missing).

---

## 8. 재실행 시 반드시 바꿀 것 (이번 정본 데이터가 뒷받침)

1. **shm 고갈 차단**: `num_workers=0~1`, batch는 이미 48로 작으니 유지, WSL `/dev/shm` 용량 상향 또는
   `--shm-size` 조정. (GPU가 아니라 시스템 RAM/shm이 병목이었음이 §1로 확인됨.)
2. **output 폭주 차단**: case별 표·이미지 전량 출력 금지. 대표 case만 plot, 나머지는 수치만 누적.
   leaderboard/평균은 노트북 끝에서 1회만. raw는 CSV로 분리. (이것이 저장 실패의 직접 원인.)
3. **suite 분리**: 1440을 한 노트북에 넣지 말고 feature group/모델 단위로 쪼개 부분 완주 보장.
4. **risk_gate_only 재설계**: 단독 risk gate는 전 case 파멸(-88~-97%). quantile 0.55를 높이거나
   point 신호와의 결합만 허용. point_plus_risk_gate가 MDD를 줄인 것은 유지할 방향.
5. **TCNLike 무거래 점검**: case 7·8 trade_count=0 → 신호 임계/스케일 조정.
6. **seed 평균 필수**: seed 분산이 매우 커(Linear -27.6% vs -95.8%) 단일 seed 결론 금지.

---

## 9. 한 줄 요약

13번은 1440 케이스 설계였으나 **WSL2 shm 고갈로 case 9까지(0.6%)만 완주**했고,
leaderboard·평균·summary는 생성된 적이 없다(소실이 아니라 미생성). 저장 실패한 노트북의
**마지막 성공 백업(13.3MB)**을 복원해 case 1~8 전체 성능표를 확보했으며, 그 결과
**case 4(PatchTSTLike+risk gate, seed42)만 MDD -2.68%로 방어 성공**, risk_gate_only는
전 case 파멸, point 예측은 calibration·분산 모두 약했다. 결론을 내려면 shm/output 정책을
고친 재실행이 필요하다.

---

## 10. 산출물

- **본 정본(이 문서)**: `test/results/13_feature_algorithm_resource_recovered_full_results_20260628.md`
- 복원 이미지 24장: `test/images/13_recovered_from_backup_20260628/img_01~24_*.png`
- (참고, 스크린샷 기반 이전본) `13_feature_algorithm_resource_full_captured_results_20260628.md` 등
- 백업 원천: `%APPDATA%\Code\Backups\030fa7eb8939ca69ef0c9b0c3b7e69de\file\7394394e`
