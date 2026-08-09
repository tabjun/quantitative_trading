# 16번 계획서 — 비정상성 정규화 + 큰 변동(분위/분포) 손실 + KRW 전 종목 cross-sectional

작성일: 2026-07-19 · 브랜치: `stock` · 토대: 공용 엔진 `engine/` + 16번 드라이버 신규 컴포넌트

> 근거 문헌: `test/research_materials/16_nonstationary_trend_capture_literature_review_20260719.md`
> 연구 초고: `test/results/paper_draft_nonstationary_crypto_trend_20260719.md`
> 지침 위계(CLAUDE.md 2.12): 학습 건전성 → 전체 변동 폭 추세 예측 → MDD는 생존 제약.

---

## 1. 왜 16번인가 (15번까지의 귀결)

15번 T1~T9로 확정된 사실:
- target을 h-step 추세로 바꿔 진폭이 부분 회복(variance_ratio 0.36)됐으나 방향 상관은 0.07,
  큰 변동 방향 정확도는 0.52로 미미하다.
- 이 약한 신호는 BTC 단일 종목의 우연이고 ETH/XRP/SOL로 일반화되지 않는다(T8).
- 진폭 압축의 원인은 시장이 아니라 (평균형 손실 + 과한 정상화 + 앙상블 상쇄)라는 설정이다(T9).

따라서 16번은 세 축을 **동시에** 바꾼다. 하나만 바꾸면(15번처럼) 다시 국소 최적에 갇힌다.

1. **비정상성 정규화**를 window_standard(경량)에서 정식 RevIN → De-stationary로 올려,
   정규화가 지운 급변 정보를 되살린다.
2. **손실**을 평균형(huber)에서 **분위(pinball)/분포(Student-t)/tail-weighted**로 바꿔 큰
   변동을 명시 모델링한다(진폭 압축 정면 해소).
3. **데이터 축**을 BTC 단일에서 **KRW 전 종목**으로 복원하고, 절대 예측이 아니라
   **cross-sectional ranking**(종목 간 상대 강도)도 objective로 도입한다.

## 2. 연구 질문

- Q1 (건전성·비정상성): RevIN/De-stationary 정규화가 window_standard 대비 variance_ratio를
  1에 가깝게 올리면서 검증 곡선 과적합을 악화시키지 않는가?
- Q2 (큰 변동): 분위/분포/tail 손실이 huber 대비 큰 변동 구간의 정밀도·재현율(tail P/R/F1)과
  variance_ratio를 유의하게 올리는가?
- Q3 (일반화): KRW 전 종목으로 학습한 모델이 종목 가로질러 양의 신호(cross-sectional IC)를
  내는가? 절대 예측이 안 되어도 상대 순위는 남는가?
- Q4 (생존 제약): 위에서 얻은 신호로 구성한 정책이, 변동 재현을 유지하면서(진폭 안 죽이고)
  MDD도 방어하는가?

## 3. 실험 축 (단계 T1~T5)

| suite | 고정 | 변화 | 답하는 질문 |
|---|---|---|---|
| **t1_normalization** | 손실=huber, 모델=ITransformer/PatchTST, BTC 단일(빠른 비교) | 정규화 {window_standard, revin, dishts_lite, none} | Q1 |
| **t2_loss** | 우승 정규화, h=16 | 손실 {huber, pinball(다중분위), student_t, tail_weighted} | Q2 |
| **t3_crosssection** | 우승 정규화·손실 | 학습 축 {단일종목, 다종목 pooled(CI), 다종목 pooled(CD)} — 전 종목 표본 | Q3 |
| **t4_ranking** | 우승 구성 | objective {absolute return 회귀, cross-sectional rank} | Q3 |
| **t5_policy** | 우승 예측 | risk gate {none, quantile-based} — 변동 재현 유지 조건 | Q4 |

- 단계마다 최적값을 CLI 인자로 명시 승계(silent default 금지, 15번 패턴).
- 격자 크기를 로그로 남겨 silent truncation 금지.

## 4. 신규 컴포넌트 (엔진 원본 미수정, 16번 드라이버 내 구현)

엔진 모델들은 스칼라 1개를 출력하므로, 16번은 아래를 드라이버에 새로 둔다.

- `RevIN`(nn.Module): instance 정규화-역정규화 + 학습 affine(γ,β). RevIN 근거 ICLR22.
- `MultiOutputHead`: backbone 특징 → {점예측 1개 | 분위 K개 | 분포 파라미터(μ,σ,ν)} 로 분기.
- 손실: `pinball_loss`(다중 분위), `student_t_nll`(DeepAR 계열), `tail_weighted_huber`.
- `CrossSectionalBatcher`: 같은 timestamp의 여러 종목을 한 배치로 묶어 rank 손실 계산(t3/t4).
- 평가: `tail_precision_recall_f1`(큰 변동 구간을 분류로 보고 P/R/F1), `cross_sectional_ic`
  (같은 시점 종목 간 예측-실제 순위상관), 기존 variance_ratio·trend_corr·MASE 재사용.

## 5. 데이터·안정성

- 원천: `upbit_krw_candle`(269종목/1,608만행, 2026-07-19 재수집). 단일종목 비교(t1/t2)는
  진단 속도를 위해 BTC로, 전 종목(t3~t5)은 유동성 상위 N종목 subset부터 확장.
- 13번 crash 교훈: 헤드리스 `.py`, per-case 그림 off, num_workers=0, OOM 자동 배치 축소.
- 결과: `test/results/16_*_20260719/*_raw.md + *.csv`, 그림 `test/images/16_*_20260719/`.

## 6. 평가·성공 기준 (지침 위계 반영)

- **본선**: variance_ratio가 1에 유의하게 가까워지고(진폭 회복), 큰 변동 tail F1과 trend_corr가
  huber 대비 오르며, 검증 곡선 과적합이 관리될 것.
- **일반화**: cross-sectional IC가 양수이고 종목 가로질러 유지될 것.
- **생존 제약**: 위 신호 위에서 MDD가 point-only 대비 개선되되, **변동을 죽여서가 아니라**
  (variance_ratio·active_share 병기로 확인) 방어할 것. MDD 단독 랭킹 금지.
- 성공/실패 무엇이든 정직하게 기록. 큰 변동을 여전히 못 잡으면 그 사실을 horizon·손실별로 남긴다.

## 7. 산출물

- 드라이버: `test/models/16_nonstationary_crosssectional_test.py` (`.py`만, `#%%` 셀 구분 +
  파일 내 마크다운 주석. 사용자 지시 2026-07-19: ipynb 미생성).
- 결과 보고서: `test/results/16_*_report_YYYYMMDD.md` (독립 문서, 그래프별 해석 형식).
- 문헌·초고: 위 2개 문서 참조.
