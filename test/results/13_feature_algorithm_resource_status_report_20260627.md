# 13번 feature-algorithm-resource 실험 상태 보고서

## 한눈에 먼저

이번 13번 실험은 `test/models/13_feature_algorithm_resource_test.ipynb`에서 실제로 대규모 출력이 생성되었지만,

1. 실행 말미에 PyTorch `DataLoader` shared memory 부족 에러가 발생했고
2. 동시에 VSCode notebook 쪽에서는 output volume이 너무 커져 `Notebook too large to backup` 문제가 발생했다.

그 결과, **화면에는 결과가 있었지만 현재 디스크에 저장된 `.ipynb` 파일에는 결과가 serialize되지 않았다.**

따라서 이번 보고서는

- 13번에서 지금 확정적으로 말할 수 있는 것
- 지금은 확정적으로 말할 수 없는 것
- 결과 복구 가능성
- 다음 실행 때 반드시 바꿔야 할 점

을 정리하는 상태 보고서다.

## 이번에 확정된 사실

### 1. 실행 에러의 직접 원인

Jupyter 로그 기준 직접 에러는 아래와 같다.

```text
RuntimeError: unable to allocate shared memory(shm) for file </torch_2521_2987317372_2>: Resource temporarily unavailable (11)
```

의미를 풀면 다음과 같다.

- 모델 구조 자체가 바로 잘못된 것이 아니라
- 학습용 `DataLoader`가 batch를 worker process에서 shared memory로 넘기는 단계에서
- 서버의 shared memory 한도를 넘겨
- 학습 루프가 중단된 것이다.

즉 이번 중단은 우선적으로 **자원 설정 문제**다.

### 2. 저장 실패의 직접 원인

VSCode renderer 로그에는 아래 메시지가 반복적으로 남아 있다.

```text
Error: Notebook too large to backup
```

의미는 다음과 같다.

- 노트북 화면에는 매우 긴 표, 출력, 그래프가 누적되었고
- VSCode가 이를 저장/복구용 snapshot으로 만드는 단계에서 실패했으며
- 그래서 일반 저장이나 다른 이름 저장도 함께 불안정해졌을 가능성이 높다.

즉 이번 저장 실패는 파일 권한 문제라기보다 **노트북 출력량 과대 문제**에 가깝다.

### 3. 현재 로컬 `.ipynb` 파일 상태

현재 저장된 로컬 파일은 사실상 실행 엔트리포인트만 남아 있다.

- 파일: `test/models/13_feature_algorithm_resource_test.ipynb`
- 크기: 약 `40 KB`
- 저장된 이미지 output: `0`
- 저장된 stream output: `0`
- 저장된 error output: `0`

즉 지금 남아 있는 `.ipynb`는 결과 원본 역할을 못 한다.

### 4. 화면에 큰 결과가 있었던 정황

VSCode notebook view state에는 해당 셀 높이가 약 `196,677px`로 남아 있다.

이건 적어도 아래를 뜻한다.

- 실행 도중 화면에는 실제로 매우 긴 결과가 생성되었다.
- 다만 그 결과가 파일 저장까지 이어지지 못했다.

## 이번에 아직 확정할 수 없는 것

현재 디스크 기준으로는 아래를 확정할 수 없다.

- 전체 leaderboard
- feature group 평균 요약
- model family 평균 요약
- feature × model heatmap
- representative diagnostic plots 원본
- 마지막까지 성공한 case 수
- top case 정확 수치

이유는 간단하다.

**결과가 파일로 저장되지 않았기 때문이다.**

즉 지금 남은 로컬 파일만 가지고는 “13번 결과 보고서 본문을 완전하게 다시 생성”할 수 없다.

## 그래서 지금 가능한 복구 범위

### 가능한 것

- 에러 원인 규명
- 저장 실패 원인 규명
- 커널/서버/실행 정황 복원
- 다음 재실행 시 수정해야 할 자원 정책 제안
- 사용자가 아직 열어둔 화면이 있다면, 거기서 수동 구조화할 가이드 제공

### 불가능한 것

- 저장되지 않은 표/그래프를 자동으로 100% 복원
- 디스크만 가지고 전체 결과 Markdown을 완전히 재생성

## “동일하게 저장”할 수 있나

### 이미 디스크에 없는 상태라면 사실상 불가능

현재 워크트리, VSCode local history, notebook JSON 기준으로는 출력이 들어간 저장본이 없다.

따라서 **지금 닫힌 결과를 동일하게 다시 저장하는 일은 디스크만으로는 불가능**하다.

### 단, 아직 VSCode 탭이 살아 있으면 마지막 기회는 있다

아직 그 노트북 탭이 닫히지 않았고, 출력이 화면에 남아 있다면 아래 순서가 가장 현실적이다.

1. 노트북 탭을 닫지 않는다.
2. 커널을 재시작하지 않는다.
3. 출력 collapse/clear를 하지 않는다.
4. 우선 leaderboard, 평균표, 대표 그래프부터 캡처 또는 복사한다.
5. 별도 Markdown 파일로 옮긴다.

즉 이번 상태에서는 노트북 자체 저장 복구보다 **화면 결과를 2차 산출물로 대피시키는 것**이 중요하다.

## 왜 “다른 이름으로 저장”도 실패했을 가능성이 큰가

겉으로는 파일명만 바꾸는 동작처럼 보여도, 실제로는 현재 notebook model 전체를 다시 serialize해야 한다.

그런데 이번 notebook model은

- 매우 긴 출력
- 많은 그래프/표
- backup snapshot 실패

상태였기 때문에, 파일명을 바꿔도 결국 같은 큰 payload를 다시 저장해야 해서 실패했을 가능성이 높다.

## 이번 13번을 연구 관점에서 어떻게 봐야 하나

이번 13번은 아직 **성능 결론 실험**이 아니라 **실행 구조 검증에서 멈춘 실험**으로 봐야 한다.

즉 지금 시점에서 말할 수 있는 결론은

- “어떤 변수군이 최선이었다”
- “어떤 모델이 top1이었다”
- “risk gate 어떤 조합이 제일 좋았다”

가 아니라,

- “현재 13번 설계는 notebook output volume과 DataLoader shared memory 사용량이 너무 커서 그대로는 안정적으로 완주되지 않는다”
- “full_resource 급 실험은 출력 전략과 worker 정책을 먼저 바꿔야 한다”

에 가깝다.

## 다음 실행에서 반드시 바꿔야 할 것

### 1. DataLoader shared memory 사용량 줄이기

우선순위는 다음이 좋다.

1. `num_workers=0` 또는 `1`
2. `batch_size` 기본값을 `1024`에서 시작하되, 필요하면 `512`로 즉시 하향
3. `pin_memory`, 큰 collate payload, 불필요한 복제 여부 점검
4. suite를 더 쪼개서 한 번에 도는 case 수 축소

### 2. notebook output량 줄이기

가장 중요하다.

- 모든 case 그래프를 매번 다 그리지 않기
- case별 상세 플롯은 대표 case만 출력
- 중간 진행 로그는 짧게
- 큰 표는 top-k와 summary만 우선 출력
- 전체 raw table은 notebook이 아니라 후속 Markdown/CSV 산출물로 보내기

### 3. full_resource를 한 노트북에서 한 번에 끝내려 하지 않기

다음처럼 분리하는 쪽이 안전하다.

- `mtf_decomposition`
- `algorithm_screen`
- `feature_algorithm_matrix`
- `risk_gate_sensitivity`
- `full_resource_top_candidates_only`

즉 최종 full matrix도 한 번에 다 찍지 말고, **후보 압축 후 심화** 형태로 가는 편이 낫다.

## 지금 바로 필요한 실무 판단

### A. 사용자가 아직 화면을 볼 수 있다면

가장 먼저 해야 할 일:

- leaderboard
- feature group average
- model family average
- final decision summary
- 대표 그래프 몇 장

을 먼저 외부 Markdown으로 옮긴다.

### B. 화면도 이미 날아갔다면

이번 13번은 결과 보고서가 아니라 **장애/복구/재실행 계획 보고서**로 마감하고,
다음 실행에서 출력 정책과 shared memory 정책을 수정한 뒤 다시 돌리는 것이 맞다.

## 관련 산출물

- 장애 원인 포렌식 메모:
  - `test/results/13_feature_algorithm_resource_output_recovery_20260627.md`

## 핵심 로그 경로

- Jupyter 실행 로그:
  - `%APPDATA%\\Code\\logs\\20260624T203952\\window1\\exthost\\output_logging_20260624T203954\\7-Jupyter.log`
- VSCode renderer 로그:
  - `%APPDATA%\\Code\\logs\\20260624T203952\\window1\\renderer.log`

## 최종 정리

이번 13번은 “모델/변수 성능 결론”보다 먼저,

- shared memory 자원 한계
- notebook output 과대
- 저장 실패

가 드러난 실행 구조 이슈를 확인한 실험이었다.

현재 디스크만으로는 전체 결과를 완전 복원할 수 없으므로, 이번 회차는

1. 장애 원인 기록
2. 현재 복구 가능 범위 정리
3. 화면이 살아 있으면 핵심 결과 수동 대피
4. 다음 실행에서 worker/output 정책 수정

순으로 처리하는 것이 가장 합리적이다.
