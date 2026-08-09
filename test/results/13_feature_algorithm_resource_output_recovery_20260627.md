# 13번 노트북 출력 복구 메모

## 목적

이 문서는 `test/models/13_feature_algorithm_resource_test.ipynb` 실행 후

1. 어떤 에러가 났는지
2. 왜 노트북 저장과 다른 이름 저장이 실패했는지
3. 현재 로컬 디스크에서 실제로 복구 가능한 결과가 무엇인지
4. 추가로 살릴 수 있는 경로가 무엇인지

를 빠르게 정리한 포렌식 메모다.

## 먼저 결론

- **실행 중단의 직접 원인**은 PyTorch `DataLoader` worker가 shared memory(`/dev/shm`)를 더 할당하지 못한 것이다.
- **저장 실패의 직접 원인**은 VSCode notebook renderer가 노트북 snapshot/backup을 만들려는 단계에서 반복적으로 `Notebook too large to backup` 에러를 낸 것이다.
- 즉 이번 13번은
  - 계산 쪽에서는 `shared memory 부족`
  - 저장 쪽에서는 `노트북 출력이 너무 커서 VSCode가 backup/snapshot 생성 실패`
  라는 두 문제가 겹쳤다.
- 현재 로컬의 `.ipynb` 파일에는 **실제 결과 표, 그래프, traceback 출력이 저장되지 않았다.**
- 따라서 **지금 디스크만 가지고는 13번 전체 결과를 완전 복원할 수 없다.**

## 현재 로컬 파일 상태

- 대상 노트북: `test/models/13_feature_algorithm_resource_test.ipynb`
- 현재 로컬 파일 크기: `40,881 bytes`
- 현재 로컬 파일 내용: 1개 코드 셀짜리 실행 엔트리포인트 상태
- 저장된 output:
  - `image/png`: 0개
  - `stream`: 0개
  - `error`: 0개
- `git diff` 기준 변경점도 커널 메타데이터 정도만 남아 있고, 결과 출력은 파일에 serialize되지 않았다.

즉 화면에서는 큰 출력이 보였지만, 디스크의 노트북 JSON에는 그 출력이 내려오지 못했다.

## 화면상 출력이 실제로 있었던 근거

VSCode notebook view state에는 13번 노트북 셀 총 높이가 약 `196,677px`로 남아 있다.

- 해석:
  - 노트북 UI 안에서는 매우 큰 출력이 실제로 생성되었다.
  - 그러나 그 출력이 `.ipynb` JSON 저장으로 이어지지 못했다.

## 실제 실행 에러

Jupyter 확장 로그에 남은 핵심 에러는 아래와 같다.

```text
RuntimeError: unable to allocate shared memory(shm) for file </torch_2521_2987317372_2>: Resource temporarily unavailable (11)
```

좀 더 풀어 쓰면 다음 뜻이다.

- `DataLoader`가 worker process에서 batch를 모을 때 tensor를 shared memory에 올리려 했다.
- 그런데 서버의 `/dev/shm` 또는 shared memory quota가 부족해서 새 공유 메모리 블록을 만들지 못했다.
- 그래서 학습 루프 진입 또는 batch fetch 단계에서 `RuntimeError`가 났다.

### 로그상 traceback 흐름

1. `13_feature_algorithm_resource_test.py`의 `main()`
2. `run_one_case(...)`
3. `fusion12.run_point_branch(...)`
4. `10_objective_ensemble_confirmation_test.py`의 `train_model(...)`
5. `for xb, yb in train_loader`
6. `torch.utils.data`의 collate 단계
7. shared storage 생성 실패

즉 모델 수식이나 손실함수 자체보다 먼저, **데이터 적재 단계에서 시스템 자원 제약으로 멈춘 케이스**다.

## 왜 저장이 안 됐는가

VSCode renderer 로그에는 아래 에러가 반복 기록되어 있다.

```text
Error: Notebook too large to backup
```

의미는 다음과 같다.

- VSCode notebook editor는 저장/복구를 위해 내부 snapshot 또는 backup을 만든다.
- 그런데 현재 노트북의 출력이 너무 커져서 이 snapshot 생성 자체가 실패했다.
- 그래서
  - 일반 저장
  - 다른 이름으로 저장
  - hot-exit backup
  중 적어도 backup 경로는 계속 실패했을 가능성이 높다.

즉 지금 증상은 단순 파일 권한 문제가 아니라,

1. **노트북 출력량이 너무 커졌고**
2. **VSCode notebook backup이 그 크기를 감당하지 못했으며**
3. **그 와중에 실제 실행도 shared memory 부족으로 중단**

한 복합 장애에 가깝다.

## 왜 “다른 이름으로 저장”도 잘 안 되었을 가능성이 높은가

`Save As`는 파일 경로만 바꾸는 동작처럼 보이지만, 내부적으로는 현재 notebook model을 다시 serialize해야 한다.

이번 상태에서는 notebook model 자체가

- 매우 긴 stdout / traceback / 표 출력
- 큰 notebook output tree
- backup snapshot 실패

상태였기 때문에, 파일명을 바꾸더라도 동일한 큰 notebook payload를 다시 써야 해서 실패했을 가능성이 높다.

## 지금 복구 가능한 것

### 복구 가능

- 실행에 사용한 커널 종류
- 실행 시각
- 마지막 에러 종류와 traceback
- 저장 실패 원인
- 화면에 큰 출력이 실제로 있었다는 정황

### 복구 불가

아래 항목은 현재 로컬 디스크만으로는 복구되지 않는다.

- 전체 leaderboard
- feature group 평균표
- model family 평균표
- heatmap
- diagnostic plot 이미지
- 마지막 성공 case까지의 실제 수치 표
- notebook 안 markdown/display 출력

이유는 이 정보들이 `.ipynb` 파일에 저장되지 않았기 때문이다.

## 지금 시점에서 동일하게 저장할 수 있는가

### 로컬 디스크만으로는 불가능

현재 저장소와 VSCode 로컬 히스토리에는 큰 출력이 들어간 notebook snapshot이 없다.

### 단, VSCode 창이 아직 살아 있고 출력이 화면에 남아 있다면

그 경우에만 마지막 기회가 있다.

가능한 순서는 다음이 가장 안전하다.

1. 노트북 탭을 닫지 않는다.
2. 커널을 재시작하지 않는다.
3. 출력 전체를 접거나 줄이려 하지 않는다.
4. 화면상으로 필요한 표와 그래프를 우선 복사 또는 캡처한다.
5. 별도 Markdown 파일로 직접 옮긴다.

이번 세션의 핵심은 **노트북 자체 저장 복구보다 화면상 결과를 다른 산출물로 옮겨 놓는 것**이다.

## 왜 지금 “전체 결과 자동 파싱”이 막혔는가

Codex가 자동 파싱하려면 원칙적으로 아래 둘 중 하나가 있어야 한다.

1. output이 저장된 `.ipynb`
2. output이 저장된 HTML/PNG/CSV/Markdown

그런데 현재는 둘 다 없다.

따라서 자동 파싱 가능한 입력 원본이 없는 상태다.

## 다음 실험에서 같은 문제를 막는 실무적 수정 방향

### 실행 쪽

- `DataLoader` worker 수를 `0` 또는 `1`로 낮춘다.
- `batch_size`를 더 공격적으로 낮춘다.
- `num_workers > 0`일 때 shared memory를 많이 먹는 구간을 줄인다.

### 저장 쪽

- case별 전체 플롯을 모두 notebook output으로 누적하지 않는다.
- `case_plots`를 기본 `False`로 두고 대표 case만 켠다.
- leaderboard/요약표는 notebook 최하단에 한 번만 출력한다.
- full suite는 suite를 더 잘게 쪼개서 한 노트북 output volume을 줄인다.

## 참고 로그

- Jupyter 실행 로그:
  - `%APPDATA%\\Code\\logs\\20260624T203952\\window1\\exthost\\output_logging_20260624T203954\\7-Jupyter.log`
- VSCode renderer 로그:
  - `%APPDATA%\\Code\\logs\\20260624T203952\\window1\\renderer.log`

이 두 파일이 이번 장애의 가장 직접적인 근거다.
