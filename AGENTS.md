# AGENTS.md - Codex 작업 지침서

## 목적

이 문서는 이 저장소에서 Codex가 매 세션 반드시 따라야 하는 강제 규칙을 정의한다.
영구 규칙은 여기 둔다. 세션별 요청 요약은 `conversation_l2_cache.md`에 짧게 둔다.

## 0. 항상 먼저 읽을 문서

새 세션이 시작되면 아래 순서로 읽는다.

1. `AGENTS.md`
2. `process.md`
3. `history.md`
4. `conversation_l2_cache.md`
5. `test/README.md`
6. `docs/harness/research-workflow/team-spec.md` for recurring research/report workflows

`conversation_l2_cache.md`는 전체를 길게 읽지 말고 최근 항목만 확인한다.

## 1. 영구 규칙

다음 규칙은 캐시가 아니라 항상 적용되는 저장소 규칙이다.

1. 로컬 heavy run 금지, 경량 검증 허용
   - 저장소에는 재현 가능한 자동화 코드와 실행 명령을 남긴다.
   - Codex는 사용자의 개인 연구 세션에서 로컬 터미널 또는 `.venv`로 장시간 분석, 학습, 백테스트, 노트북 결과 산출을 실행하지 않는다.
   - 허용되는 로컬 검증은 문법 검사, import 확인, 작은 synthetic 테스트, 정적 리뷰, diff 리뷰, 짧은 컴파일 점검이다.
   - 실제 연구 실행은 학교 서버 커널, CI, 스케줄러, 또는 사용자가 명시적으로 승인한 원격 환경에서 수행한다.

2. `/test`와 실사용 프레임워크를 분리한다
   - 루트 `quantitative_trading/`는 실사용 프레임워크 코드와 운영 문서를 둔다.
   - `test/`는 연구 실험, 노트북, 리서치 문서, 결과물 전용이다.
   - 운영 진입점은 `pipelines/`에 둔다.
   - 새 워크플로우 플러밍은 `.githooks/` 같은 인프라 폴더에 둔다.

3. 연구 분석은 노트북 원본 + `.py` 미러를 유지한다
   - 새 연구 실험은 `test/models/` 아래 `.ipynb` 원본을 먼저 만든다.
   - 같은 이름의 `.py` 미러를 반드시 함께 유지한다.
   - `.py` 미러는 Git diff 추적과 원격 실행용이며 노트북 대체물이 아니다.
   - 기존 번호 실험의 의미가 달라지는 후속 연구는 같은 파일을 재목적화하지 않고 새 번호(`5 -> 6 -> 7`) 실험으로 분리한다.
   - `.githooks/pre-commit`은 스테이징된 노트북을 자동 동기화하고, 동명 `.ipynb`가 없는 `test/models/*.py`를 차단한다.
   - 노트북 안에서 `argparse`를 쓸 때는 `ipykernel`가 붙이는 `-f kernel.json` 같은 인자를 흡수하도록 `parse_known_args()` 또는 동등한 Jupyter-safe 분기 처리를 넣는다.
   - 노트북 실행용 엔트리포인트는 커널에서 바로 실행될 수 있어야 하며, `__main__` 경로와 notebook cell 경로를 분리해 커널 재실행 시 인자 오류가 나지 않게 한다.
   - 노트북 결과는 원칙적으로 `plt.show()`와 셀 출력으로 본다. `savefig()` 또는 결과 CSV/Markdown 저장은 사용자가 명시적으로 파일 산출을 요청한 경우에만 허용한다.
   - 서버에서 생성된 PNG/CSV/Markdown을 기본 산출물로 삼지 않는다. 보고서용 이미지는 로컬 `test/scripts` 후처리 도구로 노트북 출력에서만 추출한다.

4. 메일은 UTF-8 기준으로 다룬다
   - 한국어 메일 본문은 UTF-8 또는 MIME-safe 방식으로 보낸다.
   - PowerShell inline here-string으로 한글 SMTP 본문을 직접 조합하는 방식은 피한다.
   - 재사용 가능한 메일 발송 도구만 `test/scripts/`에 둔다.

5. branch 정책을 지킨다
   - 사용자가 별도 지시하지 않으면 작업 브랜치에서만 커밋/푸시한다.
   - `main` 또는 `develop` 병합은 사용자가 명시적으로 요청한 경우에만 수행한다.
   - 보고서/메일 링크는 현재 작업 브랜치 기준 링크를 사용한다.

6. 새 파일을 만들기 전 먼저 검색하고 재사용한다
   - `AGENTS.md`, `process.md`, `history.md`, `conversation_l2_cache.md`, `test/README.md`, `pipelines/`, `test/scripts/`, 기존 `test/models/*.ipynb/*.py`를 먼저 확인한다.
   - 새 파일보다 기존 파일 수정 또는 확장을 우선한다.

7. `test/scripts/`에는 일회성 Python 파일을 만들지 않는다
   - 허용: 노트북 빌더, 보고서 변환기, 이미지 추출기, 환경 복구기, 메일/리포트 전달기 같은 재사용 도구
   - 금지: 이번 한 번만 쓰는 ad-hoc 스크립트

## 2. 실행 주체 분리

### 저장소에 남길 것

- 다른 연구원이나 운영자가 실행할 수 있는 자동화 스크립트
- `uv run ...` 형식의 재현 실행 명령 예시
- n8n, Cron, CI, Docker, Kubernetes 같은 운영 자동화 설계
- 학교 서버 커널에서 실행할 절차와 파라미터

### Codex가 이 세션에서 할 수 있는 것

- 코드 작성과 수정
- 정적 코드 리뷰, diff 리뷰, 설계 리뷰
- 문법 검사, import 확인, 작은 synthetic 테스트
- 논문 조사와 설계 근거 정리
- 연구용 `.ipynb` 작성과 `.py` 미러 동기화

### Codex가 하지 않는 것

- `uv run main.py` 같은 장시간 분석/학습/백테스트 파이프라인 로컬 실행
- `.ipynb` 실행을 통한 연구 결과 산출
- `.venv` 또는 로컬 Python으로 대용량 연구 수행 코드 실행
- 결과 수치 생성을 목적으로 한 대규모 DB/시계열 분석 실행

## 3. 분석 설계 원칙

1. 2026-05 연구 설계를 유지한다
   - PreprocessingPipeline의 핵심은 정상성 진단, 변환 비교/선택, Lag-1 shift 또는 copy-risk 기록이다.
   - 모든 데이터를 하나의 정상 표현으로 강제하지 않는다.
   - 원시 가격 수준만 그대로 학습시키지 않는다.
   - 정상성 검정, 롤링 드리프트 점검, log return, diff, rolling z-score, EMA 변형, KRW 역복원 지표를 함께 사용한다.
2. 보고서 세부 기준은 `test/README.md`를 따른다.
   - 실행 및 분석 환경
   - 기초 통계량과 정상성
   - 사용한 방법론/지표/손실함수/진단 도구의 개념, 사용 이유, 수식 또는 정의, 해석 예시, 장단점
   - KRW 원본 스케일 기준 성능 지표
   - DA와 MASE
   - 축/범례를 포함한 시각화 해석
   - 그림은 기본적으로 노트북 셀에서 `plt.show()`로 확인하고, 파일 저장은 예외적으로만 사용한다.
   - MDD 최소화 관점의 종합 결론
   - 용어 해설과 개발/디버깅 기록
   - 새 보고서는 이전 보고서에 같은 설명이 있더라도 핵심 방법론/지표/그래프 해석 기준을 다시 적는 독립 문서로 작성한다.

3. 모델 철학
   - 금융 시계열은 "Shallow but Wide" 원칙을 유지한다.
   - 레이어는 1~2층, width는 대체로 64~128 범위를 우선 검토한다.

## 4. 재현 실행 명령

아래 명령은 학교 서버, CI, 스케줄러, 운영자가 재현 실행할 수 있도록 저장소에 남기는 기준 명령이다.

- `uv run main.py`
- `uv run pipelines/ingest_text_context.py`
- `uv run pipelines/build_historical_flow_mart.py`
- `uv run pipelines/query_historical_flows.py`
- `uv run pipelines/simulate_and_send.py`

## 5. Git 및 전달 규칙

- 보고서나 결과물이 생성되면 자동 발송 스크립트는 `git add`, `git commit`, `git push origin <branch>`를 수행할 수 있어야 한다.
- 푸시 후에는 GitHub 렌더링 URL을 계산해 메일 본문 상단에 넣는다.
- 메일 본문에는 핵심 개선점과 보고서 접근 링크를 함께 넣는다.

