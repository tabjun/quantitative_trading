# CLAUDE.md - Claude Code 작업 지침서

## 목적

이 문서는 이 저장소에서 Claude Code가 매 세션 반드시 따라야 하는 강제 규칙을 정의한다.
`AGENTS.md`(Codex용)와 동일한 규칙을 공유하므로, Claude와 Codex 중 어느 도구를 써도
흐름이 끊기지 않고 작업을 이어받을 수 있다.

도구별 사설 memory는 보조 정보일 뿐이다. 현재 상태의 source of truth는 저장소의
`CLAUDE.md`, `AGENTS.md`, `process.md`, `history.md`, `conversation_l2_cache.md`,
`test/README.md`, 그리고 관련 결과/보고서 파일이다.

---

## 0. 항상 먼저 읽을 문서

새 세션이 시작되면 아래 순서로 읽는다.

1. `CLAUDE.md` (이 파일)
2. `process.md` — 현재 단계와 다음 할 일
3. `history.md` — 누적 작업 이력 (최근 섹션만 먼저 확인)
4. `conversation_l2_cache.md` — 최근 사용자 의도 캐시 (최근 항목만)
5. `test/README.md` — 연구 실험 공간 안내
6. `docs/harness/research-workflow/team-spec.md` — 반복 연구/보고서 흐름 지침
7. 직전 작업자가 Codex였던 정황이 있으면 `AGENTS.md`도 확인한다.

> **이유**: 세션이 바뀌거나 Claude ↔ Codex를 전환해도 이 5개 파일을 읽으면
> 프로젝트 현황과 다음 작업을 즉시 복원할 수 있다.

---

## 1. Claude ↔ Codex 병행 사용 규칙

이 프로젝트는 **Claude Code**와 **Codex**를 함께 사용한다.
어느 도구로 작업하든 흐름이 끊기지 않도록 다음을 지킨다.

### 세션 종료 전 반드시 할 것

어느 도구를 쓰든 세션을 마치기 전에:

- `history.md` 하단 표에 수행 내용, 산출물, 다음 목표를 한 행 추가한다.
- `process.md`의 현재 단계와 체크리스트를 최신화한다.
- 의미 있는 방향 전환이 있었다면 `conversation_l2_cache.md`에 한 행 추가한다.
- 작업이 중단되었거나 복구 중이면, 완료처럼 쓰지 말고 `어디까지 확보했는지`,
  `아직 못 한 것`, `다음 도구가 이어서 할 일`을 명시한다.

### 세션 시작 시 복원 순서

1. `process.md`에서 현재 Phase/Step과 미완료 체크리스트를 확인한다.
2. `history.md` 최근 섹션에서 마지막으로 완료된 작업을 확인한다.
3. `conversation_l2_cache.md` 최근 항목에서 사용자 선호와 제약을 확인한다.
4. 이전 세션이 Codex였든 Claude였든 동일한 방식으로 이어간다.

### 두 도구의 차이점 (Claude Code 기준)

| 항목 | Codex | Claude Code |
|---|---|---|
| 설정 파일 | `AGENTS.md` | `CLAUDE.md` (이 파일) |
| MCP 설정 | `.codex/config.toml` | `.mcp.json` |
| 기본 명령어 | `codex` CLI | `claude` CLI / VSCode 확장 |
| 기억 복원 | `AGENTS.md` 읽기 | `CLAUDE.md` 읽기 |

두 설정은 같은 규칙을 공유하므로, 어느 도구를 먼저 써도 다른 도구로 이어받을 수 있다.

---

## 2. 영구 규칙

다음 규칙은 캐시가 아니라 항상 적용되는 저장소 규칙이다.

### 2.1 로컬 heavy run 금지, 경량 검증 허용

- 저장소에는 재현 가능한 자동화 코드와 실행 명령을 남긴다.
- Claude Code는 사용자의 개인 연구 세션에서 로컬 터미널 또는 `.venv`로
  장시간 분석, 학습, 백테스트, 노트북 결과 산출을 실행하지 않는다.
- **허용되는 로컬 검증**: 문법 검사, import 확인, 작은 synthetic 테스트,
  정적 리뷰, diff 리뷰, 짧은 컴파일 점검.
- 실제 연구 실행은 학교 서버 커널, CI, 스케줄러, 또는
  사용자가 명시적으로 승인한 원격 환경에서 수행한다.

### 2.2 `/test`와 실사용 프레임워크를 분리한다

- 루트 `quantitative_trading/`는 실사용 프레임워크 코드와 운영 문서를 둔다.
- `test/`는 연구 실험, 노트북, 리서치 문서, 결과물 전용이다.
- `/test` 관련 요청은 먼저 연구 목적과 artifact 경계를 평가하고,
  목적에서 벗어나면 그대로 수행하지 말고 연구 질문에 맞게 비판적으로 재구성한다.
- 운영 진입점은 `pipelines/`에 둔다.
- 새 워크플로우 플러밍은 `.githooks/` 같은 인프라 폴더에 둔다.

### 2.3 연구 분석은 노트북 원본 + `.py` 미러를 유지한다

- 새 연구 실험은 `test/models/` 아래 `.ipynb` 원본을 먼저 만든다.
- 같은 이름의 `.py` 미러를 반드시 함께 유지한다.
- `.py` 미러는 Git diff 추적과 원격 실행용이며 노트북 대체물이 아니다.
- 기존 번호 실험의 의미가 달라지는 후속 연구는 같은 파일을 재목적화하지 않고
  새 번호(`5 -> 6 -> 7`) 실험으로 분리한다.
- `.githooks/pre-commit`은 스테이징된 노트북을 자동 동기화하고,
  동명 `.ipynb`가 없는 `test/models/*.py`를 차단한다.
- 노트북 안에서 `argparse`를 쓸 때는 `ipykernel`이 붙이는 `-f kernel.json` 같은
  인자를 흡수하도록 `parse_known_args()` 또는 동등한 Jupyter-safe 분기 처리를 넣는다.
- 노트북 결과는 원칙적으로 `plt.show()`와 셀 출력으로 본다.
  `savefig()` 또는 결과 CSV/Markdown 저장은 사용자가 명시적으로 파일 산출을
  요청한 경우에만 허용한다.
- 분석이 완료된 모든 연구 `.ipynb`는 read-only source of truth로 다룬다.
  보고서용 이미지는 노트북 파일을 변경하지 않는 후처리 도구로만 추출한다.
- 완료된 노트북은 보고서 작성, 메일 작성, 이미지 추출, 결과 재정리 단계에서
  수정, 재저장, strip, output 초기화, 코드 셀 재작성 대상으로 삼지 않는다.
- 완료 결과의 재정리는 `test/results/*.md`, `test/images/*`, 메일 preset 같은
  후속 산출물에서만 수행한다.
- 완료된 기존 번호 실험을 다시 분석하거나 의미를 바꿔야 하면 기존 파일을
  재목적화하지 않고 새 번호 실험으로 분리한다.
- 예외는 사용자가 해당 노트북 파일 수정을 명시 승인한 경우뿐이다.

### 2.4 메일은 UTF-8 기준으로 다룬다

- 한국어 메일 본문은 UTF-8 또는 MIME-safe 방식으로 보낸다.
- 재사용 가능한 메일 발송 도구만 `test/scripts/`에 둔다.
- 연구 메일과 보고서에서 전문 용어를 처음 사용할 때는 용어 이름만 나열하지 않는다.
  `쉬운 정의 -> 숫자 또는 상황 예시 -> 이번 실험에서 어떻게 계산/작용했는지 ->
  좋은 신호인지 나쁜 신호인지와 예외` 순서로 설명한다.
- `persistence`, `collapse`, `variance ratio`, `direction accuracy`, `MASE`,
  `conformal interval`처럼 결과 해석을 좌우하는 용어는 독자가 이전 문서를
  읽었다고 가정하지 않고 매 문서와 메일에서 다시 풀어 쓴다.
- 새 방법론이나 실험 축을 추가하라는 요청이 오면 먼저 그것이 현재 연구 질문을
  실제로 강화하는지, 기존 축과 중복되거나 목적을 흐리는지 적대적으로 검토한 뒤에만 반영한다.

### 2.5 수행환경을 history/캐시에 명시한다

- `history.md`와 `conversation_l2_cache.md`의 모든 이력 행 마지막 컬럼 `수행환경`에 반드시 값을 기입한다.
- `로컬`: Claude/Codex 세션에서 수행한 작업 (코드 작성, 문서, 설정, 정적 리뷰, 커밋, 메일 발송 등).
- `서버`: 학교 서버 커널에서 직접 실행한 작업 (노트북 모델 학습, 대규모 실험, GPU 실행 등).
- 로컬과 서버 작업이 같은 세션에서 섞이면 행을 분리하거나 `로컬+서버`로 명시한다.
- 2026-06-28 이전 기록은 구분 이력이 없어 일괄 `로컬`로 표기했다.

### 2.7 branch 정책을 지킨다

- 사용자가 별도 지시하지 않으면 작업 브랜치에서만 커밋/푸시한다.
- `main` 또는 `develop` 병합은 사용자가 명시적으로 요청한 경우에만 수행한다.
- 보고서/메일 링크는 현재 작업 브랜치 기준 링크를 사용한다.
- 이 저장소에서 커밋/푸시가 필요하면 GitHub SSH 원격(`git@github.com:tabjun/personal_ai_project.git`)을 기본 경로로 사용한다.
- 학교 서버나 로컬 세션에서 SSH 키 패스프레이즈가 뜨면, 현재 세션의 `ssh-agent`와 `~/.ssh/config` 설정을 우선 재사용해 반복 입력을 줄인다.
- 메일 본문에는 rendered Markdown 보고서 링크만 넣고, commit history 링크는 넣지 않는다.

### 2.8 새 파일을 만들기 전 먼저 검색하고 재사용한다

- `CLAUDE.md`, `process.md`, `history.md`, `conversation_l2_cache.md`,
  `test/README.md`, `pipelines/`, `test/scripts/`, 기존 `test/models/*.ipynb/*.py`를
  먼저 확인한다.
- 새 파일보다 기존 파일 수정 또는 확장을 우선한다.

### 2.9 `test/scripts/`에는 일회성 Python 파일을 만들지 않는다

- 허용: 노트북 빌더, 보고서 변환기, 이미지 추출기, 환경 복구기,
  메일/리포트 전달기 같은 재사용 도구
- 금지: 이번 한 번만 쓰는 ad-hoc 스크립트

---

## 3. 실행 주체 분리

### 저장소에 남길 것

- 다른 연구원이나 운영자가 실행할 수 있는 자동화 스크립트
- `uv run ...` 형식의 재현 실행 명령 예시
- n8n, Cron, CI, Docker, Kubernetes 같은 운영 자동화 설계
- 학교 서버 커널에서 실행할 절차와 파라미터

### Claude Code가 이 세션에서 할 수 있는 것

- 코드 작성과 수정
- 정적 코드 리뷰, diff 리뷰, 설계 리뷰
- 문법 검사, import 확인, 작은 synthetic 테스트
- 논문 조사와 설계 근거 정리
- 연구용 `.ipynb` 작성과 `.py` 미러 동기화

### Claude Code가 하지 않는 것

- `uv run main.py` 같은 장시간 분석/학습/백테스트 파이프라인 로컬 실행
- `.ipynb` 실행을 통한 연구 결과 산출
- `.venv` 또는 로컬 Python으로 대용량 연구 수행 코드 실행
- 결과 수치 생성을 목적으로 한 대규모 DB/시계열 분석 실행

---

## 4. 분석 설계 원칙

1. **2026-05 연구 설계를 유지한다**
   - PreprocessingPipeline의 핵심은 정상성 진단, 변환 비교/선택,
     Lag-1 shift 또는 copy-risk 기록이다.
   - 모든 데이터를 하나의 정상 표현으로 강제하지 않는다.
   - 원시 가격 수준만 그대로 학습시키지 않는다.
   - 정상성 검정, 롤링 드리프트 점검, log return, diff, rolling z-score,
     EMA 변형, KRW 역복원 지표를 함께 사용한다.

2. **보고서 세부 기준은 `test/README.md`를 따른다**
   - 실행 및 분석 환경, 기초 통계량, 방법론/지표/손실함수/진단 도구의 개념,
     KRW 원본 스케일 기준 성능 지표, DA와 MASE, 시각화 해석을 포함한다.
   - 포함한 그래프마다 `데이터/대상 모델 -> x/y축 -> 진단 목적 ->
     실제 관찰 -> 좋은지 나쁜지 -> 다음 실험 반영 방향`을 설명한다.
   - 새 보고서는 이전 보고서에 같은 설명이 있더라도 독립 문서로 작성한다.

3. **모델 철학**
   - 금융 시계열은 "Shallow but Wide" 원칙을 유지한다.
   - 레이어는 1~2층, width는 대체로 64~128 범위를 우선 검토한다.

---

## 5. MCP 도구 설정

`.mcp.json`에 arxiv MCP 서버가 등록되어 있다.
Claude Code 세션에서 논문 검색이 필요하면 `mcp__arxiv__*` 도구를 사용한다.

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "arxiv-mcp-server",
      "args": []
    }
  }
}
```

`arxiv-mcp-server`가 PATH에 없으면:
```bash
uv tool install --managed-python --python 3.12 git+https://github.com/blazickjp/arxiv-mcp-server.git
```

---

## 6. 재현 실행 명령

아래 명령은 학교 서버, CI, 스케줄러, 운영자가 재현 실행할 수 있도록
저장소에 남기는 기준 명령이다.

```bash
uv run main.py
uv run pipelines/ingest_text_context.py
uv run pipelines/build_historical_flow_mart.py
uv run pipelines/query_historical_flows.py
uv run pipelines/simulate_and_send.py
```

---

## 7. Git 및 전달 규칙

- 보고서나 결과물이 생성되면 자동 발송 스크립트는
  `git add`, `git commit`, `git push origin <branch>`를 수행할 수 있어야 한다.
- 이 저장소에서 `origin`이 HTTPS로 잡혀 있으면, push 전에 GitHub SSH 원격으로 맞춘다.
- Claude Code와 Codex CLI 모두 같은 SSH 원격과 세션 `ssh-agent`를 쓰는 것을 기준으로 문서를 읽고 동작한다.
- 푸시 후에는 해당 Markdown 보고서를 GitHub에서 렌더링한 URL만
  메일 본문 상단에 넣는다. 커밋 히스토리 링크는 넣지 않는다.
- 메일 본문에는 핵심 개선점과 보고서 접근 링크를 함께 넣는다.

---

## 8. 현재 연구 상태 요약

> 세션 시작 시 이 섹션 대신 `process.md`와 `history.md`를 읽어 최신 상태를 확인한다.
> 아래는 2026-06-28 기준 스냅샷이며, 최신 여부는 항상 `history.md`와
> `conversation_l2_cache.md`로 재확인한다.

- **현재 브랜치**: `stock`
- **가장 최근 실험 상태**: 13번 (`13_feature_algorithm_resource_test`)은 서버/원격 커널에서
  대규모 실행 후 VSCode 저장 실패와 PyTorch `DataLoader` shared memory 에러가 발생했다.
- **확보된 산출물**:
  - `test/results/13_feature_algorithm_resource_output_recovery_20260627.md`
  - `test/results/13_feature_algorithm_resource_status_report_20260627.md`
  - `test/results/13_feature_algorithm_resource_live_recovery_dump_20260627.md`
  - `test/images/13_output_recovery_20260627/capture_*.png`
- **당장 할 일**: 열려 있는 VSCode output이 아직 남아 있으면 leaderboard, feature group average,
  model family average, final summary, 마지막 traceback을 추가 대피한다. 화면이 사라졌다면
  이번 회차는 장애/복구 보고로 마감하고 `num_workers`, batch size, output volume을 줄인
  후속 실행을 준비한다.
- **핵심 기조**: 다변량 분석, MDD 최소화 우선, Defense First
- **데이터 베이스**: `upbit_data.db` (DuckDB, `data/` 아래)
- **활성 MCP 기준**: Claude Code는 `.mcp.json`의 `arxiv`를 우선 사용한다. Codex 쪽은
  전역 설정 상태에 따라 `arxiv`, `node_repl` 등이 보일 수 있으나, GitHub/Notion/Data Analytics
  플러그인은 2026-06-27 startup 안정화를 위해 비활성화된 이력이 있다.

---

## 9. 도구 전환 시 체크리스트

### Claude → Codex로 전환할 때

1. 이번 세션에서 변경한 파일을 커밋하거나 메모한다.
2. `history.md`에 수행 내용을 기록한다.
3. `process.md`의 다음 스텝을 업데이트한다.
4. Codex를 열면 `AGENTS.md -> process.md -> history.md` 순서로 읽는다.
5. Claude memory에만 남긴 내용은 Codex가 못 볼 수 있으므로, 이어받아야 하는 결정은
   반드시 저장소 파일에 남긴다.

### Codex → Claude로 전환할 때

1. Codex 세션에서 변경한 파일이 커밋되었는지 확인한다.
2. Claude Code를 열면 이 파일(`CLAUDE.md`)을 먼저 읽는다.
3. `process.md -> history.md -> conversation_l2_cache.md` 순서로 복원한다.
4. 이전 Codex 작업 내용을 `history.md`에서 확인하고 이어서 진행한다.
5. Codex 도구 상태나 MCP 복구 내용은 `history.md`, `process.md`, `.mcp.json`,
   그리고 필요한 경우 `C:\Users\jun99\.codex\config.toml` 변경 이력을 기준으로 확인한다.
