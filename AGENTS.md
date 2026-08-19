# AGENTS.md - Codex/Claude Code 공용 작업 지침서

## 목적

이 문서는 이 저장소에서 Codex와 Claude Code가 매 세션 반드시 따라야 하는 강제 규칙을
정의하는 **단일 source of truth**다. `CLAUDE.md`는 별도 규칙을 담지 않고 이 파일을 가리키는
참조 스텁이다(2026-08-09부터 — 이전에는 두 파일에 동일 규칙을 중복 관리하며 섹션 구조까지
맞췄으나, 드리프트 위험을 없애기 위해 이 파일 하나로 합쳤다). 규칙을 고칠 때는 이 문서만
고치면 되고, `CLAUDE.md`는 건드릴 필요가 없다(Claude Code 전용 예외 2가지만 그 파일에 남음).

도구별 사설 memory는 보조 정보일 뿐이다. source of truth는 저장소 문서다:
`AGENTS.md`(영구 규칙, Claude Code는 `CLAUDE.md` 스텁을 거쳐 진입) → `process.md`(현재 단계) →
`state.md`(요청 단위 완료 체크) → `history.md`(이력) → `conversation_l2_cache.md`(최근 요청
원문) → `test/known_pitfalls.md`(실행 직전 체크) → `test/README.md`(연구 공간 안내).

### 문서별 역할 (겹치지 않게 유지, 2026-08-09 확정)

각 문서는 아래 역할 하나만 맡는다. 새 내용을 적을 때 이 표로 어느 파일에 쓸지 먼저 판단한다.

| 문서 | 역할 | 갱신 주체 | 형식 |
| :--- | :--- | :--- | :--- |
| `README.md` | 프로젝트 전체 설명(무엇인지, 왜 하는지) — 거의 안 바뀜 | 사용자+AI | 서술 |
| `AGENTS.md`/`CLAUDE.md` | 항상 적용되는 영구 규칙(source of truth) | 사용자 승인 후 AI | 서술 |
| `process.md` | 연구 단계별 To-Do(무엇을 할 것인가, Phase/실험 단위) | AI | `[x]`/`[ ]` + 서술 |
| `state.md` | **사용자 요청 단위로 "실행됐는가"만** 체크(연구 서술 없음) | AI(요청마다 자동 갱신) | `[x]`/`[ ]`/`[~]` 표 |
| `history.md` | 완료된 작업의 이력(무엇을·왜·어떻게·결과) | AI | 표 |
| `conversation_l2_cache.md` | 사용자 요청 **원문** 보존(최근 20개, 요약 안 함) | AI | 표 |

`process.md`와 `state.md`를 혼동하지 않는다 — `process.md`는 "이 연구를 어떤 순서로 할
것인가"이고, `state.md`는 "사용자가 요청한 각각의 일이 끝났는가"다. 전자는 실험이
바뀔 때만 갱신되고, 후자는 매 요청마다 갱신된다.

이 문서는 **압축 유지가 원칙**이다. 새 사건이 생겨도 문단을 계속 붙이지 않는다 — 기계적으로
판정 가능한 규칙은 `test/known_pitfalls.md`(코드 게이트)로, 한 시점 상태는 `process.md`로,
사건 이력은 `history.md`로, 배경 설명이 긴 것은 `test/results/*.md` 참조 문서로 보낸다.
여기 남기는 건 "항상 적용되는 짧은 규칙"뿐이다. append 전에 기존 항목과 중복인지 먼저 확인하고,
중복이면 새로 쓰지 않고 기존 항목의 날짜만 갱신한다.

---

## 0. 항상 먼저 읽을 문서

새 세션이 시작되면: `AGENTS.md`(이 파일, Claude Code는 `CLAUDE.md` 스텁을 거쳐 여기로 옴) →
`process.md` → `state.md`(미완료 요청 확인) → `history.md` 최근 섹션 →
`conversation_l2_cache.md` 최근 항목 → `test/known_pitfalls.md` → `test/README.md`.

---

## 1. Claude ↔ Codex 병행 사용

- 규칙 파일은 이 문서 하나(`AGENTS.md`)뿐이다. Claude Code는 자동으로 읽는 `CLAUDE.md`가
  이 파일을 가리키는 3줄 스텁이라 결과적으로 같은 규칙을 본다. MCP 설정 파일만 도구별로
  분리되어 있다: Codex=`.codex/config.toml`, Claude Code=`.mcp.json`. 어느 도구로 시작해도
  같은 규칙을 이어받는다.
- **요청을 받을 때마다**: `conversation_l2_cache.md`에 새 인덱스로 요청 원문을 먼저 기록하고,
  같은 인덱스로 `state.md`에 `[ ]`(진행중) 행을 추가한다. 요청을 완료하면 `state.md`의 그
  행을 `[x]`로 갱신한다(중단/보류는 `[~]`).
- **세션 종료 전**: `history.md`에 수행 내용·산출물·다음 목표 한 행 추가, `process.md` 현재
  단계 갱신, `state.md`에 미완료로 남은 항목이 있으면 `[~]`와 비고로 정확히 표시. 방향
  전환이 있었으면 `conversation_l2_cache.md`에 요청 원문 그대로 추가(요약 금지). 중단/
  복구 중이면 완료처럼 쓰지 말고 확보한 것·못한 것·다음 도구가 할 일을 명시한다.
- **세션 시작 시**: `process.md` 미완료 체크리스트 → `state.md`의 `[ ]`/`[~]` 행(무엇이
  안 끝났는지) → `history.md` 최근 완료 작업 → `conversation_l2_cache.md` 최근 선호·제약
  순으로 복원한다.

---

## 2. 영구 규칙

### 2.1 로컬 heavy run 금지, 경량 검증 허용

- Codex는 개인 연구 세션에서 로컬 터미널/`.venv`로 장시간 분석·학습·백테스트·노트북 결과
  산출을 실행하지 않는다. 허용: 문법 검사, import 확인, 작은 synthetic 테스트, 정적/diff
  리뷰, 짧은 컴파일 점검. 실제 연구 실행은 학교 서버 커널, CI, 스케줄러, 또는 사용자가 명시
  승인한 원격 환경에서 한다.
- **현재 환경 단서**: 세션 자체가 학교 서버(GPU RTX 4090 24GB) 위에서 도는 경우가 있다. 핵심은
  "환경 이름"이 아니라 "GPU heavy run은 커널/명시 승인으로만, 세션은 경량 검증까지만"이다.
  서버 세션에서도 경량 체크는 허용되고, heavy run은 여전히 노트북 커널/명시 승인으로만 한다.
- `history.md`/`conversation_l2_cache.md`의 수행환경 컬럼: `로컬`(코드·문서·경량검증·커밋·메일)
  / `서버`(GPU heavy 실행) / 섞이면 `로컬+서버`.

### 2.2 `/test`와 실사용 프레임워크를 분리한다

- 루트 `quantitative_trading/`는 실사용 프레임워크·운영 문서. `test/`는 연구 실험·노트북·
  리서치 문서·결과물 전용. `/test` 요청은 먼저 연구 목적·artifact 경계를 평가하고, 목적에서
  벗어나면 그대로 수행하지 말고 비판적으로 재구성한다. 운영 진입점은 `pipelines/`, 새 워크플로우
  플러밍은 `.githooks/` 같은 인프라 폴더에 둔다.

### 2.3 연구 분석은 `.py` 헤드리스 드라이버를 기본으로 한다 (2026-07-16 갱신)

- 세션이 원격 서버(GPU)에서 직접 도는 현재 환경에서는 `.ipynb` 커널 실행 대신 **헤드리스 `.py`
  드라이버**를 기본으로 한다(`#%%` 셀 구분 + 파일 내 마크다운 주석). 실행과 결과 저장(raw md +
  csv + png, 실험 태그 디렉터리)을 `.py`가 전담한다.
- **`.ipynb` 미러는 요구하지 않는다 (2026-07-22 규칙 폐지).** `.py` 단독으로 충분하며, 설명은
  `.py` 안의 `# %% [markdown]` 셀로 담는다. 새 실험마다 ipynb를 만드는 토큰 낭비를 없앤다.
  pre-commit 훅의 "동명 `.ipynb` 없는 `test/models/*.py` 차단" 규칙도 함께 제거했다.
- 기존 번호 실험의 의미가 달라지는 후속 연구는 같은 파일을 재목적화하지 않고 새 번호로 분리한다
  (실험 번호는 불변 식별자 — `test/known_pitfalls.md` P6).
- 완료된 실험(코드+결과)은 read-only source of truth로 다룬다. 재정리는 `test/results/*.md`,
  `test/images/*`, 메일 preset 같은 후속 산출물에서만 한다. 예외는 사용자의 명시 승인뿐이다.
- (2026-05 이전 순수 노트북 시절 유산) `.ipynb` 원본 + `.py` 미러를 직접 실행하던 실험들은
  그 구조 그대로 read-only로 남긴다 — 소급 전환하지 않는다.

### 2.4 메일은 UTF-8 기준으로 다룬다

- 한국어 메일 본문은 UTF-8/MIME-safe로 보낸다(PowerShell inline here-string 조합 방식 피함).
  재사용 가능한 메일 도구만 `test/scripts/`에 둔다.
- 결과 해석을 좌우하는 용어(persistence, collapse, variance_ratio, direction accuracy, MASE,
  trend_corr 등)는 독자가 이전 문서를 안 읽었다고 가정하고 매 문서·메일에서 `쉬운 정의 → 예시 →
  이번 실험에서의 작용 → 좋은/나쁜 신호와 예외` 순으로 다시 푼다.
- "상대적으로 보존했다", "평평해졌다" 같은 비교 표현은 무엇과 비교했는지, 어느 방향으로
  변했는지, 그것만으로 우수하다고 결론 낼 수 있는지를 함께 적는다.
- 새 방법론·실험 축 추가 요청은 현재 연구 질문을 실제로 강화하는지 적대적으로 검토한 뒤에만 반영한다.

### 2.5 수행환경 표기 (2.1 참조)

`history.md`/`conversation_l2_cache.md` 이력 행 끝 `수행환경` 컬럼은 항상 채운다(로컬/서버/
로컬+서버). 2026-06-28 이전 기록은 이력 없어 일괄 `로컬`.

### 2.6 커밋 메시지 규칙

- 형식: `<type>(<scope>): <제목>` + 빈 줄 + `무엇을 왜` 본문 + 빈 줄 +
  `Co-Authored-By: <현재 모델명> <noreply@anthropic.com>`.
- header 필수, scope는 가능하면 붙인다(`serving`/`router`/`scheduler`/`io`/`collect`/
  `prototype`/`docs`/`research`/`vscode`/`infra`). 본문은 `어떻게`보다 `무엇을 왜`.
- 한 커밋에 성격(`feat`/`build`/`refactor`/`fix`/`chore`/`docs`/`test`/`style`/`perf`/`ci`)을
  섞지 않는다 — 여러 성격이면 커밋을 분리한다.

### 2.7 branch·Git 전달 규칙

- 사용자 지시 없으면 작업 브랜치에서만 커밋/푸시. `main`/`develop` 병합은 명시 요청 시만.
- **브랜치별 역할 분리 (2026-08-09 확정)**: `stock`은 연구 브랜치라 작업 지침 파일을 포함한
  전량을 커밋한다. `develop`/`main`은 최종 결과물만 올라가는 브랜치라 아래 지침 파일 6종은
  **존재 자체를 제외**한다 — `AGENTS.md`, `CLAUDE.md`, `conversation_l2_cache.md`, `history.md`,
  `process.md`, `state.md`. 일반 `git merge stock`을 그대로 실행해도 되며, 이 파일들이
  스테이징되면 `.githooks/check_repo_policy.py`(pre-commit)가 `develop`/`main`에서 커밋
  자체를 자동 차단하고 되돌리는 명령을 안내한다(삭제는 허용). 훅이 막았을 때 되돌리는 예:
  ```bash
  git restore --staged --worktree -- AGENTS.md CLAUDE.md conversation_l2_cache.md history.md process.md state.md
  # 해당 브랜치에 그 파일이 원래 없었다면(신규 추가로 잡힌 경우)
  git rm --cached -- <해당 파일>
  ```
  `test/`, `pipelines/`, `marts/`, `engine/`, `analysis/` 등 코드·결과물은 그대로 병합된다.
  이 자동 차단은 로컬 git이 `core.hooksPath=quantitative_trading/.githooks`를 가리킬 때만
  동작한다 — 새로 clone한 환경에서는 `git config core.hooksPath quantitative_trading/.githooks`를
  먼저 실행한다.
- **서버 ↔ GitHub은 SSH 원격**(`git@github.com:tabjun/personal_ai_project.git`)이 기본이다.
  학교 서버는 인바운드 SSH(22번)는 방화벽에 막혀 있지만 아웃바운드는 열려 있어 서버에서
  `ssh -T git@github.com` 인증은 정상 동작한다(2026-07-14 확인). **로컬 PC ↔ GitHub은 이
  규칙과 무관** — 로컬은 HTTPS 원격을 그대로 써도 된다(2026-08-09 확인, push 정상 동작).
  즉 "SSH가 기본"인 대상은 서버의 git 원격뿐이고, 로컬 origin 프로토콜까지 SSH로 맞출
  필요는 없다.
- VS Code Remote Tunnels의 GitHub 로그인(device code OAuth, `tools/USAGE.md` 방법 2)은
  이 git 원격 설정과 무관한 별개 채널이다 — SSH가 막혀 있어도 아웃바운드 방식이라 서버
  접속엔 쓸 수 있지만, git push/pull의 SSH/HTTPS 선택에는 영향을 주지 않는다.
- 보고서/메일 링크는 GitHub 렌더링 Markdown URL만(commit history 링크 금지). 메일 본문엔
  핵심 개선점 + 보고서 링크.
- **메일에 `.md` 원본 파일을 첨부하지 않는다 (2026-07-23 재확인).** raw `.md`는 받는 쪽에서
  텍스트로 열려 이미지·표가 안 보이고 가독성이 나쁘다(2026-05-28에 이미 겪고 고쳐 렌더링
  링크 방식으로 정착시켰는데, 2026-07-22 preset에서 재발). 그림이 필요하면 본문에 inline
  이미지(cid)로 임베드하거나 PNG만 첨부하고, `.md`는 항상 GitHub 렌더링 링크로만 전달한다.

### 2.8 새 파일보다 기존 파일을 우선한다

`AGENTS.md`(+ `CLAUDE.md` 스텁), `process.md`, `state.md`, `history.md`,
`conversation_l2_cache.md`, `test/known_pitfalls.md`, `test/README.md`, `pipelines/`,
`test/scripts/`, 기존 `test/models/*`를 먼저 확인하고, 새 파일보다 기존 파일 수정·확장을
우선한다.

### 2.9 `test/scripts/`에는 재사용 도구만

허용: 노트북 빌더, 보고서 변환기, 이미지 추출기, 환경 복구기, 메일/리포트 전달기, MCP 호출기.
금지: 이번 한 번만 쓰는 ad-hoc 스크립트.

`check_repo_policy.py`는 기본적으로 `test/scripts/`의 **신규 `.py`를 전부 차단**한다(ad-hoc
유입 방지). 재사용 도구라서 예외가 필요하면 그 훅의 `TEST_SCRIPTS_ALLOWED`에 파일명을
명시적으로 올린다 — 예외는 자동이 아니라 매번 의도적으로 선언한다(2026-08-10 신설).

### 2.9b 전처리 계보를 보고서에 반드시 명시한다 (2026-08-14 신설)

**배경**: 차분(1차 로그차분)이 `engine/data.py:121`·`:133`에 내장돼 모든 실험이 통과하는데,
어느 보고서도 "우리 예측 대상은 차분값이다"를 명시하지 않아 사용자가 비정상성 모델링을 하는 줄
알고 있었다. 17번 S4가 레벨 vs 차분을 비교했으나 그것은 *데이터 성질 진단*이었고 *우리 선택의
선언*이 아니어서 1~16번과 교수님 브리프에 전파되지 않았다.

**규칙**: 모델링·평가가 들어간 보고서는 아래 계보를 표로 먼저 밝힌다.
`원본 레벨 → 변환(로그 등) → 차분 차수 d → 타깃 정의 → 스케일링/정규화 → 파생변수`.
근거·실측은 `test/results/18_differencing_decision_report_20260814.md`(D0 절) 참조.

- **로그 변환은 차분이 아니다** — 단위 변경일 뿐이고 로그 레벨도 비정상이다.
- **평가축을 명시한다**: 레벨축 R²는 무모델도 0.9999를 받고 실모델과 구별되지 않는다
  (차이 0.0000001). 성적표에는 척도-자유 지표(MASE·QLIKE)와 naive 기준선을 함께 놓는다.
- **파라미터로 모델을 판정하지 않는다**: GARCH 지속성·반감기는 이 데이터에서 표본에 따라
  55배 요동해 식별되지 않는다. 판정은 **target horizon의 표본외 성능**으로 한다.

### 2.9d 성능 지표는 축과 함께 보고한다 (2026-08-19 신설)

**배경**: "R²가 유효하지 않다"고만 알려져 있었는데, 실측하니 문제는 지표가 아니라 축이었다.
레벨축에서는 11개 지표 전부가 무모델과 실모델을 구별하지 못하고(분리도 0.0~0.2%), 차분축에서는
R²가 가장 민감한 분리자(112%)다. 근거: `test/results/18f_metric_validity_20260819/`.

**규칙**: 모델 성능을 보고할 때 아래를 **함께** 낸다. 지표 하나만 쓰면 판별이 불가능하다.

```
차분축 필수:  R² · MAE · MASE · DA · DA(큰변동) · variance_ratio
변동성 실험:  QLIKE · MAE · MASE · corr(√예측, |r|)
항상 병기:    무모델(naive) 값과 **어느 축에서 쟀는지**
```

- **읽기 쉬운 지표와 판별하는 지표를 구분한다**: MAE·MAPE는 크기 감각(직관), 차분축 R²·
  variance_ratio는 판별, MASE·copy_risk는 무모델 대비. MASE는 축 불변이지만 분리도는 낮다.
- **레벨축 성적은 보고하지 않는다**(원단위 감각이 필요하면 naive 대비 비율로만).

### 2.9e 데이터 준비는 코드가 확인한다 (2026-08-19 신설)

DB는 `.gitignore`의 `*.db*`로 제외돼 `git pull`로 오지 않는다. 실험 드라이버는 시작 시
`engine/preflight.py`의 `ensure_data()`를 호출해 DB·테이블·종목·행수를 점검하고, 부족하면
수집하고, 수집 후에도 부족하면 **예외로 중단**한다(조용한 실패 금지).

```python
from engine import preflight
preflight.ensure_data(table="upbit_krw_candle", tickers=["KRW-BTC"])   # 필요한 종목만
preflight.ensure_data(table="upbit_krw_candle", tickers="all")          # 전 종목
```

장시간 수집이 곤란한 세션은 `QT_NO_AUTOBUILD=1`로 점검만 한다.

### 2.9c 다종목 테이블은 ticker 없이 읽지 않는다 (2026-08-14 신설)

`upbit_krw_candle`처럼 `ticker` 컬럼이 있는 테이블을 ticker 지정 없이 읽으면 269종목이
timestamp 순으로 섞여 나오고, `make_features`의 `log_close.diff()`가 **서로 다른 코인 사이의
차분**을 계산한다(KRW-ELF 366원 → KRW-HBAR 66원 = "−171% 수익률"). 에러가 나지 않아 조용히
쓰레기가 만들어진다. `engine/data.py:load_price_data`가 이제 예외로 차단하며, 집계 목적이면
`allow_mixed_tickers=True`를 명시하고 호출자가 종목별 groupby를 책임진다.

실험 드라이버 기본값은 `--table upbit_krw_candle --ticker KRW-BTC`다(2026-08-14 교정).
`btc_15m_advance`는 4번부터 굳어진 편의 기본값이었고 ticker 컬럼이 없어 이 함정이 드러나지 않았다.

### 2.10 가상환경은 재사용한다

서버의 기존 uv venv 두 개를 재사용한다(매번 새로 만들지 않음):
`.venvs/quant_uv_py312_...`(Python 3.12, 기본 정합성) / `.venvs/quant_uv_py313_...`(3.13).
단일 실험은 3.12로 통일. `11_` vs `12_`처럼 상반 실험을 **동시 비교**할 때만 3.12+3.13 병렬.
`bootstrap_*.sh`는 env가 깨졌을 때 재구축 용도로만(평소엔 `source .venvs/<env>/bin/activate`).

### 2.11 코드 작성 루프 규율 (Loop Engineering Field Notes)

> 출처: Karpathy, "Field Notes on Getting a Language Model to Write Code You Will Not
> Rewrite". 원문은 `personal_ai_project/CLAUDE.original.md`.

Read Before You Write(정독, 패턴 모르면 질문) · Think Before You Code(가정·트레이드오프 명시,
헷갈리면 멈춰서 질문) · Simplicity(눈앞 문제만, 성급한 추상화 금지) · Surgical Changes(diff
최소, 안 시킨 곳 금지) · Verification(버그는 실패 테스트로 재현 후 수정) · Goal-Driven
Execution(코드 전 성공기준) · Debugging(추측 금지, 재현 후 한 번에 하나씩) · Dependencies(표준
라이브러리 우선, 추가 시 이유 명시) · Communication(우려·불확실성 구체적으로) · Common Failure
Modes(Kitchen Sink/Wrong Abstraction/Optimistic Path/Runaway Refactor 경계).

### 2.12 연구 방향 최상위 원칙과 세션 운영 (2026-07-18~19 확정)

> **실행 코드 작성 직전엔 `test/known_pitfalls.md`(코드 게이트 20줄)만 대조.** 여기 아래는
> 코드로 못 박을 수 없는 원칙만 남긴다. 배경·근거가 긴 것은 참조 문서로 보낸다(각 항목 끝 링크).

- **[최상위] 연구 목적이 모든 개별 지시보다 우선한다.** 목적은 3요소 결합: ① 학습 건전성
  (최적화·손실이 쉬운 해로 붕괴 안 함, 과적합 관리, 비정상성 가정 충족) ② 전체 변동 폭 추세
  예측(작은 폭뿐 아니라 큰 변동까지 — 정확도 단독이 아니라 정밀도·재현율 관점) ③ MDD는
  생존 제약(하방선 무너지면 전체가 무너짐, 목적 자체를 대체하지 않음).
- **국소 회귀 금지 + 매 요청 방향 일치성 판단**: 사용자가 하나를 짚어도 그것'만'을 새
  최우선으로 갈아끼우지 않는다. 요청 수행 전 "연구 목적과 일치하는가"를 판단하고 근거를
  보고에 남긴다(어긋나면 재구성/확인, 2.2와 동일 정신).
- **보고 방식**: 중간엔 진척만("~단계 완료 → 다음"), 상세(무엇을·왜·어떻게·결과)는 요청 단위
  종료 시 1회. 실행 중 발견한 버그·이슈·수정은 결과 raw md에 전량 남긴다(취사선택은 사용자 몫).
- **suite 승계·챔피언 선정**: 단일 지표 최고로 뽑지 않는다. 진폭 건전성(variance_ratio)과 큰
  변동 포착(tail_f1·large_move_da)을 함께 본다 — `known_pitfalls.md` P1/P4.
- **요청 원문 보존(원문=대화캐시, 요약=process/history)**: `conversation_l2_cache.md`는 사용자
  요청을 **원문 그대로** 남기는 곳이다 — 이 파일에서는 요약하지 않는다. 요약·정리·구조화는
  `process.md`(현재 상태)와 `history.md`(이력)가 맡는다.
- **데이터 축 기본값**: 업비트 KRW 전 종목 15분봉(원설계). 단일 종목은 진단 목적에 한해 쓰고
  결과는 "단일 축 한정"으로만 해석한다 — P3.
- 배경·전체 사례: `test/results/governance_drift_rootcause_20260719.md`(왜 지침이 실행에
  안 붙는지의 근본원인·해소책), `test/results/15_research_trajectory_audit_20260718.md`
  (MDD 단독 프레임 폐기 근거).

---

## 3. 실행 주체 분리

**저장소에 남길 것**: 다른 연구원/운영자가 실행할 자동화 스크립트, `uv run ...` 재현 명령,
n8n/Cron/CI/Docker/Kubernetes 설계, 학교 서버 커널 실행 절차·파라미터.

**Codex가 이 세션에서 할 수 있는 것**: 코드 작성/수정, 정적·diff·설계 리뷰, 문법/import/
작은 synthetic 테스트, 논문 조사, 연구용 `.py` 드라이버 작성과 `.ipynb` 미러 동기화.

**하지 않는 것**: `uv run main.py` 류 장시간 로컬 실행, `.venv`/로컬 Python으로 대용량 연구
수행, 결과 수치 생성을 목적으로 한 대규모 DB/시계열 분석 실행.

---

## 4. 분석 설계 원칙

1. **정상성 처리**: 원시 가격 수준만 그대로 학습시키지 않는다. 정상성 검정(ADF/KPSS), 롤링
   드리프트 점검, log return/diff/rolling z-score/RevIN 계열, KRW 역복원 지표를 함께 쓴다.
   모든 데이터를 하나의 정상 표현으로 강제하지 않는다.
2. **보고서 기준**: 실행·분석 환경, 기초 통계량, 방법론/지표/손실함수/진단 도구 개념, KRW
   원본 스케일 성능 지표, DA·MASE, 시각화 해석 포함. 그래프마다 `데이터/모델 → x/y축 → 진단
   목적 → 관찰 → 좋음/나쁨 → 다음 반영`을 설명한다. 새 보고서는 이전 설명이 있어도 독립 문서.
3. **모델 철학**: "Shallow but Wide" — 레이어 1~2층, width 64~128 우선 검토.

---

## 5. MCP 도구 설정

| 서버 | 등록 위치 | 용도 | 인증 |
| :--- | :--- | :--- | :--- |
| `arxiv` (stdio) | `.mcp.json`(Claude Code) + `.codex/config.toml`(Codex) | 논문 검색·초록·BibTeX·인용그래프 | 불필요 |
| `huggingface` (http) | `.mcp.json` | 모델·데이터셋·Space 검색(`hub_repo_search` 등) | 익명 가능, `HF_TOKEN` 있으면 rate limit 완화 |

**논문 검색은 arxiv MCP를 쓴다.** 인용은 `search_papers` → `get_abstract`로 **초록까지 대조**한
것만 쓰고, 대조 못 한 인용은 보고서에 "미검증 인용"으로 명시한다. HF MCP에는 논문 검색 툴이
없다(모델·데이터셋 전용) — 문헌 근거를 HF로 대체하지 않는다.

arxiv 서버 미설치 시:
`uv tool install --managed-python --python 3.12 git+https://github.com/blazickjp/arxiv-mcp-server.git`

**세션 밖에서 호출하기.** MCP 서버는 에이전트 세션 **시작 시점**에만 로드되므로, 새로 등록한
서버는 같은 세션에서 툴로 노출되지 않는다. 이때는 웹검색으로 우회하지 말고
`test/scripts/mcp_client.py`를 경유해 같은 서버를 그대로 호출한다.

```bash
python test/scripts/mcp_client.py arxiv tools
python test/scripts/mcp_client.py arxiv search_papers '{"query": "...", "max_results": 6}'
python test/scripts/mcp_client.py arxiv get_abstract '{"paper_id": "2109.12142v2"}'
python test/scripts/mcp_client.py huggingface hub_repo_search '{"query": "garch", "type": "model"}'
```

**`.mcp.json`은 지침 파일이 아니다** — 2.7절 `GOVERNANCE_FILES`에 넣지 않는다. `mcp_client.py`
(연구 재현 스크립트, develop/main에도 올라간다)가 이 파일을 읽어 서버 정의를 얻으므로,
develop/main에서 제외하면 그 브랜치의 재현 명령이 깨진다.

---

## 6. 재현 실행 명령

```bash
uv run main.py
uv run pipelines/ingest_text_context.py
uv run pipelines/build_historical_flow_mart.py
uv run pipelines/query_historical_flows.py
uv run pipelines/rebuild_price_mart.py
uv run pipelines/simulate_and_send.py
```

---

## 7. 도구 전환 체크리스트

두 도구가 같은 `AGENTS.md`를 보므로(Claude Code는 `CLAUDE.md` 스텁을 거침) 규칙 드리프트
걱정 없이 전환 가능하다. 전환 시 확인할 것은 다음 두 가지뿐이다.

**Codex → Claude / Claude → Codex 공통**: 변경 파일 커밋/메모 → `history.md` 기록 →
`process.md` 다음 스텝 갱신 → `state.md`에 미완료 항목 정확히 표시. 새 세션은
`AGENTS.md`(또는 `CLAUDE.md` 스텁) → `process.md` → `state.md` → `history.md` →
`conversation_l2_cache.md` 순으로 복원한다. 어느 쪽 memory에만 남긴 결정도 반드시 저장소
파일에 남긴다.
