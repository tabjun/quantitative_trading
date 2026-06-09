# 대화 L2 캐시

목적: 전체 대화 원문이 아니라 최근 사용자 의도와 제약만 압축해서 남긴다.
영구 규칙은 `AGENTS.md`와 `skills.md`에 둔다.

## 유지 원칙

- 최근 핵심 요청 최대 20개만 유지한다.
- 오래된 항목 중 영구 규칙은 `AGENTS.md` 또는 `skills.md`로 승격한다.
- 작업 이력은 `history.md`, 현재 단계는 `process.md`에 맡긴다.
- 이 파일은 최근 사용자 의도와 선호를 복원하는 용도로만 쓴다.

## 최근 요청 캐시

| 날짜 | 요청 | 제약 | 조치/산출물 | 다음 단계 |
| :--- | :--- | :--- | :--- | :--- |
| 2026-06-08 | 실시간 텍스트 데이터를 독립변수로 쓰는 환경 구축 | 로컬 heavy run 금지. DuckDB/uv 사용. 텍스트는 다변량 컨텍스트로 결합. | `contexts/text_context.py`, `pipelines/ingest_text_context.py`, `text_features_15m`, `text_risk_guard` 추가. | 서버 환경에서 텍스트 반영 분석 리포트를 생성한다. |
| 2026-06-08 | 전체 대화 원문이 아닌 L2 캐시형 요청 로그 추가 | 원문 대화 저장 금지. 루트 프로젝트 파일 선호. | `conversation_l2_cache.md` 생성. | 의미 있는 방향 전환마다 압축 항목을 추가한다. |
| 2026-06-08 | 실행 주체 분리 규칙 명확화 | 자동화 코드는 보존. Codex 로컬 heavy run 금지. 경량 검증 허용. | 실행 경계를 문서에 반영. | 영구 규칙은 AGENTS/skills에 유지한다. |
| 2026-06-08 | 한국어 메일 본문 깨짐 수정 및 재발송 | 분석 실행 금지. 메일 인코딩만 수정. | UTF-8 기반 메일 발송 방식으로 교정. | 한글 메일은 UTF-8/MIME-safe 방식 유지. |
| 2026-06-08 | 주식/코인 예측용 독립변수를 논문 기반으로 조사 | `stock` 브랜치만 사용. 로컬 학습/백테스트 금지. | 독립변수 문헌 리뷰 보고서 작성 및 메일 발송. | feature mart 구현 시 변수 스키마로 반영한다. |
| 2026-06-08 | 과거 유사 흐름 데이터마트를 KRW 전체 종목 기준으로 구축 | BTC 단일 종목 금지. shape + factor + context 유사도. 로컬 full-market build 금지. | `marts/historical_flow.py`와 관련 pipeline/리서치 문서 추가. | 서버 환경에서 full mart build 후 query를 분석에 결합한다. |
| 2026-06-08 | 루트 모듈을 역할군 패키지로 재정리 | `main.py`는 루트 유지. `/test`는 연구/실험 전용. | `analysis/`, `advisors/`, `integrations/`, `contexts/`, `marts/`, `pipelines/` 구조로 정리. | `/test` 코드가 `contexts.*`, `marts.*`를 사용하도록 유지한다. |
| 2026-06-09 | Codex 호환 Meta Harness를 저장소 로컬에 설치 | 프로젝트 로컬 설치만 수행. 도메인별 하네스 구성은 보류. | repo-local harness 설치 후 메타데이터 확인. | 반복 업무가 명확할 때만 `$harness` 확장. |
| 2026-06-09 | Meta Harness를 전역 Codex 스킬로도 설치 | user scope 설치. 플러그인 패키징은 하지 않음. | 전역 harness 설치 후 메타데이터 확인. | 다른 프로젝트에서도 `$harness` 사용 가능. |
| 2026-06-09 | 4번 텍스트 독립변수 분석 코드를 추가 | 로컬 장시간 학습/백테스트 금지. DA/MASE와 KRW 원스케일 포함. | `test/models/4_text_independent_variable_analysis.py` 추가. | 서버 환경에서 실제 리포트를 생성한다. |
| 2026-06-09 | 4번 스크립트 모델 범위를 5개 대표 계열로 축소 | 로컬 heavy model run 금지. 테스트는 작게 유지. | LSTM/Transformer/Mamba-like/TimeXer-like/iTransformer-like로 축소. | 서버에서 소형 점검 후 확장. |
| 2026-06-09 | 4번 스크립트 OOM/Optuna 안전장치 추가 | raw close shortcut 방지. batch 안전장치와 clipping 필요. | log-return target 유지, adaptive batch, `grad_clip_norm`, `--optuna-trials` 추가. | 승인 환경에서 점진 확장. |
| 2026-06-09 | `.ipynb`와 동명 `.py` 미러를 hook으로 강제 | 노트북 실행 금지. Git diff 추적 필요. | pre-commit과 미러 동기화 helper 반영. | 노트북 커밋마다 `.py` diff도 함께 리뷰. |
| 2026-06-09 | hook 유틸리티와 4번 실험 구조를 다시 바로잡기 | `.ipynb` + `.py` 미러 복원. `test/scripts` 난립 금지. | helper를 `.githooks/`로 이동하고 4번 노트북/미러 재정렬. | 연구 실험은 항상 노트북 원본 우선. |
| 2026-06-09 | initial commit~2026-05-28 흐름을 기준으로 보호 규칙 복원 | disposable `test/scripts` 금지. hook으로 자동 강제. | `AGENTS.md`, `test/README.md`, `.githooks/check_repo_policy.py` 반영. | 이후 추가는 기존 유틸 재사용 또는 notebook+mirror 구조 준수. |
| 2026-06-09 | 메일 임시 파일과 분산된 문서 묶음 정리 | 연구 실행 금지. 운영 프레임워크와 `/test` 경계 유지. | 임시 메일 산출물과 중복 문서 제거, stage 비움. | 루트 문서 중심으로 작업 이어간다. |
| 2026-06-09 | 문서와 캐시를 역할별로 재정리 | 영구 규칙은 캐시에 두지 않음. 컨텍스트 무한 증가 방지. | `AGENTS.md`, `skills.md`, `README.md` 재배치. L2 캐시는 최근 요청 중심으로 축소. | 다음 세션부터 루트 문서와 짧은 캐시만 읽고 작업을 이어간다. |
| 2026-06-09 | L2 캐시 보존 개수를 20개 수준으로 확대 | 최근 한 세션 요청량을 반영. 여전히 무한 증식 금지. | 유지 원칙을 “최근 핵심 요청 최대 20개”로 수정. | 오래된 항목은 영구 규칙 승격 또는 history/process로 이관한다. |
| 2026-06-09 | `test/scripts` 정리와 메일/노트북 유틸 중복 제거 | `test`의 다른 연구 코드/문서는 웬만하면 유지. 스크립트만 실사용 기준으로 정리. | 메일 모듈 통합, obsolete notebook builder/report 유틸 제거, 이미지 추출 모듈 단일화. | 남은 스크립트만 기준으로 `/test` 보조 유틸을 유지한다. |
| 2026-06-09 | `test/data`가 빈 이유 확인 및 역할 점검 | DB는 로컬 보관 가능하되 Git 추적과 코드 기본 경로 구분 필요. | `test/data`는 로컬 데이터 보관 폴더이고, 실제 DB는 root `.gitignore`의 `*.db*` 규칙 때문에 비어 있을 수 있음을 문서화. | 필요 시 로컬 복사본을 두되 코드 기본 DB 경로와 사용 목적을 명확히 유지한다. |

