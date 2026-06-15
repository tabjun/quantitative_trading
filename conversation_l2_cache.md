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
| 2026-06-13 | 5번 최적화 진단의 아키텍처 범위 복구 | 대표 알고리즘 5개 비교 의도 유지. 노트북 원본과 `.py` 미러 동기화 필요. | `architecture_probe`와 `full_matrix`를 `Linear`, `LSTM`, `GRU`, `TCN`, `Transformer`로 복구하고 해설 문구도 함께 갱신했다. | 출력 셀은 승인된 환경에서 다시 실행할 때만 최신화한다. |
| 2026-06-13 | 5번 진단 범위 전체 복구 | LSTM 중심 서술이 다시 관성으로 돌아갈 위험 제거. 출력 셀은 실행 전 상태로 비움. | `quick_probe`와 `objective_probe`까지 5개 대표군으로 확장하고, 초록 문구도 5개 대표군 비교로 통일했다. | 다음 실행은 승인된 환경에서만 새 결과를 생성한다. |
| 2026-06-13 | quick_probe 보고서 정리 | 현재 저장본은 LSTM quick_probe 요약이고, 최신 5개 대표군 결과는 별도 재생성이 필요. | `test/results/5_optimization_diagnostics_quick_probe_20260613.md`를 현재 저장본 기준으로 정리했다. | 최신 결과가 저장되면 보고서를 다시 덮어쓴다. |
| 2026-06-13 | quick_probe 보고서 서식 전환 | 출력이 비어 있어 결과 수치 대신 5개 대표군 기준 틀이 필요했다. | `test/results/5_optimization_diagnostics_quick_probe_20260613.md`를 전체 비교용 보고서 골격으로 다시 작성했다. | 다음 결과가 생기면 같은 서식에 실제 수치를 채운다. |
| 2026-06-13 | 보고서 source of truth 고정 | 보고서 작성과 해석은 사용자가 실제로 실행한 현재 로컬 파일 출력이 기준이어야 함. `git`의 과거 상태나 HEAD 출력으로 대체하면 안 됨. | 이후 보고서 작업은 현재 워크트리의 `.ipynb` 출력 또는 그 로컬 산출물을 우선 source of truth로 사용한다. | 다음부터 결과 해석은 무조건 로컬 실행 파일 기준으로 이어간다. |
| 2026-06-13 | quick_probe 로컬 출력 재반영 | 5번 보고서는 현재 로컬 노트북 출력과 이미지 추출 흐름으로 복구해야 한다. | 현재 `test/models/5_optimization_diagnostics_test.ipynb`의 Markdown 출력으로 `test/results/5_optimization_diagnostics_quick_probe_20260613.md`를 다시 썼고, `test/scripts/extract_notebook_images.py`로 PNG 34장을 추출했다. | 이후 보고서 수정은 로컬 `.ipynb` 출력과 `test/images` 번들을 함께 기준으로 한다. |
| 2026-06-13 | 연구 요청 추적성 강화 | 최근 20개 캐시만으로는 연구 시작부터의 문제 제기, 수정 요청, 해석 방식 변화가 충분히 안 보일 수 있음. 논문으로 이어질 수 있게 전체 흐름은 커밋과 이력 문서에서 복원 가능해야 함. | 앞으로 `conversation_l2_cache.md`는 최근 의도만 압축 유지하고, 중요한 방향 전환/방법론 수정/보고서 해석 원칙 변화는 `history.md`와 의미 있는 커밋 메시지에도 남긴다. | 캐시는 얇게 유지하되, 연구 흐름은 commit diff + history log만 보면 복원 가능하도록 문서/커밋 품질을 맞춘다. |
| 2026-06-13 | 6번 최적화 안정화 코드를 더 구체화 | 오래 걸려도 되지만 OOM은 피해야 함. GPU 학습은 서버/승인 환경에서만. 5번 수정 이유도 먼저 설명 필요. | 5번에 normalization/loss ablation 축을 추가하고, 6번에 stage별 서버 실행, `server_2048` 프로필, dry-run, OOM batch retry, 문헌 기반 계획서를 반영했다. | 서버에서 6번 전체 실행 후 CSV/이미지/노트북 출력 기준으로 원인-개선 보고서를 작성한다. |
| 2026-06-13 | 최근 시계열·금융 예측 논문을 5개 이상 구체적으로 조사하고 교수님 공유용 보고서/메일 작성 | 코드 생성보다 리서치와 보고서가 목적. 3번 Autoformer 착시, 4/5/6번 목적, 최적화 안정화가 논문화에 왜 필요한지 설명해야 함. | 최근 알고리즘과 금융·코인 예측 논문의 학습/평가 흐름을 `forecasting_methodology_literature_review_20260613.md`에 정리하고, `test/README.md`와 메일 preset에 연결했다. | 커밋/푸시 후 렌더링 링크를 교수님께 발송하고 6번 서버 실행 결과로 다음 보고서를 작성한다. |
| 2026-06-13 | 학교 서버에서 `py`가 없고 `python3.10`만 보이는 환경 재구성 | 외부 SSH 불가. `uv init`이 홈 디렉터리에서 별도 프로젝트를 만들었고, Jupyter kernel 등록 절차가 필요함. | `test/README.md`에 `uv venv --python 3.12`, `uv sync`, `ipykernel install`, `Kernel -> Shut Down All Kernels` 흐름을 추가했고, `AGENTS.md`에는 notebook-safe argparse 규칙을 넣었다. | 서버에서 실제로 쓸 Python 버전을 확정하고, 그 env로 kernel을 다시 등록한다. |
| 2026-06-13 | 서버 환경을 3.12로 고정하고 cu126 PyTorch를 쓰는 쪽으로 정리 | `nvidia-smi`에서 CUDA 12.6 / RTX 4090이 확인됐고, `uv`가 3.13을 먼저 잡는 문제가 있었다. | `.python-version=3.12`를 추가하고 `test/README.md`의 복붙 명령을 `uv venv --python 3.12`, `UV_CACHE_DIR`, `torch==2.10.0+cu126`, `quant312` kernel 등록 순서로 다시 정리했다. | 서버에서 3.12 커널이 실제로 보이는지만 확인하면 된다. |
| 2026-06-13 | 서버용 환경 명령을 `uv`와 일반 `venv`로 분기 | 새 터미널에서 바로 따라칠 수 있는 3.12 기준 명령 세트가 필요했다. | `test/README.md`에 `uv` 경로와 `python3.12`가 실제로 있을 때만 쓰는 일반 `venv` 경로를 분리해 적었다. | 서버에서 어떤 Python 경로가 가능한지 확인한 뒤 그 경로만 사용한다. |
| 2026-06-13 | 서버 환경 bootstrap 스크립트 4종 추가 | 환경 구축, CUDA 체크, `uv sync`, 커널 등록을 매번 손으로 치지 않도록 자동화가 필요했다. | `test/scripts/bootstrap_uv_313.sh`, `bootstrap_uv_312.sh`, `bootstrap_venv_313.sh`, `bootstrap_venv_312.sh`를 추가하고, `test/README.md`는 스크립트 호출 중심으로 다시 줄였다. | 다음부터는 서버에서 필요한 버전의 스크립트 하나만 실행해 환경을 복구한다. |
| 2026-06-13 | 서버 환경 이름 충돌 방지와 정리 안내 추가 | 고정 `.venv`를 덮어쓰면 기존 환경과 새 환경이 섞여 혼란스러울 수 있었다. | 공통 bootstrap 로직을 `bootstrap_env_common.sh`로 빼고, 새 환경을 `.venvs/<env_name>`에 timestamp 기반으로 만들며, 기존 env/kernels 목록과 제거 명령도 출력하도록 바꿨다. | 다음부터는 서버에서 어떤 env가 생겼는지와 어떻게 지울지를 스크립트 출력만 보고 판단한다. |
| 2026-06-13 | `uv sync` 후 `ipykernel` 누락 문제 보정 | 서버에서 `uv sync`는 끝났지만 `python -m ipykernel install ...`에서 `No module named ipykernel`이 발생했다. | `pyproject.toml` 기본 dependencies에 `ipykernel`을 추가했다. bootstrap 스크립트는 이미 별도 설치를 하고 있어 수정하지 않았다. | 기존 서버 env는 `uv sync`를 한 번 더 실행해 `ipykernel`을 받아오고, 그 뒤 커널 등록을 다시 시도한다. |
| 2026-06-13 | 6번은 결과 보고서로 마감하고 7번은 새 번호 확장 실험으로 분리 | 5/6의 의미를 바꾸지 않고 보존해야 함. 새 보고서는 이전 설명이 있더라도 다시 적는 독립 문서여야 함. | `6_optimization_stabilization_stage_report_20260613.md`를 추가했고, 7번은 breadth/ensemble/normalization/loss/scale suite를 가진 새 notebook+mirror+spec+report template로 분리했다. 하네스와 지침에도 “후속 연구는 새 번호”, “보고서는 항상 독립 문서” 규칙을 보강했다. | 커밋/푸시 후 6번 전용 메일 preset으로 교수님께 링크를 보내고, 실제 확장 결과는 7번 번호로만 누적한다. |
| 2026-06-15 | 7번은 아직 전체 breadth 학습기가 아니라 자원-인식 오케스트레이터라는 점을 코드에 명시하고, 학교 서버 4090 프로필을 추가 | 7번은 6번 이후 확장 의도를 유지해야 하지만, 현재 저장소에는 expanded family 전체를 직접 학습하는 backend가 아직 없음. 동시에 VSCode 원격 Jupyter에서는 stale kernel metadata와 invalid password/token 문제도 점검해야 함. | `test/models/7_optimization_breadth_expansion_test.ipynb/.py`에 `school_4090_15gb` 프로필, CPU/RAM/GPU/CUDA/PyTorch 자동 감지, BLAS/torch thread 제어, suite별 backend pending 표시를 추가했다. 7번 노트북의 실패 출력과 특정 원격 커널 metadata는 지우고 generic kernelspec으로 되돌렸다. `pyproject.toml`과 bootstrap에는 `ipywidgets`를 추가했고, `test/README.md`에는 fresh token 재연결과 `Canceled future ...` 점검 순서를 보강했다. | 서버에서 bootstrap을 다시 돌려 새 커널을 등록하고, VSCode에서 Existing Jupyter Server를 fresh token URL로 다시 연결한 뒤 7번 노트북을 새 커널로 연다. |
| 2026-06-16 | 학교 서버 JupyterLab 저버전 호환 세트를 저장소에 고정 | 실제 서버 확인 결과, `torch/cuda/ipykernel`은 정상이었고 핵심은 JupyterLab 저버전과 최신 widget/kernel stack의 호환성 문제였음. | `pyproject.toml`, 네 개 bootstrap 스크립트, smoke test, `test/README.md`에 `ipykernel==6.29.5`, `jupyter_client==8.6.3`, `traitlets==5.14.3`, `pyzmq==26.2.1`, `ipywidgets==8.1.8`, `jupyterlab_widgets` compatibility set을 반영했다. | 다음부터는 env 전체 재구축 전에 이 compatibility set을 먼저 적용해 보고, 그 다음 VSCode를 fresh token URL로 재연결한다. |
