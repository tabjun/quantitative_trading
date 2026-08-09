"""공용 연구 엔진 패키지.

번호 실험(`test/models/*.ipynb`)이 서로의 `.py`를 `load_module`로 끌어다 쓰던
코드 의존성을 끊기 위한 공용 엔진 모듈을 둔다. 실험은 이 패키지를 import해서
쓰고, 이전 실험 파일을 직접 실행하지 않는다.

- 이전 실험의 *결과/결론*은 참고하되, *함수*를 import하는 의존성은 갖지 않는다.
- read-only로 동결된 완료 노트북(8~13)을 수정하지 않고도, 교정된 엔진을 여기서
  단일 source로 유지한다.

## import 계약 (cwd 독립)

이 repo의 다른 실사용 패키지(`contexts`, `marts`, `analysis`, `pipelines`)와 동일하게,
`quantitative_trading/`를 import 루트(sys.path)로 두고 **top-level `import engine`**로 쓴다.
`quantitative_trading.engine`처럼 부모 패키지 경로로는 import하지 않는다
(repo에 `quantitative_trading/__init__.py`를 두지 않는 기존 관례를 따른다).

따라서 노트북/드라이버는 엔진을 쓰기 전에 repo root를 sys.path에 넣어야 한다:

    import sys
    from pathlib import Path
    def _engine_root(start: Path) -> Path:
        # engine/와 pyproject.toml을 함께 가진 디렉터리(=quantitative_trading)를 sys.path 루트로 쓴다.
        # cwd가 repo root(personal_ai_project)든 test/models/든 동작하도록 부모와 홈 fallback을 함께 본다.
        candidates = [start, *start.parents, Path.home() / "personal_ai_project" / "quantitative_trading"]
        for c in candidates:
            if (c / "pyproject.toml").exists() and (c / "engine").is_dir():
                return c
        raise RuntimeError("engine을 담은 quantitative_trading 디렉터리를 찾지 못했다. 직접 sys.path에 넣어라.")
    ROOT = _engine_root(Path.cwd())
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import engine.fusion  # 이후 from engine import ... 가능

엔진 내부 모듈은 모두 상대 import(`from . import ...`)라, 이 계약만 지키면 cwd에 무관하게 동작한다.
"""
