"""서비스 파이프라인 조립: 시세 → 뉴스 → 지표 묶음 → 규칙 기준선 → (선택) LLM 판단.

구조
- `ServiceConfig`: 실행 설정 한곳.
- `Step`(추상): 단계 하나. `run(ctx)`가 요약 dict를 돌려준다. 단계는 도메인 모듈(marts·contexts·agents)을 부르기만 하고
  계산 로직을 갖지 않는다.
- `CandleStep`·`NewsStep`·`IndicatorStep`·`BaselineStep`·`JudgeStep`: 실제 단계.
- `RunLog`: AGENTS.md 2.14 형식으로 `logs/service/status.txt`(단계 시작·완료·실패)와 `mem.log`(단계마다 사용 가능 RAM·
  부하·GPU 메모리)를 남긴다. 연구 세션이 무거운 작업 전에 이 기록을 본다.
- `ServicePipeline`: 단계를 순서대로 실행한다. 한 단계가 실패하면 그 뒤 단계는 건너뛰고 실패를 기록한다
  (앞 단계 결과를 쓰기 때문).

LLM 판단(`JudgeStep`)은 기본으로 빠져 있다. 연구 28번이 GPU를 쓰는 동안(2026-10-23 무렵까지) 연구 세션과 합의한 대로
사람이 켤 때만(`ServiceConfig.run_judge=True`) 단발로 돈다.
"""

from __future__ import annotations

import json
import subprocess
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from common.errors import ServiceError
from database.paths import resolve_db_path, resolve_service_db_path


@dataclass
class ServiceConfig:
    service_db: str = field(default_factory=resolve_service_db_path)
    snapshot_db: str = field(default_factory=lambda: resolve_db_path("data/upbit_data_20261012.db"))
    candle_count: int = 400  # 종목당 최근 15분봉 수(약 4일). 스냅숏 DB 이후 공백을 메울 만큼
    news_lookback_days: int = 7
    news_max_items: int = 30
    collect_news: bool = True
    run_judge: bool = False  # 2026-10-23 무렵까지 사람이 켤 때만
    judge_tickers: list[str] | None = None
    log_dir: Path = Path("logs/service")


@dataclass
class RunContext:
    config: ServiceConfig
    started_at: datetime = field(default_factory=datetime.now)
    results: dict[str, dict] = field(default_factory=dict)


class RunLog:
    """단계별 상태와 자원 기록(AGENTS.md 2.14)."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.status, self.mem = log_dir / "status.txt", log_dir / "mem.log"

    @staticmethod
    def _now() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def event(self, text: str) -> None:
        with open(self.status, "a") as fh:
            fh.write(f"{self._now()} {text}\n")

    def resources(self) -> str:
        avail = next(int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if l.startswith("MemAvailable:"))
        load = Path("/proc/loadavg").read_text().split()[0]
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                                 capture_output=True, text=True, timeout=10).stdout.strip().replace(" ", "")
        except (OSError, subprocess.SubprocessError):  # 자원 기록은 보조 정보라 실패해도 단계는 계속한다
            gpu = "unknown"
        line = f"{avail / 1048576:.1f}GB load {load} gpu {gpu}"
        with open(self.mem, "a") as fh:
            fh.write(f"{self._now()} {line}\n")
        return line


class Step(ABC):
    name: str
    required: bool = True  # 실패하면 뒤 단계를 멈출지. 뒤 단계가 이 결과 없이도 돌 수 있으면 False

    @abstractmethod
    def run(self, ctx: RunContext) -> dict: ...


class CandleStep(Step):
    name = "candles"

    def run(self, ctx: RunContext) -> dict:
        from marts.live_candles import fetch_all

        return fetch_all(ctx.config.service_db, ctx.config.candle_count)


class NewsStep(Step):
    name = "news"
    required = False  # 뉴스가 없어도 지표·기준선은 계산할 수 있다

    def run(self, ctx: RunContext) -> dict:
        if not ctx.config.collect_news:
            return {"skipped": "collect_news=False"}
        from pipelines.ingest_text_context import ingest

        r = ingest(ctx.config.service_db, ctx.config.news_lookback_days, ctx.config.news_max_items)
        r.pop("per_ticker", None)
        return r


class IndicatorStep(Step):
    name = "indicators"

    def run(self, ctx: RunContext) -> dict:
        from marts.indicators import build_all

        got = build_all(ctx.config.service_db, ctx.config.snapshot_db)
        snaps = list(got.ok.values())
        return {"tickers": len(snaps), "as_of": snaps[0]["as_of"], "failed": got.failed,
                "garch_suspect": sum(s["garch_t"]["fit_suspect"] for s in snaps)}


class BaselineStep(Step):
    name = "baselines"

    def run(self, ctx: RunContext) -> dict:
        from agents.baselines import BaselineSuite

        suite = BaselineSuite(ctx.config.service_db)
        df = suite.run()
        counts = df.groupby(["rule", "decision"]).size().unstack(fill_value=0).to_dict(orient="index") if len(df) else {}
        return {"rows": len(df), "failed": suite.failed, "by_rule": counts}


class JudgeStep(Step):
    name = "judge"

    def run(self, ctx: RunContext) -> dict:
        if not ctx.config.run_judge:
            return {"skipped": "run_judge=False(연구 GPU 우선 기간)"}
        from agents.risk_judge import RiskJudge

        got = RiskJudge(ctx.config.service_db).run(ctx.config.judge_tickers, log_path=ctx.config.log_dir / "llm_server.log")
        return {"judged": len(got.ok), "failed": got.failed,
                "decisions": {tk: f"{r.decision} ({r.probs[r.decision]:.2f})" for tk, r in got.ok.items()}}


class ServicePipeline:
    def __init__(self, config: ServiceConfig, steps: list[Step] | None = None):
        self.config = config
        self.steps = steps or [CandleStep(), NewsStep(), IndicatorStep(), BaselineStep(), JudgeStep()]
        names = [s.name for s in self.steps]
        if len(set(names)) != len(names):  # 결과를 이름으로 모으므로 겹치면 앞 결과가 덮인다
            raise ValueError(f"단계 이름이 겹친다: {names}")
        self.log = RunLog(config.log_dir)

    def run(self) -> RunContext:
        ctx = RunContext(self.config)
        self.log.event(f"파이프라인 시작 ({', '.join(s.name for s in self.steps)})")
        failed = None
        for step in self.steps:
            if failed:
                ctx.results[step.name] = {"skipped": f"앞 단계 {failed} 실패"}
                self.log.event(f"건너뜀 {step.name}(앞 단계 {failed} 실패)")
                continue
            t0 = time.time()
            self.log.event(f"시작 {step.name}")
            try:
                out = step.run(ctx)
            except ServiceError as exc:  # 예상한 실패: 사유만 남긴다
                ctx.results[step.name] = {"error": f"{type(exc).__name__}: {exc}"}
            except Exception as exc:  # 예상 밖(코드 결함 등): 추적 정보까지 남긴다
                ctx.results[step.name] = {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=5)}
            else:
                out.update(seconds=round(time.time() - t0, 1), resources=self.log.resources())
                ctx.results[step.name] = out
                self.log.event(f"완료 {step.name} {out['seconds']}초" + (f", 일부 실패 {list(out['failed'])}" if out.get("failed") else ""))
                continue
            self.log.event(f"실패 {step.name} {ctx.results[step.name]['error'][:200]}")
            failed = step.name if step.required else failed
        errors = [k for k, v in ctx.results.items() if "error" in v]
        self.log.event("파이프라인 끝" + (f"(중단: {failed} 실패)" if failed else f"(필수 아닌 단계 실패: {errors})" if errors else ""))
        return ctx

    @staticmethod
    def summary(ctx: RunContext) -> str:
        return json.dumps(ctx.results, ensure_ascii=False, indent=1, default=str)
