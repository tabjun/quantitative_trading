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
import logging
import subprocess
import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from common.errors import ServiceError
from common.logging_setup import set_run_id
from common.ops_store import OpsStore
from database.paths import resolve_db_path, resolve_service_db_path


log = logging.getLogger(__name__)


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
    trigger: str = "manual"  # 실행 방법(manual·cron 등), ops_runs에 남는다


@dataclass
class RunContext:
    config: ServiceConfig
    run_id: str = "-"
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

    def params(self, ctx: RunContext) -> dict:
        """이 단계가 쓰는 설정·초매개변수(ops_params에 남는다)."""
        return {}


class CandleStep(Step):
    name = "candles"

    def run(self, ctx: RunContext) -> dict:
        from marts.live_candles import fetch_all

        r = fetch_all(ctx.config.service_db, ctx.config.candle_count)
        return {**r, "items_ok": r["tickers"]}

    def params(self, ctx: RunContext) -> dict:
        return {"source": "upbit 공개 API(pyupbit)", "interval": "minute15", "candle_count": ctx.config.candle_count,
                "closed_bars_only": True}


class NewsStep(Step):
    name = "news"
    required = False  # 뉴스가 없어도 지표·기준선은 계산할 수 있다

    def run(self, ctx: RunContext) -> dict:
        if not ctx.config.collect_news:
            return {"skipped": "collect_news=False"}
        from pipelines.ingest_text_context import ingest

        r = ingest(ctx.config.service_db, ctx.config.news_lookback_days, ctx.config.news_max_items)
        r.pop("per_ticker", None)
        return {**r, "items_ok": r["collected"]}

    def params(self, ctx: RunContext) -> dict:
        from contexts.text_context import DEFAULT_RSS_QUERIES

        return {"lookback_days": ctx.config.news_lookback_days, "max_items_per_source": ctx.config.news_max_items,
                "market_queries": DEFAULT_RSS_QUERIES, "ticker_query": '("한국어 이름" OR 심볼) 코인, 제목에 종목 없으면 버림',
                "time_rule": "KST, max(발행, 처음 수집)이 속한 봉"}


class IndicatorStep(Step):
    name = "indicators"

    def run(self, ctx: RunContext) -> dict:
        from marts.indicators import build_all

        got = build_all(ctx.config.service_db, ctx.config.snapshot_db)
        snaps = list(got.ok.values())
        return {"tickers": len(snaps), "items_ok": len(snaps), "as_of": snaps[0]["as_of"], "failed": got.failed,
                "garch_suspect": sum(s["garch_t"]["fit_suspect"] for s in snaps)}

    def params(self, ctx: RunContext) -> dict:
        from marts import indicators as ind, regime_boundaries as rb

        return {"snapshot_db": ctx.config.snapshot_db, "garch": "GARCH(1,1)-t 평균 0, 주기 스케일 보정",
                "garch_scales": ind.GARCH_SCALES, "fit_window": [str(rb.DATA_START), str(rb.SPLIT)],
                "forecast_min": ind.FORECAST_MIN, "regime_min": ind.REGIME_MIN, "regime_version": rb.VERSION}


class BaselineStep(Step):
    name = "baselines"

    def run(self, ctx: RunContext) -> dict:
        from agents.baselines import BaselineSuite

        suite = BaselineSuite(ctx.config.service_db)
        df = suite.run()
        counts = df.groupby(["rule", "decision"]).size().unstack(fill_value=0).to_dict(orient="index") if len(df) else {}
        return {"rows": len(df), "items_ok": df["ticker"].nunique() if len(df) else 0, "failed": suite.failed, "by_rule": counts}

    def params(self, ctx: RunContext) -> dict:
        from agents.baselines import RULE_VERSION, default_rules

        return {"rule_version": RULE_VERSION,
                "rules": {r.name: {k: v for k, v in vars(r).items() if not k.startswith("_") and k not in ("name", "rules")}
                          for r in default_rules()}}


class JudgeStep(Step):
    name = "judge"

    def run(self, ctx: RunContext) -> dict:
        if not ctx.config.run_judge:
            return {"skipped": "run_judge=False(연구 GPU 우선 기간)"}
        from agents.risk_judge import RiskJudge

        judge = RiskJudge(ctx.config.service_db)
        got = judge.run(ctx.config.judge_tickers, log_path=ctx.config.log_dir / "llm_server.log")
        return {"judged": len(got.ok), "reused": judge.reused, "items_ok": len(got.ok), "failed": got.failed,
                "decisions": {tk: f"{r.decision} ({r.probs[r.decision]:.2f})" for tk, r in got.ok.items()}}

    def params(self, ctx: RunContext) -> dict:
        if not ctx.config.run_judge:
            return {}
        from agents.llm_runtime import LlmServerConfig
        from agents.risk_judge import NO_TRADE_HIGH, NO_TRADE_LOW, PERMS, PROMPT_VERSION, RiskJudge

        cfg = LlmServerConfig()
        return {"prompt_version": PROMPT_VERSION, "model_file": cfg.model_path.name, "ctx_per_slot": cfg.ctx_per_slot,
                "parallel": cfg.parallel, "temperature": 0, "option_permutations": PERMS,
                "news_lookback_hours": RiskJudge.news_lookback_hours, "liquidity_bounds": [NO_TRADE_LOW, NO_TRADE_HIGH]}


class ServicePipeline:
    def __init__(self, config: ServiceConfig, steps: list[Step] | None = None):
        self.config = config
        self.steps = steps or [CandleStep(), NewsStep(), IndicatorStep(), BaselineStep(), JudgeStep()]
        names = [s.name for s in self.steps]
        if len(set(names)) != len(names):  # 결과를 이름으로 모으므로 겹치면 앞 결과가 덮인다
            raise ValueError(f"단계 이름이 겹친다: {names}")
        self.log = RunLog(config.log_dir)
        self.ops = OpsStore(config.service_db)

    def _run_step(self, ctx: RunContext, step: Step) -> tuple[str, dict]:
        """(상태, 결과). 예상한 실패는 사유만, 예상 밖 실패는 추적 정보까지 남긴다."""
        t0 = time.time()
        try:
            out = step.run(ctx)
        except ServiceError as exc:
            log.error("단계 실패 %s: %s", step.name, exc)
            return "failed", {"error": f"{type(exc).__name__}: {exc}", "seconds": round(time.time() - t0, 1)}
        except Exception as exc:
            log.exception("단계 예상 밖 오류 %s", step.name)
            return "failed", {"error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc(limit=5),
                              "seconds": round(time.time() - t0, 1)}
        out.update(seconds=round(time.time() - t0, 1), resources=self.log.resources())
        log.info("단계 완료 %s %.1f초 성공 %s 실패 %d", step.name, out["seconds"], out.get("items_ok"), len(out.get("failed") or {}))
        return "ok", out

    def run(self) -> RunContext:
        ctx = RunContext(self.config)
        ctx.run_id = self.ops.start_run(asdict(self.config), self.config.trigger)
        set_run_id(ctx.run_id)
        self.log.event(f"파이프라인 시작 {ctx.run_id} ({', '.join(s.name for s in self.steps)})")
        failed = None
        for step in self.steps:
            started = datetime.now()
            param_hash = self.ops.record_params(step.name, step.params(ctx))
            if failed:
                status, out = "skipped", {"skipped": f"앞 단계 {failed} 실패"}
                self.log.event(f"건너뜀 {step.name}(앞 단계 {failed} 실패)")
            else:
                self.log.event(f"시작 {step.name}")
                status, out = self._run_step(ctx, step)
                self.log.event(f"{'완료' if status == 'ok' else '실패'} {step.name} {out['seconds']}초"
                               + (f" {out['error'][:200]}" if "error" in out else "")
                               + (f", 일부 실패 {list(out['failed'])}" if out.get("failed") else ""))
                failed = step.name if status == "failed" and step.required else failed
            ctx.results[step.name] = out
            self.ops.record_step(ctx.run_id, step.name, started, status, out, param_hash)
        errors = [k for k, v in ctx.results.items() if "error" in v or v.get("failed")]
        status = "failed" if failed else "partial" if errors else "ok"
        self.ops.finish_run(ctx.run_id, status)
        self.log.event(f"파이프라인 끝 {ctx.run_id} 상태 {status}" + (f" (문제 단계 {errors})" if errors else ""))
        set_run_id("-")
        return ctx

    @staticmethod
    def summary(ctx: RunContext) -> str:
        return json.dumps(ctx.results, ensure_ascii=False, indent=1, default=str)
