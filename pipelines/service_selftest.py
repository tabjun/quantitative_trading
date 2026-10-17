"""서비스 오프라인 자체 시험: 합성 데이터·임시 디렉터리만 쓰고 서비스 DB와 네트워크는 건드리지 않는다.

코드를 고칠 때마다 반복해서 돌린다(사용자 지시 2026-10-18: 데이터는 필요할 때 한 번만 수동 실행하고, 보완은 단계에 맞는
시험을 반복). 실데이터 확인이 필요하면 `python main.py`를 사람이 한 번 실행한다.

    python -m pipelines.service_selftest      # 전부 통과하면 종료 코드 0
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from common.errors import DataSourceError, ResourceUnavailable, ServiceError, each


def check_each() -> str:
    def fetch(k: str) -> str:
        if k == "b":
            raise DataSourceError("시험용")
        return k

    got = each(["a", "b"], fetch)
    assert got.ok == {"a": "a"} and list(got.failed) == ["b"], got
    try:
        each([1], lambda k: 1 / 0)
    except ZeroDivisionError:
        return "예상한 실패만 모으고 예상 밖 예외는 올린다"
    raise AssertionError("예상 밖 예외를 삼켰다")


def check_pipeline() -> str:
    from pipelines.service_pipeline import ServiceConfig, ServicePipeline, Step

    class Ok(Step):
        def __init__(self, name: str):
            self.name = name

        def run(self, ctx):
            return {"done": True}

    class Fail(Step):
        def __init__(self, name: str, required: bool):
            self.name, self.required = name, required

        def run(self, ctx):
            raise DataSourceError("시험용")

    with tempfile.TemporaryDirectory() as tmp:
        cfg = ServiceConfig(log_dir=Path(tmp))
        soft = ServicePipeline(cfg, [Ok("a"), Fail("b", required=False), Ok("c")]).run().results
        hard = ServicePipeline(cfg, [Ok("a"), Fail("b", required=True), Ok("c")]).run().results
    assert soft["c"].get("done") and "error" in soft["b"], soft
    assert "skipped" in hard["c"], hard
    try:
        ServicePipeline(ServiceConfig(log_dir=Path(tempfile.gettempdir())), [Ok("x"), Ok("x")])
    except ValueError:
        return "필수 아닌 단계 실패는 계속, 필수 단계 실패는 뒤 단계 건너뜀, 이름 중복 차단"
    raise AssertionError("단계 이름 중복을 막지 않았다")


def _bounds() -> pd.DataFrame:
    q = {"q20": 0.004, "q40": 0.006, "q60": 0.008, "q80": 0.012}
    return pd.DataFrame([{"ticker": "KRW-TEST", "horizon_min": h, **q} for h in (60, 240, 720)]).set_index("horizon_min")


def _snapshot(regime: str, rv12: float, no_trade: float = 0.0) -> dict:
    return {"ticker": "KRW-TEST", "as_of": "2026-10-18 04:00:00", "data_end_bar": "2026-10-18 03:45:00",
            "last_trade_bar": "2026-10-18 03:45:00",
            "regime": {"values": {f"{h}m": {"regime": regime} for h in (60, 240, 720)}},
            "garch_t": {"rv_forecast": {"60m": rv12 / 3, "240m": rv12 / 1.7, "720m": rv12}},
            "price_stall": {"1d": {"zero_return_share": 0.1, "no_trade_share": no_trade}}}


def check_baselines() -> str:
    from agents.baselines import RuleInput, default_rules

    news = pd.DataFrame({"topic_risk": [1, 1], "ticker_hint": ["KRW-TEST", "MARKET"]})
    calm = RuleInput("KRW-TEST", pd.Timestamp("2026-10-18 04:00"), _snapshot("Q1", 0.005), _bounds(), news.iloc[:0])
    stormy = RuleInput("KRW-TEST", pd.Timestamp("2026-10-18 04:00"), _snapshot("Q5", 0.02), _bounds(), news)
    dc = {r.name: r.decide(calm).decision for r in default_rules()}
    ds = {r.name: r.decide(stormy).decision for r in default_rules()}
    assert set(dc.values()) == {"keep"}, dc
    assert ds["B1_regime_1h"] == ds["B2_garch_rv_12h"] == ds["B3_vol_target_12h"] == ds["B4_news_risk_6h"] == "reduce", ds
    assert ds["B0_always_keep"] == "keep" and ds["B5_b2_or_b4"] == "reduce", ds
    return "잔잔한 입력은 전부 유지, 요동·악재 입력은 B0 말고 전부 축소"


def check_prompt_labels() -> str:
    from agents.risk_judge import PromptBuilder

    lab = PromptBuilder.liquidity_label
    assert lab(0.0).startswith("정상") and lab(0.1).startswith("다소 낮음") and lab(0.6).startswith("낮음"), (lab(0.0), lab(0.1), lab(0.6))
    assert "위험 신호가 아니다" in PromptBuilder.flat_label(0.5, 0.0)
    state = PromptBuilder().state(_snapshot("Q4", 0.013, no_trade=0.0), _bounds().reset_index())
    assert state["garch_t_rv_forecast_level"]["720m"] == "Q5" and state["liquidity"].startswith("정상"), state
    return "유동성·가격 정지 문장 판정과 예측 RV 분위 칸"


def check_garch() -> str:
    from marts.indicators import garch_t_forecast

    rng = np.random.default_rng(0)
    n, om, al, be = 6000, 1e-7, 0.08, 0.9
    h, r = om / (1 - al - be), np.empty(n)
    for i in range(n):  # GARCH(1,1)-t 합성 15분 수익률
        r[i] = np.sqrt(h) * rng.standard_t(6) / np.sqrt(6 / 4)
        h = om + al * r[i] ** 2 + be * h
    grid = pd.date_range("2026-01-01", periods=n, freq="15min")
    g = garch_t_forecast(grid, r, grid[4000])
    rv = g["rv_forecast"]
    assert not g["fit_suspect"] and all(np.isfinite(v) and v > 0 for v in rv.values()), g
    assert rv["60m"] < rv["240m"] < rv["720m"], rv
    return f"합성 GARCH 적합 척도 ×{g['scale']}, 예측 RV가 구간에 따라 커짐"


def check_time_alignment() -> str:
    from contexts.text_context import selftest_time_alignment

    out = selftest_time_alignment()
    return f"뉴스 선행 없음(이전 방식 {out['cases'][0]['old_lead_hours']}시간)"


def check_headroom() -> str:
    from agents.llm_runtime import LlmServerConfig, check_headroom as headroom

    try:
        headroom(LlmServerConfig(gpu_budget_mib=10 ** 7))
    except ResourceUnavailable:
        return "여유선 부족이면 서버를 띄우기 전에 멈춘다"
    raise AssertionError("여유선 부족을 통과시켰다")


CHECKS: dict[str, Callable[[], str]] = {
    "each 도우미": check_each, "파이프라인 실패 처리": check_pipeline, "규칙 기준선": check_baselines,
    "지시문 전처리": check_prompt_labels, "GARCH-t 적합": check_garch, "뉴스 시각": check_time_alignment,
    "LLM 여유선": check_headroom,
}


def main() -> int:
    got = each(CHECKS, lambda k: CHECKS[k](), expected=(AssertionError, ServiceError))
    for name in CHECKS:
        print(f"[{'통과' if name in got.ok else '실패'}] {name}: {got.ok.get(name) or got.failed.get(name)}")
    print(f"{len(got.ok)}/{len(CHECKS)} 통과")
    return 0 if not got.failed else 1


if __name__ == "__main__":
    sys.exit(main())
