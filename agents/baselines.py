"""규칙 기반 기준선 B0~B5(docs/service_design.md §11-3, 초안 — 사용자 확정 전).

LLM 판단이 단순 규칙보다 나은지 나중에 같은 라벨·같은 비용으로 비교하려고, 판단 시각마다 규칙 판단도 함께 남긴다.
규칙은 판단 시각에 아는 값만 쓴다(지표 묶음 = `indicator_snapshots`, 뉴스 = `load_news_as_of`). 정확도는 계산하지
않는다(라벨 §11-2 미확정, 2026-10-06 이후는 누설 규칙). 경계값은 모두 초안이며 확정 때 규칙 객체의 인자만 바꾼다.

구조: 규칙 하나 = `Rule` 하위 클래스(전략 패턴). `BaselineSuite`가 규칙 목록을 종목별 입력(`RuleInput`)에 적용해
`baseline_decisions`에 저장한다. 새 규칙은 `Rule`을 상속해 `decide`만 구현하고 `default_rules()`에 더한다.
판단 순서(신중한 정도): 유지(keep) < 관망(watch) < 축소(reduce).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

import duckdb
import pandas as pd

from contexts.text_context import TextFeatureBuilder

RULE_VERSION = "draft_20261018"
TABLE = "baseline_decisions"
CAUTION = {"keep": 0, "watch": 1, "reduce": 2}


@dataclass(frozen=True)
class Decision:
    decision: str  # keep / watch / reduce
    why: str


@dataclass(frozen=True)
class RuleInput:
    """규칙 하나가 종목 하나를 판단할 때 보는 값. 모두 판단 시각(as_of)에 알려진 것."""

    ticker: str
    as_of: pd.Timestamp
    snapshot: dict  # indicator_snapshots.payload
    bounds: pd.DataFrame  # 그 종목의 국면 경계(horizon_min 인덱스)
    news: pd.DataFrame  # load_news_as_of 결과(종목 + 시장)

    @property
    def regime(self) -> dict[str, str | None]:
        return {k: v["regime"] for k, v in self.snapshot["regime"]["values"].items()}

    @property
    def rv_forecast(self) -> dict[str, float]:
        return self.snapshot["garch_t"]["rv_forecast"]


def regime_to_decision(q: str | None) -> str:
    """국면 칸 Q1~Q5 → Q5 축소, Q4 관망, 그 밖 유지. 국면을 모르면(점검 등) 관망."""
    if q is None:
        return "watch"
    return {"Q5": "reduce", "Q4": "watch"}.get(q, "keep")


def level_of(value: float, row: pd.Series) -> str:
    """값을 학습 구간 직전 RV 5분위 칸(Q1~Q5)으로 읽는다."""
    return f"Q{sum(value >= row[c] for c in ('q20', 'q40', 'q60', 'q80')) + 1}"


class Rule(ABC):
    name: str

    @abstractmethod
    def decide(self, x: RuleInput) -> Decision: ...


class AlwaysKeep(Rule):
    name = "B0_always_keep"

    def decide(self, x: RuleInput) -> Decision:
        return Decision("keep", "항상 유지")


class RegimeRule(Rule):
    """B1: 직전 H분 국면이 Q5면 축소, Q4면 관망."""

    def __init__(self, horizon_min: int):
        self.horizon_min = horizon_min
        self.name = f"B1_regime_{horizon_min // 60}h"

    def decide(self, x: RuleInput) -> Decision:
        q = x.regime.get(f"{self.horizon_min}m")
        return Decision(regime_to_decision(q), f"직전 {self.horizon_min // 60}시간 국면 {q}")


class ForecastLevelRule(Rule):
    """B2: GARCH-t H분 예측 RV가 학습 구간 직전 H분 RV의 Q5 칸이면 축소, Q4면 관망."""

    def __init__(self, horizon_min: int = 720):
        self.horizon_min = horizon_min
        self.name = f"B2_garch_rv_{horizon_min // 60}h"

    def decide(self, x: RuleInput) -> Decision:
        rv = x.rv_forecast[f"{self.horizon_min}m"]
        if self.horizon_min not in x.bounds.index:
            return Decision("watch", "국면 경계 없음")
        lv = level_of(rv, x.bounds.loc[self.horizon_min])
        return Decision(regime_to_decision(lv), f"GARCH-t {self.horizon_min // 60}시간 예측 RV {rv:.4f}가 {lv} 칸")


class VolTargetRule(Rule):
    """B3: 목표 비중 min(1, 목표 변동성 / 예측 변동성)을 세 단계로. 목표 = 학습 구간 직전 RV 중앙값 근사."""

    def __init__(self, horizon_min: int = 720, keep_at: float = 0.8, watch_at: float = 0.5):
        self.horizon_min, self.keep_at, self.watch_at = horizon_min, keep_at, watch_at
        self.name = f"B3_vol_target_{horizon_min // 60}h"

    def decide(self, x: RuleInput) -> Decision:
        if self.horizon_min not in x.bounds.index:
            return Decision("watch", "국면 경계 없음")
        row = x.bounds.loc[self.horizon_min]
        target = float((row["q40"] + row["q60"]) / 2)  # 경계 테이블에 q50이 없어 q40·q60 평균으로 근사
        rv = x.rv_forecast[f"{self.horizon_min}m"]
        w = min(1.0, target / rv) if rv > 0 else 1.0
        d = "keep" if w >= self.keep_at else "watch" if w >= self.watch_at else "reduce"
        return Decision(d, f"목표 비중 min(1, {target:.4f} / {rv:.4f}) = {w:.2f}")


class NewsRiskRule(Rule):
    """B4: 최근 lookback 시간 종목·시장 기사 중 위험 주제 기사가 reduce_at건 이상 축소, watch_at건 이상 관망."""

    name = "B4_news_risk_6h"

    def __init__(self, reduce_at: int = 2, watch_at: int = 1):
        self.reduce_at, self.watch_at = reduce_at, watch_at

    def decide(self, x: RuleInput) -> Decision:
        n = int(x.news["topic_risk"].sum()) if len(x.news) else 0
        d = "reduce" if n >= self.reduce_at else "watch" if n >= self.watch_at else "keep"
        return Decision(d, f"위험 주제 기사 {n}건")


class MostCautious(Rule):
    """B5: 여러 규칙 중 가장 신중한 판단."""

    def __init__(self, name: str, rules: list[Rule]):
        self.name, self.rules = name, rules

    def decide(self, x: RuleInput) -> Decision:
        parts = [(r.name, r.decide(x)) for r in self.rules]
        best = max((d for _, d in parts), key=lambda d: CAUTION[d.decision])
        return Decision(best.decision, " / ".join(f"{n} {d.decision}" for n, d in parts) + " 중 가장 신중한 쪽")


def default_rules() -> list[Rule]:
    b2, b4 = ForecastLevelRule(720), NewsRiskRule()
    return [AlwaysKeep(), RegimeRule(60), RegimeRule(240), b2, VolTargetRule(720), b4,
            MostCautious("B5_b2_or_b4", [b2, b4])]


class BaselineSuite:
    """규칙 목록을 최신(또는 지정 시각) 지표 묶음의 모든 종목에 적용해 저장한다."""

    news_lookback_hours = 6

    def __init__(self, service_db: str, rules: list[Rule] | None = None):
        self.service_db = service_db
        self.rules = rules or default_rules()
        self.news = TextFeatureBuilder(db_path=service_db)

    def inputs(self, as_of: pd.Timestamp | None = None) -> list[RuleInput]:
        with duckdb.connect(self.service_db, read_only=True) as con:
            when = as_of or con.execute("SELECT MAX(as_of) FROM indicator_snapshots").fetchone()[0]
            snaps = con.execute("SELECT ticker, payload FROM indicator_snapshots WHERE as_of = ?", [when]).fetchall()
            bounds = con.execute("SELECT * FROM regime_boundaries").df()
        out = []
        for tk, payload in snaps:
            snap = json.loads(payload)
            t = pd.Timestamp(snap["as_of"])
            out.append(RuleInput(tk, t, snap, bounds[bounds["ticker"] == tk].set_index("horizon_min"),
                                 self.news.load_news_as_of(t.to_pydatetime(), ticker=tk, lookback_hours=self.news_lookback_hours)))
        return out

    def run(self, as_of: pd.Timestamp | None = None) -> pd.DataFrame:
        rows = []
        for x in self.inputs(as_of):
            for rule in self.rules:
                d = rule.decide(x)
                rows.append({"as_of": x.as_of, "ticker": x.ticker, "rule": rule.name, "decision": d.decision,
                             "why": d.why, "rule_version": RULE_VERSION, "logged_at": datetime.now()})
        df = pd.DataFrame(rows)
        if not df.empty:
            self._save(df)
        return df

    def _save(self, df: pd.DataFrame) -> None:
        with duckdb.connect(self.service_db) as con:
            con.register("b", df)
            con.execute(f"CREATE TABLE IF NOT EXISTS {TABLE} AS SELECT * FROM b WHERE FALSE")
            con.execute(f"DELETE FROM {TABLE} t USING b WHERE t.as_of = b.as_of AND t.ticker = b.ticker "
                        "AND t.rule = b.rule AND t.rule_version = b.rule_version")
            con.execute(f"INSERT INTO {TABLE} SELECT * FROM b")
            con.unregister("b")
