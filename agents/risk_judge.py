"""4단계 판단: 보조 지표 스냅숏 + 판단 시각 이전 뉴스 → 유지/관망/축소 확률과 근거(로컬 LLM).

전처리: 스냅숏 JSON(사전 국면, GARCH-t 예측 RV와 그 값이 학습 구간 직전 RV 분포의 몇 분위 칸인지, 가격 정지 비율)과
`load_news_as_of`(max(발행, 처음 수집) <= as_of)로 고른 종목·시장 기사 제목을 지시문에 넣는다. 미래 수익률 같은
방향 정보와 유사 흐름 마트는 넣지 않는다(마트 연결은 연구 검증 전 보류).
판단: 지시문 v3(v1 = docs/service_design.md §10-1, v2·v3 = §10-7)의 보기를 A/B/C 한 글자로 두고, 보기 배치 3가지(순환)의 다음 1토큰
확률 평균으로 정한다(잔잔한 국면의 위치 편향 대응). 설명: 판단을 고정한 채 JSON 스키마로 근거를 받는다.
후처리: 판단과 위험 수준이 어긋나면(축소인데 위험 1~2, 유지인데 위험 4~5) `mismatch`로 표시하고,
입력·출력·모델 파일 해시를 `decision_log`에 남긴다(페이퍼 기록, 주문 없음).
확률은 아직 보정하지 않았다(라벨 정의 §11 확정 전).

구조: `PromptBuilder`(전처리: 지표 묶음·뉴스 → 지시문), `RiskJudge`(판단·근거·불일치 표시, 서버는 `run` 안에서만
떠 있다), `DecisionLog`(저장). `JudgeResult`가 한 종목의 결과다.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from agents.baselines import level_of
from common.errors import BatchResult, DataIntegrityError, LlmError, each
from agents.llm_runtime import LlmServerConfig, llm_server, option_probs, post_json
from contexts.text_context import TextFeatureBuilder
from database.paths import resolve_service_db_path

OPTIONS = ("keep", "watch", "reduce")
NAMES = {"keep": "유지", "watch": "관망", "reduce": "축소"}
DESC = {
    "keep": "유지: 현재 포지션을 그대로 두고 신규 진입도 허용한다",
    "watch": "관망: 현재 포지션은 그대로 두되 신규 진입은 보류한다",
    "reduce": "축소: 포지션 비중을 줄인다",
}
PERMS = [("keep", "watch", "reduce"), ("reduce", "keep", "watch"), ("watch", "reduce", "keep")]
SYSTEM_PROMPT = (
    "너는 암호화폐 포지션의 위험 관리자다. 방향(오를지 내릴지)은 예측하지 않는다. 주어진 국면·변동성 지표·뉴스만 근거로, "
    "아래 기준에 따라 세 가지 중 하나를 고른다. 위험 신호가 없으면 유지가 기본이다.\n"
    "- 유지: 위험 신호가 평소 수준이다. 직전 변동성 국면이 Q1~Q3이고, 변동성 예측이 평소 범위이며, 거래소·보안·규제 같은 직접 악재 뉴스가 없다.\n"
    "- 관망: 위험 신호가 엇갈리거나 한 가지만 나쁘다, 또는 판단할 정보가 부족하다.\n"
    "- 축소: 위험 신호가 둘 이상 함께 나쁘다. 예: 요동 국면(Q4~Q5)과 변동성 예측 상승, 직접 악재 뉴스(해킹·입출금 중단·상장 폐지 등), "
    "거래 없는 봉 증가(유동성 저하).\n"
    "용어: 국면 Q1~Q5는 그 종목의 과거(학습 구간) 직전 변동성 5분위다. Q1이 가장 잔잔하고 Q5가 가장 요동친다. "
    "liquidity와 price_flat_note는 이미 해석된 문장이므로 그 판정(정상/낮음, 위험 신호인지 아닌지)을 그대로 따른다. "
    "뉴스가 '종목 기사 없음'이면 종목 고유 악재가 확인되지 않은 것이지 악재가 없다는 증거는 아니다."
)
EXPLAIN_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_level": {"type": "integer", "minimum": 1, "maximum": 5, "description": "1=매우 낮음, 2=낮음, 3=보통, 4=높음, 5=매우 높음"},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 4},
        "uncertainty": {"type": "string", "enum": ["낮음", "보통", "높음"]},
        "rationale": {"type": "string"},
    },
    "required": ["risk_level", "evidence", "uncertainty", "rationale"],
}
PROMPT_VERSION = "v3_20261018"  # v2: 정지 비율 뜻 구분·종목 기사 없음 명시, v3: 유동성·가격 정지를 규칙으로 먼저 해석한 문장으로
LOG_TABLE = "decision_log"


NO_TRADE_LOW, NO_TRADE_HIGH = 0.05, 0.20  # 초안 경계(§11과 함께 확정): 거래 없는 15분 봉 비율(직전 1일)


@dataclass(frozen=True)
class JudgeResult:
    ticker: str
    as_of: pd.Timestamp
    decision: str
    probs: dict[str, float]
    probs_by_perm: list[dict[str, float]]
    position_spread: float
    option_mass_min: float
    explain: dict
    mismatch: bool
    state: dict
    news: list[str]


class PromptBuilder:
    """전처리: 지표 묶음과 판단 시각 이전 뉴스를 지시문으로 바꾼다. 숫자의 방향은 규칙으로 먼저 해석해 문장으로 준다(v3)."""

    version = PROMPT_VERSION
    system = SYSTEM_PROMPT

    @staticmethod
    def liquidity_label(no_trade: float) -> str:
        if no_trade >= NO_TRADE_HIGH:
            return f"낮음(위험 신호): 직전 하루 15분 봉의 {no_trade:.0%}에서 거래가 없었다"
        if no_trade >= NO_TRADE_LOW:
            return f"다소 낮음: 직전 하루 15분 봉의 {no_trade:.0%}에서 거래가 없었다"
        return f"정상(위험 신호 아님): 거래 없는 봉 {no_trade:.0%}"

    @staticmethod
    def flat_label(flat: float, no_trade: float) -> str:
        if flat >= 0.3 and no_trade < NO_TRADE_LOW:
            return f"가격이 직전 봉과 같은 봉이 {flat:.0%}지만 거래는 계속 있었다. 호가 한 칸이 가격 대비 커서 생기는 현상이며 위험 신호가 아니다"
        return f"가격이 직전 봉과 같은 봉 {flat:.0%}(판단 근거로 쓰지 않는다)"

    def state(self, snap: dict, bounds: pd.DataFrame) -> dict:
        tb = bounds[bounds["ticker"] == snap["ticker"]].set_index("horizon_min")
        rv = snap["garch_t"]["rv_forecast"]
        stall = snap["price_stall"]["1d"]
        return {
            "ticker": snap["ticker"],
            "as_of_kst": snap["as_of"],
            "prior_regime": {k: v["regime"] for k, v in snap["regime"]["values"].items()},
            "garch_t_rv_forecast": {k: round(v, 5) for k, v in rv.items()},
            "garch_t_rv_forecast_level": {k: level_of(v, tb.loc[int(k[:-1])]) if int(k[:-1]) in tb.index else None
                                          for k, v in rv.items()},
            # v3: 4B 모델이 비율의 방향을 거꾸로 읽어(v2 시험: 무체결 0%를 "유동성 극단 악화"로) 규칙으로 먼저 해석한 말을 준다.
            "liquidity": self.liquidity_label(stall["no_trade_share"]),
            "price_flat_note": self.flat_label(stall["zero_return_share"], stall["no_trade_share"]),
            "minutes_since_last_trade": int((pd.Timestamp(snap["data_end_bar"]) - pd.Timestamp(snap["last_trade_bar"])).total_seconds() // 60),
        }

    @staticmethod
    def news_lines(news: pd.DataFrame, ticker: str, k_ticker: int = 8, k_market: int = 5) -> list[str]:
        out = []
        if news.empty or not (news["ticker_hint"] == ticker).any():
            out.append("[종목 기사 없음] 최근 24시간 이 종목 이름이 제목에 든 기사가 수집되지 않았다(시장 기사만 아래에 있다).")
        if news.empty:
            return out
        for tag, k in ((ticker, k_ticker), ("MARKET", k_market)):
            sub = news[news["ticker_hint"] == tag].head(k)
            for _, row in sub.iterrows():
                label = "종목" if tag == ticker else "시장"
                out.append(f"[{label} {pd.Timestamp(row['effective_at']):%m-%d %H:%M}] {row['title']}")
        return out

    def messages(self, state: dict, news: list[str], perm: tuple[str, ...]) -> list[dict]:
        opts = ", ".join(f"{L}={DESC[o]}" for L, o in zip("ABC", perm))
        user = ("상태(JSON):\n" + json.dumps(state, ensure_ascii=False) + "\n최근 24시간 뉴스(판단 시각 이전만):\n"
                + ("\n".join(f"- {n}" for n in news) if news else "- 해당 없음(수집된 기사 없음)")
                + f"\n\n질문: 현재 포지션을 어떻게 할 것인가? 보기 {opts}. 보기 글자 하나로만 답하라.")
        return [{"role": "system", "content": self.system}, {"role": "user", "content": user}]


class DecisionLog:
    """판단 결과를 서비스 DB `decision_log`에 쌓는다(페이퍼 기록)."""

    def __init__(self, service_db: str):
        self.service_db = service_db

    @staticmethod
    def row(r: JudgeResult, model_file: str, model_sha256: str) -> dict:
        return {
            "logged_at": datetime.now(), "as_of": r.as_of, "ticker": r.ticker,
            "decision": r.decision, "p_keep": r.probs["keep"], "p_watch": r.probs["watch"], "p_reduce": r.probs["reduce"],
            "position_spread": r.position_spread, "risk_level": r.explain["risk_level"],
            "uncertainty": r.explain["uncertainty"], "mismatch": r.mismatch, "rationale": r.explain["rationale"],
            "input_json": json.dumps({"state": r.state, "news": r.news}, ensure_ascii=False),
            "output_json": json.dumps({"decision": r.decision, "probs": r.probs, "probs_by_perm": r.probs_by_perm,
                                       "position_spread": r.position_spread, "option_mass_min": r.option_mass_min,
                                       "explain": r.explain, "mismatch": r.mismatch}, ensure_ascii=False),
            "prompt_version": PROMPT_VERSION, "model_file": model_file, "model_sha256": model_sha256, "calibrated": False,
        }

    def save(self, rows: list[dict]) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        with duckdb.connect(self.service_db) as con:
            con.register("d", df)
            con.execute(f"CREATE TABLE IF NOT EXISTS {LOG_TABLE} AS SELECT * FROM d WHERE FALSE")
            con.execute(f"INSERT INTO {LOG_TABLE} SELECT * FROM d")
            con.unregister("d")


class RiskJudge:
    """최신 지표 묶음의 종목들을 판단한다. 로컬 LLM 서버는 `run` 안에서만 떠 있다(호출 때만 적재)."""

    news_lookback_hours = 24

    def __init__(self, service_db: str | None = None, cfg: LlmServerConfig | None = None, prompt: PromptBuilder | None = None):
        self.service_db = resolve_service_db_path(service_db)
        self.cfg = cfg or LlmServerConfig()
        self.prompt = prompt or PromptBuilder()
        self.news = TextFeatureBuilder(db_path=self.service_db)
        self.log = DecisionLog(self.service_db)

    def decide(self, live: LlmServerConfig, ticker: str, as_of: pd.Timestamp, state: dict, news: list[str]) -> JudgeResult:
        per_perm, masses = [], []
        for perm in PERMS:
            p, mass = option_probs(live, self.prompt.messages(state, news, perm), ("A", "B", "C"))
            per_perm.append({o: p[L] for L, o in zip("ABC", perm)})
            masses.append(mass)
        probs = {o: sum(d[o] for d in per_perm) / len(per_perm) for o in OPTIONS}
        decision = max(probs, key=probs.get)
        spread = max(max(d[o] for d in per_perm) - min(d[o] for d in per_perm) for o in OPTIONS)
        letter = "ABC"[PERMS[0].index(decision)]
        msgs = self.prompt.messages(state, news, PERMS[0]) + [
            {"role": "assistant", "content": letter},
            {"role": "user", "content": f"판단 {letter}({NAMES[decision]})는 고정이다. 근거를 스키마에 맞춰 한국어로 써라. "
                                        "위험 수준은 1=매우 낮음, 2=낮음, 3=보통, 4=높음, 5=매우 높음이다. 판단을 바꾸지 마라."},
        ]
        d = post_json(live, "/v1/chat/completions", {"messages": msgs, "max_tokens": 400, "temperature": 0,
                      "response_format": {"type": "json_schema", "json_schema": {"name": "explain", "schema": EXPLAIN_SCHEMA}}})
        try:
            explain = json.loads(d["choices"][0]["message"]["content"])
            risk = explain["risk_level"]
        except (KeyError, IndexError, TypeError, ValueError) as exc:  # 스키마를 강제해도 잘린 응답 등은 가능하다
            raise LlmError(f"근거 응답을 읽을 수 없다: {exc}") from exc
        mismatch = (decision == "reduce" and risk <= 2) or (decision == "keep" and risk >= 4)
        return JudgeResult(ticker, as_of, decision, probs, per_perm, spread, min(masses), explain, bool(mismatch), state, news)

    def _latest(self) -> tuple[dict[str, dict], pd.DataFrame]:
        with duckdb.connect(self.service_db, read_only=True) as con:
            snaps = con.execute("SELECT ticker, payload FROM indicator_snapshots "
                                "WHERE as_of = (SELECT MAX(as_of) FROM indicator_snapshots)").fetchall()
            bounds = con.execute("SELECT * FROM regime_boundaries").df()
        if not snaps:
            raise DataIntegrityError("지표 묶음이 없다. 지표 단계를 먼저 돌린다")
        return {tk: json.loads(p) for tk, p in snaps}, bounds

    def _judge_one(self, live: LlmServerConfig, snaps: dict[str, dict], bounds: pd.DataFrame, tk: str) -> JudgeResult:
        try:
            snap = snaps[tk]
        except KeyError as exc:
            raise DataIntegrityError(f"{tk}의 최신 지표 묶음이 없다") from exc
        as_of = pd.Timestamp(snap["as_of"])
        news = self.news.load_news_as_of(as_of.to_pydatetime(), ticker=tk, lookback_hours=self.news_lookback_hours)
        return self.decide(live, tk, as_of, self.prompt.state(snap, bounds), self.prompt.news_lines(news, tk))

    def run(self, tickers: list[str] | None = None, log_path: Path | None = None) -> BatchResult[str, JudgeResult]:
        """종목 하나의 실패(응답 이상·지표 없음)는 그 종목만 실패로 남기고 계속한다. 서버를 못 띄우면 예외로 끝난다."""
        snaps, bounds = self._latest()
        model_hash = file_sha256(self.cfg.model_path)
        with llm_server(self.cfg, log_path=log_path) as live:
            got = each(tickers or sorted(snaps), lambda tk: self._judge_one(live, snaps, bounds, tk))
        self.log.save([DecisionLog.row(r, self.cfg.model_path.name, model_hash) for r in got.ok.values()])
        return got


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()
