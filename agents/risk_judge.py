"""4단계 판단: 보조 지표 스냅숏 + 판단 시각 이전 뉴스 → 유지/관망/축소 확률과 근거(로컬 LLM).

전처리: 스냅숏 JSON(사전 국면, GARCH-t 예측 RV와 그 값이 학습 구간 직전 RV 분포의 몇 분위 칸인지, 가격 정지 비율)과
`load_news_as_of`(max(발행, 처음 수집) <= as_of)로 고른 종목·시장 기사 제목을 지시문에 넣는다. 미래 수익률 같은
방향 정보와 유사 흐름 마트는 넣지 않는다(마트 연결은 연구 검증 전 보류).
판단: 지시문 v1(docs/service_design.md §10-1)의 보기를 A/B/C 한 글자로 두고, 보기 배치 3가지(순환)의 다음 1토큰
확률 평균으로 정한다(잔잔한 국면의 위치 편향 대응). 설명: 판단을 고정한 채 JSON 스키마로 근거를 받는다.
후처리: 판단과 위험 수준이 어긋나면(축소인데 위험 1~2, 유지인데 위험 4~5) `mismatch`로 표시하고,
입력·출력·모델 파일 해시를 `decision_log`에 남긴다(페이퍼 기록, 주문 없음).
확률은 아직 보정하지 않았다(라벨 정의 §11 확정 전).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import duckdb
import pandas as pd

from agents.llm_runtime import LlmServerConfig, option_probs, post_json
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
SYSTEM_V1 = (
    "너는 암호화폐 포지션의 위험 관리자다. 방향(오를지 내릴지)은 예측하지 않는다. 주어진 국면·변동성 지표·뉴스만 근거로, "
    "아래 기준에 따라 세 가지 중 하나를 고른다. 위험 신호가 없으면 유지가 기본이다.\n"
    "- 유지: 위험 신호가 평소 수준이다. 직전 변동성 국면이 Q1~Q3이고, 변동성 예측이 평소 범위이며, 거래소·보안·규제 같은 직접 악재 뉴스가 없다.\n"
    "- 관망: 위험 신호가 엇갈리거나 한 가지만 나쁘다, 또는 판단할 정보가 부족하다.\n"
    "- 축소: 위험 신호가 둘 이상 함께 나쁘다. 예: 요동 국면(Q4~Q5)과 변동성 예측 상승, 직접 악재 뉴스(해킹·입출금 중단·상장 폐지 등), "
    "가격이 멈춘 봉 증가(유동성 저하).\n"
    "국면 Q1~Q5는 그 종목의 과거(학습 구간) 직전 변동성 5분위다. Q1이 가장 잔잔하고 Q5가 가장 요동친다."
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
PROMPT_VERSION = "v1_20261018"
LOG_TABLE = "decision_log"


def _level(value: float | None, row: pd.Series | None) -> str | None:
    """예측 RV를 같은 구간의 학습 구간 직전 RV 5분위 칸(Q1~Q5)으로 읽는다."""
    if value is None or row is None:
        return None
    edges = [row["q20"], row["q40"], row["q60"], row["q80"]]
    return f"Q{sum(value >= e for e in edges) + 1}"


def build_state(snap: dict, bounds: pd.DataFrame) -> dict:
    tb = bounds[bounds["ticker"] == snap["ticker"]].set_index("horizon_min")
    rv = snap["garch_t"]["rv_forecast"]
    return {
        "ticker": snap["ticker"],
        "as_of_kst": snap["as_of"],
        "prior_regime": {k: v["regime"] for k, v in snap["regime"]["values"].items()},
        "garch_t_rv_forecast": {k: round(v, 5) for k, v in rv.items()},
        "garch_t_rv_forecast_level": {k: _level(v, tb.loc[int(k[:-1])] if int(k[:-1]) in tb.index else None) for k, v in rv.items()},
        "price_stall_1d": snap["price_stall"]["1d"],
        "minutes_since_last_trade": int((pd.Timestamp(snap["data_end_bar"]) - pd.Timestamp(snap["last_trade_bar"])).total_seconds() // 60),
    }


def news_lines(news: pd.DataFrame, ticker: str, k_ticker: int = 8, k_market: int = 5) -> list[str]:
    out = []
    for tag, k in ((ticker, k_ticker), ("MARKET", k_market)):
        sub = news[news["ticker_hint"] == tag].head(k)
        for _, row in sub.iterrows():
            label = "종목" if tag == ticker else "시장"
            out.append(f"[{label} {pd.Timestamp(row['effective_at']):%m-%d %H:%M}] {row['title']}")
    return out


def _messages(state: dict, news: list[str], perm: tuple[str, ...]) -> list[dict]:
    opts = ", ".join(f"{L}={DESC[o]}" for L, o in zip("ABC", perm))
    user = ("상태(JSON):\n" + json.dumps(state, ensure_ascii=False) + "\n최근 24시간 뉴스(판단 시각 이전만):\n"
            + ("\n".join(f"- {n}" for n in news) if news else "- 해당 없음(수집된 기사 없음)")
            + f"\n\n질문: 현재 포지션을 어떻게 할 것인가? 보기 {opts}. 보기 글자 하나로만 답하라.")
    return [{"role": "system", "content": SYSTEM_V1}, {"role": "user", "content": user}]


def judge(cfg: LlmServerConfig, state: dict, news: list[str]) -> dict:
    per_perm, masses = [], []
    for perm in PERMS:
        p, mass = option_probs(cfg, _messages(state, news, perm), ("A", "B", "C"))
        per_perm.append({o: p[L] for L, o in zip("ABC", perm)})
        masses.append(mass)
    probs = {o: sum(d[o] for d in per_perm) / len(per_perm) for o in OPTIONS}
    decision = max(probs, key=probs.get)
    spread = max(max(d[o] for d in per_perm) - min(d[o] for d in per_perm) for o in OPTIONS)
    letter = "ABC"[PERMS[0].index(decision)]
    msgs = _messages(state, news, PERMS[0]) + [
        {"role": "assistant", "content": letter},
        {"role": "user", "content": f"판단 {letter}({NAMES[decision]})는 고정이다. 근거를 스키마에 맞춰 한국어로 써라. "
                                    "위험 수준은 1=매우 낮음, 2=낮음, 3=보통, 4=높음, 5=매우 높음이다. 판단을 바꾸지 마라."},
    ]
    d = post_json(cfg, "/v1/chat/completions", {"messages": msgs, "max_tokens": 400, "temperature": 0,
                  "response_format": {"type": "json_schema", "json_schema": {"name": "explain", "schema": EXPLAIN_SCHEMA}}})
    explain = json.loads(d["choices"][0]["message"]["content"])
    risk = explain["risk_level"]
    mismatch = (decision == "reduce" and risk <= 2) or (decision == "keep" and risk >= 4)
    return {"decision": decision, "probs": probs, "probs_by_perm": per_perm, "position_spread": spread,
            "option_mass_min": min(masses), "explain": explain, "mismatch": bool(mismatch)}


def file_sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def log_decisions(rows: list[dict], service_db: str) -> None:
    df = pd.DataFrame(rows)
    with duckdb.connect(service_db) as con:
        con.register("d", df)
        con.execute(f"CREATE TABLE IF NOT EXISTS {LOG_TABLE} AS SELECT * FROM d WHERE FALSE")
        con.execute(f"INSERT INTO {LOG_TABLE} SELECT * FROM d")
        con.unregister("d")


def run(tickers: list[str] | None = None, service_db: str | None = None, log_path: Path | None = None) -> list[dict]:
    """최신 스냅숏의 종목들을 판단한다. 서버는 이 함수 안에서만 떠 있다(호출 때만 적재)."""
    from agents.llm_runtime import llm_server

    service_db = resolve_service_db_path(service_db)
    with duckdb.connect(service_db, read_only=True) as con:
        snaps = con.execute("SELECT ticker, payload FROM indicator_snapshots WHERE as_of = (SELECT MAX(as_of) FROM indicator_snapshots)").fetchall()
        bounds = con.execute("SELECT * FROM regime_boundaries").df()
    snaps = {tk: json.loads(p) for tk, p in snaps}
    tickers = tickers or sorted(snaps)
    builder = TextFeatureBuilder(db_path=service_db)
    cfg = LlmServerConfig()
    model_hash = file_sha256(cfg.model_path)
    rows = []
    with llm_server(cfg, log_path=log_path) as live:
        for tk in tickers:
            snap = snaps[tk]
            state = build_state(snap, bounds)
            news = builder.load_news_as_of(pd.Timestamp(snap["as_of"]).to_pydatetime(), ticker=tk, lookback_hours=24)
            lines = news_lines(news, tk)
            out = judge(live, state, lines)
            rows.append({
                "logged_at": datetime.now(), "as_of": pd.Timestamp(snap["as_of"]), "ticker": tk,
                "decision": out["decision"], "p_keep": out["probs"]["keep"], "p_watch": out["probs"]["watch"],
                "p_reduce": out["probs"]["reduce"], "position_spread": out["position_spread"],
                "risk_level": out["explain"]["risk_level"], "uncertainty": out["explain"]["uncertainty"],
                "mismatch": out["mismatch"], "rationale": out["explain"]["rationale"],
                "input_json": json.dumps({"state": state, "news": lines}, ensure_ascii=False),
                "output_json": json.dumps(out, ensure_ascii=False),
                "prompt_version": PROMPT_VERSION, "model_file": cfg.model_path.name, "model_sha256": model_hash,
                "calibrated": False,
            })
    log_decisions(rows, service_db)
    return rows
