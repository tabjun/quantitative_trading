"""
Realtime text-context ingestion and feature engineering for market analysis.

The module intentionally uses only the standard library plus pandas/duckdb so it
can run in the current uv environment without extra setup. API-backed sources
are optional; RSS and local CSV inputs work out of the box.

시각 규약(2026-10-17, S188 지적 4): 서비스 DB의 텍스트 시각은 시세(`upbit_krw_candle`, pyupbit
`candle_date_time_kst`)와 같은 **KST, 시간대 표시 없음**이다. 기사는 `published_at`(발행, KST)과
`available_at`(우리가 처음 수집한 시각, KST)을 함께 저장하고, 실제로 쓸 수 있는 시각
`effective_at = max(published_at, available_at)`이 속한 15분 봉 라벨에 넣는다. 라벨 T의 특징은
봉이 닫히는 T+15분에 알려진 정보다. 판단 시각 t에 쓸 기사는 `effective_at <= t`만 고른다
(`TextFeatureBuilder.load_news_as_of`). 이전 구현은 UTC 시각에서 시간대만 떼어 KST 봉에 붙여
기사가 실제 공개보다 9시간 앞선 봉에 들어갔다.

    python -m contexts.text_context --selftest   # 합성 시각 시험(서비스 DB에 쓰지 않음)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Sequence
from zoneinfo import ZoneInfo

import duckdb
import pandas as pd
from database.paths import resolve_db_path, resolve_service_db_path


KST = ZoneInfo("Asia/Seoul")
MARKET_TAG = "MARKET"  # 특정 종목이 아닌 시장 전체 뉴스의 ticker_hint
DEFAULT_LOOKBACK_DAYS = 7
UPBIT_MARKETS_URL = "https://api.upbit.com/v1/market/all?isDetails=false"

DEFAULT_RSS_QUERIES = [
    "bitcoin OR BTC crypto market",
    "upbit bitcoin korea crypto",
    "Federal Reserve interest rate crypto",
]

POSITIVE_TERMS = {
    "상승",
    "반등",
    "호재",
    "강세",
    "돌파",
    "매수",
    "회복",
    "완화",
    "유입",
    "승인",
    "랠리",
    "bull",
    "bullish",
    "rally",
    "surge",
    "gain",
    "rebound",
    "breakout",
    "inflow",
    "approve",
    "approval",
    "easing",
    "beat",
}

NEGATIVE_TERMS = {
    "하락",
    "급락",
    "악재",
    "약세",
    "매도",
    "공포",
    "규제",
    "긴축",
    "유출",
    "청산",
    "해킹",
    "소송",
    "bear",
    "bearish",
    "drop",
    "plunge",
    "crash",
    "selloff",
    "outflow",
    "hack",
    "lawsuit",
    "tightening",
    "fear",
    "risk",
}

TOPIC_TERMS = {
    "macro": {"fed", "fomc", "rate", "cpi", "inflation", "금리", "연준", "물가", "인플레이션"},
    "risk": {"war", "hack", "lawsuit", "risk", "crash", "전쟁", "해킹", "소송", "리스크", "급락"},
    "crypto": {"btc", "bitcoin", "ethereum", "eth", "upbit", "crypto", "비트코인", "이더리움", "업비트"},
    "regulation": {"sec", "etf", "regulation", "ban", "approval", "규제", "승인", "금지", "감독"},
    "liquidity": {"volume", "inflow", "outflow", "liquidity", "거래량", "유입", "유출", "유동성"},
}


@dataclass(frozen=True)
class TextRecord:
    source_type: str
    source_name: str
    published_at: datetime  # 시간대 포함(UTC)
    title: str
    body: str
    url: str
    ticker_hint: str = MARKET_TAG
    available_at: datetime | None = None  # 처음 수집한 시각(시간대 포함). None이면 저장 시각을 쓴다

    @property
    def record_id(self) -> str:
        # available_at은 넣지 않는다: 같은 기사를 다시 수집해도 같은 id가 되어 처음 수집 시각을 보존한다.
        raw = "|".join([self.source_type, self.source_name, self.url, self.title, self.published_at.isoformat()])
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def clean_text(value: str | None) -> str:
    text = html.unescape(value or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            parsed = datetime.now(timezone.utc)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def to_kst_naive(value: datetime) -> datetime:
    """시간대 포함 시각 → KST 시간대 표시 없음(시세와 같은 규약). 시간대가 없으면 UTC로 본다."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(KST).replace(tzinfo=None)


def now_kst_naive() -> datetime:
    return datetime.now(KST).replace(tzinfo=None)


def last_closed_bar(now: datetime | None = None) -> pd.Timestamp:
    """지금(KST) 기준 마지막으로 닫힌 15분 봉의 라벨."""
    return pd.Timestamp(now or now_kst_naive()).floor("15min") - pd.Timedelta(minutes=15)


def google_news_rss_url(query: str, lookback_days: int) -> str:
    encoded = urllib.parse.quote_plus(f"{query} when:{int(lookback_days)}d")
    return f"https://news.google.com/rss/search?q={encoded}&hl=ko&gl=KR&ceid=KR:ko"


def default_rss_urls(lookback_days: int = DEFAULT_LOOKBACK_DAYS) -> list[str]:
    return [google_news_rss_url(query, lookback_days) for query in DEFAULT_RSS_QUERIES]


def fetch_upbit_korean_names(timeout: int = 10) -> dict[str, str]:
    """현재 상장된 KRW 마켓의 {마켓: 한국어 이름}. 목록에 없는 종목은 상장 폐지로 본다."""
    with urllib.request.urlopen(UPBIT_MARKETS_URL, timeout=timeout) as response:
        rows = json.loads(response.read())
    return {row["market"]: row["korean_name"] for row in rows if row["market"].startswith("KRW-")}


def ticker_aliases(ticker: str, korean_name: str) -> tuple[list[str], str]:
    """(한국어 이름들, 심볼). "엑스알피(리플)"은 ["엑스알피", "리플"]."""
    return [n.strip() for n in re.split(r"[()]", korean_name) if n.strip()], ticker.split("-", 1)[1]


def ticker_news_query(ticker: str, korean_name: str) -> str:
    """종목별 검색어. 일반 단어와 겹치는 이름(아더·아크·세이·수이 등)이 있어 "코인"을 함께 건다."""
    names, symbol = ticker_aliases(ticker, korean_name)
    terms = " OR ".join([f'"{n}"' for n in names] + [symbol])
    return f"({terms}) 코인"


def title_mentions(title: str, ticker: str, korean_name: str) -> bool:
    """제목에 종목의 한국어 이름(부분 일치)이나 심볼(영문 단어 경계)이 있는가. 검색 결과에 섞이는 무관 기사를 거른다.
    한국어는 조사가 붙어 단어 경계를 쓸 수 없으므로 "세이펜"처럼 이름을 포함한 다른 단어는 남는다(한계)."""
    names, symbol = ticker_aliases(ticker, korean_name)
    if any(n in title for n in names):
        return True
    return re.search(rf"(?<![A-Za-z]){re.escape(symbol)}(?![A-Za-z])", title, flags=re.IGNORECASE) is not None


def env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [part.strip() for part in raw.split("|") if part.strip()]


class TextDataCollector:
    """Collects realtime news/report/SNS text into normalized records.

    ticker_names({마켓: 한국어 이름})가 있으면 종목마다 검색어 하나로 구글 뉴스 RSS를 받고, 제목에 그 종목 이름·심볼이
    있는 기사만 ticker_hint를 그 종목으로 달아 남긴다. 일반 검색어(rss_urls)는 MARKET 태그다.
    발행 시각이 lookback_days보다 오래된 기사는 버린다.
    요청은 순차로 보낸다(공유 서버, 병렬 워커 금지).
    """

    def __init__(
        self,
        rss_urls: Sequence[str] | None = None,
        local_csv_paths: Sequence[str] | None = None,
        ticker_names: dict[str, str] | None = None,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        request_pause_s: float = 1.0,
    ):
        self.lookback_days = lookback_days
        self.rss_urls = list(rss_urls or env_list("TEXT_RSS_URLS") or default_rss_urls(lookback_days))
        self.local_csv_paths = list(local_csv_paths or env_list("TEXT_LOCAL_CSVS"))
        self.ticker_names = dict(ticker_names or {})
        self.dropped_off_topic: dict[str, int] = {}
        self.request_pause_s = request_pause_s

    def collect_all(self, max_items_per_source: int = 30) -> list[TextRecord]:
        records: list[TextRecord] = []
        records.extend(self.collect_rss(max_items_per_source=max_items_per_source))
        records.extend(self.collect_ticker_rss(max_items_per_source=max_items_per_source))
        records.extend(self.collect_naver_news(max_items=max_items_per_source))
        records.extend(self.collect_local_csv())
        return self._recent_only(deduplicate_records(records))

    def _recent_only(self, records: list[TextRecord]) -> list[TextRecord]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=self.lookback_days)
        return [record for record in records if record.published_at >= cutoff]

    def _fetch_rss(self, url: str, ticker_hint: str, max_items: int) -> list[TextRecord]:
        try:
            with urllib.request.urlopen(url, timeout=12) as response:
                payload = response.read()
            fetched_at = datetime.now(timezone.utc)
            root = ET.fromstring(payload)
        except Exception as exc:
            print(f"[WARN] RSS fetch failed: {url} ({exc})")
            return []
        channel = root.find("channel")
        source_name = clean_text(channel.findtext("title") if channel is not None else url)
        records = []
        for item in root.findall(".//item")[:max_items]:
            records.append(
                TextRecord(
                    source_type="news_rss",
                    source_name=source_name or "rss",
                    published_at=parse_datetime(item.findtext("pubDate")),
                    title=clean_text(item.findtext("title")),
                    body=clean_text(item.findtext("description")),
                    url=clean_text(item.findtext("link")),
                    ticker_hint=ticker_hint,
                    available_at=fetched_at,
                )
            )
        return records

    def collect_rss(self, max_items_per_source: int = 30) -> list[TextRecord]:
        records: list[TextRecord] = []
        for url in self.rss_urls:
            records.extend(self._fetch_rss(url, MARKET_TAG, max_items_per_source))
            time.sleep(self.request_pause_s)
        return records

    def collect_ticker_rss(self, max_items_per_source: int = 30) -> list[TextRecord]:
        records: list[TextRecord] = []
        for ticker, name in self.ticker_names.items():
            url = google_news_rss_url(ticker_news_query(ticker, name), self.lookback_days)
            fetched = self._fetch_rss(url, ticker, max_items_per_source)
            kept = [r for r in fetched if title_mentions(r.title, ticker, name)]
            self.dropped_off_topic[ticker] = len(fetched) - len(kept)
            records.extend(kept)
            time.sleep(self.request_pause_s)
        return records

    def collect_naver_news(self, query: str = "비트코인 업비트 증시", max_items: int = 30) -> list[TextRecord]:
        client_id = os.getenv("NAVER_CLIENT_ID")
        client_secret = os.getenv("NAVER_CLIENT_SECRET")
        if not client_id or not client_secret:
            return []

        params = urllib.parse.urlencode({"query": query, "display": min(max_items, 100), "sort": "date"})
        request = urllib.request.Request(f"https://openapi.naver.com/v1/search/news.xml?{params}")
        request.add_header("X-Naver-Client-Id", client_id)
        request.add_header("X-Naver-Client-Secret", client_secret)

        try:
            with urllib.request.urlopen(request, timeout=12) as response:
                payload = response.read()
            fetched_at = datetime.now(timezone.utc)
            root = ET.fromstring(payload)
        except Exception as exc:
            print(f"[WARN] Naver news fetch failed: {exc}")
            return []

        records = []
        for item in root.findall(".//item")[:max_items]:
            records.append(
                TextRecord(
                    source_type="news_api",
                    source_name="Naver News Search",
                    published_at=parse_datetime(item.findtext("pubDate")),
                    title=clean_text(item.findtext("title")),
                    body=clean_text(item.findtext("description")),
                    url=clean_text(item.findtext("originallink") or item.findtext("link")),
                    available_at=fetched_at,
                )
            )
        return records

    def collect_local_csv(self) -> list[TextRecord]:
        records: list[TextRecord] = []
        for raw_path in self.local_csv_paths:
            path = Path(raw_path)
            if not path.exists():
                print(f"[WARN] Local text CSV not found: {path}")
                continue
            loaded_at = datetime.now(timezone.utc)
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    records.append(
                        TextRecord(
                            source_type=row.get("source_type") or "local_text",
                            source_name=row.get("source_name") or path.stem,
                            published_at=parse_datetime(row.get("published_at") or row.get("timestamp")),
                            title=clean_text(row.get("title")),
                            body=clean_text(row.get("body") or row.get("text")),
                            url=clean_text(row.get("url")) or f"local://{path.name}",
                            ticker_hint=row.get("ticker_hint") or MARKET_TAG,
                            available_at=parse_datetime(row["available_at"]) if row.get("available_at") else loaded_at,
                        )
                    )
        return records


def deduplicate_records(records: Iterable[TextRecord]) -> list[TextRecord]:
    by_id: dict[str, TextRecord] = {}
    for record in records:
        if record.title:
            by_id[record.record_id] = record
    return list(by_id.values())


RAW_COLUMNS = [
    "record_id", "source_type", "source_name", "published_at", "available_at", "title", "body", "url",
    "ticker_hint", "sentiment_score", "positive_hits", "negative_hits", "topic_macro", "topic_risk",
    "topic_crypto", "topic_regulation", "topic_liquidity", "inserted_at",
]
FEATURE_COLUMNS = [
    "text_event_count", "text_sentiment_mean", "text_sentiment_sum", "text_sentiment_abs_mean",
    "text_positive_hits", "text_negative_hits", "text_macro_count", "text_risk_count", "text_crypto_count",
    "text_regulation_count", "text_liquidity_count", "text_news_count", "text_report_count", "text_sns_count",
    "text_shock_z", "text_sentiment_momentum_1h",
]


def _columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    return [row[0] for row in con.execute(
        "SELECT column_name FROM information_schema.columns WHERE table_name = ? ORDER BY ordinal_position", [table]
    ).fetchall()]


def ensure_text_schema(con: duckdb.DuckDBPyConnection) -> None:
    """text_events_raw를 KST 규약 스키마로 만들고, 예전 스키마(UTC, available_at 없음)면 옮긴다."""
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS text_events_raw (
            record_id VARCHAR, source_type VARCHAR, source_name VARCHAR,
            published_at TIMESTAMP, available_at TIMESTAMP,
            title VARCHAR, body VARCHAR, url VARCHAR, ticker_hint VARCHAR,
            sentiment_score DOUBLE, positive_hits INTEGER, negative_hits INTEGER,
            topic_macro INTEGER, topic_risk INTEGER, topic_crypto INTEGER, topic_regulation INTEGER,
            topic_liquidity INTEGER, inserted_at TIMESTAMP
        )
        """
    )
    if "available_at" not in _columns(con, "text_events_raw"):
        # 2026-10-15 이전 행: published_at·inserted_at이 UTC(시간대 없음). KST로 옮기고, 처음 수집 시각은 inserted_at.
        # 예전 기본 태그 KRW-BTC는 종목 검색이 아니라 시장 검색어에 붙은 값이라 MARKET으로 바꾼다.
        con.execute("ALTER TABLE text_events_raw ADD COLUMN available_at TIMESTAMP")
        con.execute(
            "UPDATE text_events_raw SET published_at = published_at + INTERVAL 9 HOUR, "
            "inserted_at = inserted_at + INTERVAL 9 HOUR, available_at = inserted_at + INTERVAL 9 HOUR, "
            f"ticker_hint = '{MARKET_TAG}'"
        )


class TextFeatureBuilder:
    """Builds 15-minute text factors aligned to the existing candle mart(KST)."""

    def __init__(self, db_path: str = "data/llm_agent.db"):
        self.db_path = resolve_service_db_path(db_path)

    def persist_raw_records(self, records: Sequence[TextRecord]) -> int:
        with duckdb.connect(self.db_path) as con:
            ensure_text_schema(con)
            if not records:
                return 0
            rows = pd.DataFrame([self._record_to_row(record) for record in records])[RAW_COLUMNS]
            con.register("rows_df", rows)
            # 다시 수집한 기사는 처음 수집 시각(available_at)을 유지한다.
            prev = con.execute(
                "SELECT r.record_id, r.available_at FROM text_events_raw r JOIN rows_df n USING (record_id)"
            ).df()
            if len(prev):
                first_seen = dict(zip(prev["record_id"], pd.to_datetime(prev["available_at"])))
                rows["available_at"] = [
                    min(a, first_seen[i]) if i in first_seen and pd.notna(first_seen[i]) else a
                    for i, a in zip(rows["record_id"], pd.to_datetime(rows["available_at"]))
                ]
                con.unregister("rows_df")
                con.register("rows_df", rows)
            con.execute("DELETE FROM text_events_raw WHERE record_id IN (SELECT record_id FROM rows_df)")
            cols = ", ".join(RAW_COLUMNS)
            con.execute(f"INSERT INTO text_events_raw ({cols}) SELECT {cols} FROM rows_df")
            con.unregister("rows_df")
        return len(records)

    def build_and_persist_15m_features(
        self, price_index: pd.DataFrame | None = None, lookback_days: int = DEFAULT_LOOKBACK_DAYS
    ) -> pd.DataFrame:
        raw_df = self.load_raw_records()
        features = self.build_15m_features(raw_df, price_index=price_index, lookback_days=lookback_days)
        with duckdb.connect(self.db_path) as con:
            if "ticker" not in _columns(con, "text_features_15m"):
                con.execute("DROP TABLE IF EXISTS text_features_15m")  # 파생 테이블이라 새 스키마로 다시 만든다
            con.register("features_df", features)
            con.execute("CREATE TABLE IF NOT EXISTS text_features_15m AS SELECT * FROM features_df WHERE FALSE")
            con.execute("DELETE FROM text_features_15m")
            con.execute("INSERT INTO text_features_15m SELECT * FROM features_df")
            con.unregister("features_df")
        return features

    def load_raw_records(self) -> pd.DataFrame:
        if not Path(self.db_path).exists():
            return pd.DataFrame()
        with duckdb.connect(self.db_path, read_only=True) as con:
            cols = _columns(con, "text_events_raw")
            if not cols:
                return pd.DataFrame()
            if "available_at" not in cols:
                raise RuntimeError("text_events_raw가 예전 UTC 스키마다. persist_raw_records를 한 번 실행해 옮긴 뒤 읽는다.")
            return con.execute("SELECT * FROM text_events_raw").df()

    def load_news_as_of(
        self, as_of: datetime, ticker: str | None = None, lookback_hours: int = 24, include_market: bool = True
    ) -> pd.DataFrame:
        """판단 시각 as_of(KST)에 실제로 알 수 있었던 기사만: effective_at <= as_of, 최근 lookback_hours."""
        raw = self.load_raw_records()
        if raw.empty:
            return raw
        raw = raw.assign(effective_at=effective_times(raw))
        as_of = pd.Timestamp(as_of)
        keep = (raw["effective_at"] <= as_of) & (raw["effective_at"] > as_of - pd.Timedelta(hours=lookback_hours))
        if ticker is not None:
            tags = [ticker, MARKET_TAG] if include_market else [ticker]
            keep &= raw["ticker_hint"].isin(tags)
        return raw[keep].sort_values("effective_at", ascending=False).reset_index(drop=True)

    def build_15m_features(
        self,
        raw_df: pd.DataFrame,
        price_index: pd.DataFrame | None = None,
        lookback_days: int = DEFAULT_LOOKBACK_DAYS,
        now: datetime | None = None,
    ) -> pd.DataFrame:
        """(ticker, 15분 라벨)별 텍스트 특징. 라벨 T = effective_at이 속한 봉, T+15분에 알려진 정보.

        격자: price_index가 있으면 그 시각들, 없으면 실시간 격자(마지막으로 닫힌 봉까지 lookback_days).
        """
        if price_index is not None and not price_index.empty:
            grid = pd.DatetimeIndex(pd.to_datetime(price_index["timestamp"]).dt.floor("15min").drop_duplicates()).sort_values()
        else:
            end = last_closed_bar(now)
            grid = pd.date_range(end - pd.Timedelta(days=lookback_days), end, freq="15min")
        tickers = sorted(set(raw_df["ticker_hint"].dropna())) if not raw_df.empty else [MARKET_TAG]
        if MARKET_TAG not in tickers:
            tickers.append(MARKET_TAG)
        base = pd.MultiIndex.from_product([tickers, grid], names=["ticker", "timestamp"]).to_frame(index=False)

        if raw_df.empty:
            grouped = base
        else:
            df = raw_df.copy()
            df["timestamp"] = effective_times(df).dt.floor("15min")
            df["ticker"] = df["ticker_hint"].fillna(MARKET_TAG)
            df["is_news"] = df["source_type"].str.contains("news|rss", case=False, na=False).astype(int)
            df["is_report"] = df["source_type"].str.contains("report", case=False, na=False).astype(int)
            df["is_sns"] = df["source_type"].str.contains("sns|reddit|x|twitter|stocktwits", case=False, na=False).astype(int)
            agg = (
                df.groupby(["ticker", "timestamp"])
                .agg(
                    text_event_count=("record_id", "count"),
                    text_sentiment_mean=("sentiment_score", "mean"),
                    text_sentiment_sum=("sentiment_score", "sum"),
                    text_sentiment_abs_mean=("sentiment_score", lambda x: x.abs().mean()),
                    text_positive_hits=("positive_hits", "sum"),
                    text_negative_hits=("negative_hits", "sum"),
                    text_macro_count=("topic_macro", "sum"),
                    text_risk_count=("topic_risk", "sum"),
                    text_crypto_count=("topic_crypto", "sum"),
                    text_regulation_count=("topic_regulation", "sum"),
                    text_liquidity_count=("topic_liquidity", "sum"),
                    text_news_count=("is_news", "sum"),
                    text_report_count=("is_report", "sum"),
                    text_sns_count=("is_sns", "sum"),
                )
                .reset_index()
            )
            grouped = base.merge(agg, on=["ticker", "timestamp"], how="left")

        for col in FEATURE_COLUMNS:
            if col not in grouped.columns:
                grouped[col] = 0.0
        grouped = grouped.sort_values(["ticker", "timestamp"]).reset_index(drop=True)
        raw_cols = [c for c in FEATURE_COLUMNS if c not in ("text_shock_z", "text_sentiment_momentum_1h")]
        grouped[raw_cols] = grouped[raw_cols].fillna(0.0).infer_objects()
        by_ticker = grouped.groupby("ticker")["text_event_count"]
        count_std = by_ticker.transform(lambda s: s.rolling(96, min_periods=4).std())
        count_mean = by_ticker.transform(lambda s: s.rolling(96, min_periods=4).mean())
        shock = (grouped["text_event_count"] - count_mean) / count_std
        grouped["text_shock_z"] = shock.where(count_std.ne(0), 0.0).fillna(0.0).astype(float)
        grouped["text_sentiment_momentum_1h"] = grouped.groupby("ticker")["text_sentiment_mean"].transform(
            lambda s: s.rolling(4, min_periods=1).mean()
        )
        grouped["updated_at"] = now_kst_naive()
        return grouped[["ticker", "timestamp", *FEATURE_COLUMNS, "updated_at"]]

    def enrich_price_frame(self, price_df: pd.DataFrame, ticker: str = MARKET_TAG) -> pd.DataFrame:
        raw_df = self.load_raw_records()
        features = self.build_15m_features(raw_df, price_index=price_df[["timestamp"]])
        features = features[features["ticker"] == ticker].drop(columns=["ticker", "updated_at"])
        merged = price_df.copy()
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])
        features["timestamp"] = pd.to_datetime(features["timestamp"])
        return merged.merge(features, on="timestamp", how="left").fillna(0.0)

    def _record_to_row(self, record: TextRecord) -> dict[str, object]:
        text = f"{record.title} {record.body}".lower()
        positive_hits = sum(1 for term in POSITIVE_TERMS if term.lower() in text)
        negative_hits = sum(1 for term in NEGATIVE_TERMS if term.lower() in text)
        sentiment_score = (positive_hits - negative_hits) / max(positive_hits + negative_hits, 1)
        topics = {
            topic: int(any(term.lower() in text for term in terms))
            for topic, terms in TOPIC_TERMS.items()
        }
        inserted_at = now_kst_naive()
        return {
            "record_id": record.record_id,
            "source_type": record.source_type,
            "source_name": record.source_name,
            "published_at": to_kst_naive(record.published_at),
            "available_at": to_kst_naive(record.available_at) if record.available_at else inserted_at,
            "title": record.title,
            "body": record.body,
            "url": record.url,
            "ticker_hint": record.ticker_hint,
            "sentiment_score": float(sentiment_score),
            "positive_hits": int(positive_hits),
            "negative_hits": int(negative_hits),
            "topic_macro": topics["macro"],
            "topic_risk": topics["risk"],
            "topic_crypto": topics["crypto"],
            "topic_regulation": topics["regulation"],
            "topic_liquidity": topics["liquidity"],
            "inserted_at": inserted_at,
        }


def effective_times(raw_df: pd.DataFrame) -> pd.Series:
    """기사를 실제로 쓸 수 있는 시각(KST) = max(발행, 처음 수집)."""
    published = pd.to_datetime(raw_df["published_at"], errors="coerce")
    available = pd.to_datetime(raw_df["available_at"], errors="coerce").fillna(published)
    return pd.concat([published, available], axis=1).max(axis=1)


def ingest_realtime_text_context(
    db_path: str = "data/llm_agent.db",
    max_items_per_source: int = 30,
    ticker_names: dict[str, str] | None = None,
    lookback_days: int = DEFAULT_LOOKBACK_DAYS,
) -> tuple[int, pd.DataFrame]:
    collector = TextDataCollector(ticker_names=ticker_names, lookback_days=lookback_days)
    records = collector.collect_all(max_items_per_source=max_items_per_source)
    builder = TextFeatureBuilder(db_path=db_path)
    inserted = builder.persist_raw_records(records)
    features = builder.build_and_persist_15m_features(lookback_days=lookback_days)
    return inserted, features


def load_price_with_text_context(
    db_path: str = "data/upbit_data.db",
    price_table: str = "btc_15m_advance",
    limit: int | None = None,
    text_db_path: str = "data/llm_agent.db",
) -> pd.DataFrame:
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    with duckdb.connect(resolve_db_path(db_path), read_only=True) as con:
        price_df = con.execute(
            f"""
            SELECT timestamp, open, high, low, close, volume, value
            FROM {price_table}
            ORDER BY timestamp DESC
            {limit_sql}
            """
        ).df()
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    return TextFeatureBuilder(db_path=text_db_path).enrich_price_frame(price_df)


def selftest_time_alignment() -> dict[str, object]:
    """합성 시각 시험: UTC 발행 기사가 KST 시세 봉에 9시간 앞서 붙지 않는지, 수집 시각 이전 봉에 붙지 않는지.

    임시 DB만 쓰고 서비스 DB는 건드리지 않는다. 실패하면 AssertionError.
    """
    utc = timezone.utc
    cases = [
        # (이름, 발행 UTC, 처음 수집 UTC): 발행 2026-10-15 01:50 UTC = 10:50 KST
        ("실시간 기사", datetime(2026, 10, 15, 1, 50, tzinfo=utc), datetime(2026, 10, 15, 1, 55, tzinfo=utc)),
        # 발행은 10-10이지만 우리가 10-15 03:00 UTC(12:00 KST)에 처음 봤다 → 12:00 봉 전에는 못 쓴다
        ("늦게 수집한 옛 기사", datetime(2026, 10, 10, 6, 0, tzinfo=utc), datetime(2026, 10, 15, 3, 0, tzinfo=utc)),
    ]
    records = [
        TextRecord("news_rss", "selftest", pub, name, "", f"test://{i}", ticker_hint="KRW-BTC", available_at=avail)
        for i, (name, pub, avail) in enumerate(cases)
    ]
    grid = pd.DataFrame({"timestamp": pd.date_range("2026-10-09 00:00", "2026-10-15 23:45", freq="15min")})  # KST 시세 격자
    out: dict[str, object] = {"cases": []}
    with tempfile.TemporaryDirectory() as tmp:
        builder = TextFeatureBuilder(db_path=str(Path(tmp) / "selftest_llm_agent.db"))
        builder.persist_raw_records(records)
        feats = builder.build_15m_features(builder.load_raw_records(), price_index=grid)
        feats = feats[feats["ticker"] == "KRW-BTC"]
        hit = feats[feats["text_event_count"] > 0][["timestamp", "text_event_count"]]
        for name, pub, avail in cases:
            effective = max(to_kst_naive(pub), to_kst_naive(avail))
            old_label = pd.Timestamp(pub.replace(tzinfo=None)).floor("15min")  # 이전 구현: UTC 시각을 그대로 KST 라벨로
            new_label = pd.Timestamp(effective).floor("15min")
            assert new_label in set(hit["timestamp"]), f"{name}: 새 라벨 {new_label}에 기사가 없다"
            # 봉 라벨 T의 정보는 T+15분에 알려진다 → 라벨 + 15분이 effective_at보다 늦어야 선행이 없다
            lead_old_h = (pd.Timestamp(effective) - (old_label + pd.Timedelta(minutes=15))).total_seconds() / 3600
            lead_new_h = (pd.Timestamp(effective) - (new_label + pd.Timedelta(minutes=15))).total_seconds() / 3600
            assert lead_new_h < 0, f"{name}: 새 방식도 선행 {lead_new_h:.2f}시간"
            out["cases"].append({
                "case": name, "published_kst": str(to_kst_naive(pub)), "available_kst": str(to_kst_naive(avail)),
                "old_label": str(old_label), "new_label": str(new_label),
                "old_lead_hours": round(lead_old_h, 2), "new_lead_hours": round(lead_new_h, 2),
            })
        assert int(hit["text_event_count"].sum()) == len(cases), "기사 수가 맞지 않다"
        early = builder.load_news_as_of(datetime(2026, 10, 15, 10, 49), ticker="KRW-BTC", lookback_hours=24 * 7)
        later = builder.load_news_as_of(datetime(2026, 10, 15, 12, 0), ticker="KRW-BTC", lookback_hours=24 * 7)
        assert len(early) == 0, "10:49 KST 판단에 10:55 KST에 처음 본 기사가 들어갔다"
        assert len(later) == 2, "12:00 KST 판단에는 두 기사 모두 들어가야 한다"
        out["as_of_check"] = {"10:49 KST 기사 수": len(early), "12:00 KST 기사 수": len(later)}
    out["passed"] = True
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="text context 유틸리티")
    parser.add_argument("--selftest", action="store_true", help="합성 시각 정렬 시험(임시 DB)")
    args = parser.parse_args()
    if args.selftest:
        print(json.dumps(selftest_time_alignment(), ensure_ascii=False, indent=2))
