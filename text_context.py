"""
Realtime text-context ingestion and feature engineering for market analysis.

The module intentionally uses only the standard library plus pandas/duckdb so it
can run in the current uv environment without extra setup. API-backed sources
are optional; RSS and local CSV inputs work out of the box.
"""

from __future__ import annotations

import csv
import hashlib
import html
import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable, Sequence

import duckdb
import pandas as pd


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
    published_at: datetime
    title: str
    body: str
    url: str
    ticker_hint: str = "KRW-BTC"

    @property
    def record_id(self) -> str:
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


def default_rss_urls() -> list[str]:
    urls = []
    for query in DEFAULT_RSS_QUERIES:
        encoded = urllib.parse.quote_plus(query)
        urls.append(f"https://news.google.com/rss/search?q={encoded}&hl=ko&gl=KR&ceid=KR:ko")
    return urls


def env_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [part.strip() for part in raw.split("|") if part.strip()]


class TextDataCollector:
    """Collects realtime news/report/SNS text into normalized records."""

    def __init__(self, rss_urls: Sequence[str] | None = None, local_csv_paths: Sequence[str] | None = None):
        self.rss_urls = list(rss_urls or env_list("TEXT_RSS_URLS") or default_rss_urls())
        self.local_csv_paths = list(local_csv_paths or env_list("TEXT_LOCAL_CSVS"))

    def collect_all(self, max_items_per_source: int = 30) -> list[TextRecord]:
        records: list[TextRecord] = []
        records.extend(self.collect_rss(max_items_per_source=max_items_per_source))
        records.extend(self.collect_naver_news(max_items=max_items_per_source))
        records.extend(self.collect_local_csv())
        return deduplicate_records(records)

    def collect_rss(self, max_items_per_source: int = 30) -> list[TextRecord]:
        records: list[TextRecord] = []
        for url in self.rss_urls:
            try:
                with urllib.request.urlopen(url, timeout=12) as response:
                    payload = response.read()
                root = ET.fromstring(payload)
            except Exception as exc:
                print(f"[WARN] RSS fetch failed: {url} ({exc})")
                continue

            channel = root.find("channel")
            source_name = clean_text(channel.findtext("title") if channel is not None else url)
            items = root.findall(".//item")[:max_items_per_source]
            for item in items:
                title = clean_text(item.findtext("title"))
                body = clean_text(item.findtext("description"))
                link = clean_text(item.findtext("link"))
                published_at = parse_datetime(item.findtext("pubDate"))
                records.append(
                    TextRecord(
                        source_type="news_rss",
                        source_name=source_name or "rss",
                        published_at=published_at,
                        title=title,
                        body=body,
                        url=link,
                    )
                )
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
                            ticker_hint=row.get("ticker_hint") or "KRW-BTC",
                        )
                    )
        return records


def deduplicate_records(records: Iterable[TextRecord]) -> list[TextRecord]:
    by_id: dict[str, TextRecord] = {}
    for record in records:
        if record.title:
            by_id[record.record_id] = record
    return list(by_id.values())


class TextFeatureBuilder:
    """Builds 15-minute text factors aligned to the existing candle mart."""

    def __init__(self, db_path: str = "upbit_data.db"):
        self.db_path = db_path

    def persist_raw_records(self, records: Sequence[TextRecord]) -> int:
        with duckdb.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS text_events_raw (
                    record_id VARCHAR,
                    source_type VARCHAR,
                    source_name VARCHAR,
                    published_at TIMESTAMP,
                    title VARCHAR,
                    body VARCHAR,
                    url VARCHAR,
                    ticker_hint VARCHAR,
                    sentiment_score DOUBLE,
                    positive_hits INTEGER,
                    negative_hits INTEGER,
                    topic_macro INTEGER,
                    topic_risk INTEGER,
                    topic_crypto INTEGER,
                    topic_regulation INTEGER,
                    topic_liquidity INTEGER,
                    inserted_at TIMESTAMP
                )
                """
            )
            if not records:
                return 0

            rows = [self._record_to_row(record) for record in records]
            ids = [row["record_id"] for row in rows]
            con.register("ids_df", pd.DataFrame({"record_id": ids}))
            con.execute("DELETE FROM text_events_raw WHERE record_id IN (SELECT record_id FROM ids_df)")
            con.unregister("ids_df")
            con.register("rows_df", pd.DataFrame(rows))
            con.execute("INSERT INTO text_events_raw SELECT * FROM rows_df")
            con.unregister("rows_df")
        return len(records)

    def build_and_persist_15m_features(self, price_index: pd.DataFrame | None = None) -> pd.DataFrame:
        raw_df = self.load_raw_records()
        features = self.build_15m_features(raw_df, price_index=price_index)
        with duckdb.connect(self.db_path) as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS text_features_15m (
                    timestamp TIMESTAMP,
                    text_event_count INTEGER,
                    text_sentiment_mean DOUBLE,
                    text_sentiment_sum DOUBLE,
                    text_sentiment_abs_mean DOUBLE,
                    text_positive_hits INTEGER,
                    text_negative_hits INTEGER,
                    text_macro_count INTEGER,
                    text_risk_count INTEGER,
                    text_crypto_count INTEGER,
                    text_regulation_count INTEGER,
                    text_liquidity_count INTEGER,
                    text_news_count INTEGER,
                    text_report_count INTEGER,
                    text_sns_count INTEGER,
                    text_shock_z DOUBLE,
                    text_sentiment_momentum_1h DOUBLE,
                    updated_at TIMESTAMP
                )
                """
            )
            con.execute("DELETE FROM text_features_15m")
            con.register("features_df", features)
            con.execute("INSERT INTO text_features_15m SELECT * FROM features_df")
            con.unregister("features_df")
        return features

    def load_raw_records(self) -> pd.DataFrame:
        with duckdb.connect(self.db_path) as con:
            exists = con.execute(
                """
                SELECT COUNT(*)
                FROM information_schema.tables
                WHERE table_name = 'text_events_raw'
                """
            ).fetchone()[0]
            if not exists:
                return pd.DataFrame()
            return con.execute("SELECT * FROM text_events_raw").df()

    def build_15m_features(self, raw_df: pd.DataFrame, price_index: pd.DataFrame | None = None) -> pd.DataFrame:
        if raw_df.empty:
            feature_df = self._empty_feature_frame(price_index)
            return feature_df

        df = raw_df.copy()
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        df = df.dropna(subset=["published_at"])
        df["timestamp"] = df["published_at"].dt.tz_convert(None).dt.floor("15min")
        df["is_news"] = df["source_type"].str.contains("news|rss", case=False, na=False).astype(int)
        df["is_report"] = df["source_type"].str.contains("report", case=False, na=False).astype(int)
        df["is_sns"] = df["source_type"].str.contains("sns|reddit|x|twitter|stocktwits", case=False, na=False).astype(int)

        grouped = (
            df.groupby("timestamp")
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

        if price_index is not None and not price_index.empty:
            base = pd.DataFrame({"timestamp": pd.to_datetime(price_index["timestamp"]).dt.floor("15min").drop_duplicates()})
            base = base.sort_values("timestamp").reset_index(drop=True)
            grouped = base.merge(grouped, on="timestamp", how="left")

        feature_cols = [col for col in grouped.columns if col != "timestamp"]
        grouped = grouped.sort_values("timestamp").reset_index(drop=True)
        grouped[feature_cols] = grouped[feature_cols].fillna(0.0).infer_objects(copy=False)
        count_std = grouped["text_event_count"].rolling(96, min_periods=4).std()
        count_mean = grouped["text_event_count"].rolling(96, min_periods=4).mean()
        text_shock_z = (grouped["text_event_count"] - count_mean) / count_std
        grouped["text_shock_z"] = text_shock_z.where(count_std.ne(0), 0.0).fillna(0.0).astype(float)
        grouped["text_sentiment_momentum_1h"] = grouped["text_sentiment_mean"].rolling(4, min_periods=1).mean()
        grouped["updated_at"] = datetime.utcnow()
        return grouped

    def enrich_price_frame(self, price_df: pd.DataFrame) -> pd.DataFrame:
        raw_df = self.load_raw_records()
        features = self.build_15m_features(raw_df, price_index=price_df[["timestamp"]])
        merged = price_df.copy()
        merged["timestamp"] = pd.to_datetime(merged["timestamp"])
        features["timestamp"] = pd.to_datetime(features["timestamp"])
        return merged.merge(features.drop(columns=["updated_at"]), on="timestamp", how="left").fillna(0.0)

    def _record_to_row(self, record: TextRecord) -> dict[str, object]:
        text = f"{record.title} {record.body}".lower()
        positive_hits = sum(1 for term in POSITIVE_TERMS if term.lower() in text)
        negative_hits = sum(1 for term in NEGATIVE_TERMS if term.lower() in text)
        sentiment_score = (positive_hits - negative_hits) / max(positive_hits + negative_hits, 1)
        topics = {
            topic: int(any(term.lower() in text for term in terms))
            for topic, terms in TOPIC_TERMS.items()
        }
        return {
            "record_id": record.record_id,
            "source_type": record.source_type,
            "source_name": record.source_name,
            "published_at": record.published_at.replace(tzinfo=None),
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
            "inserted_at": datetime.utcnow(),
        }

    def _empty_feature_frame(self, price_index: pd.DataFrame | None) -> pd.DataFrame:
        if price_index is None or price_index.empty:
            timestamps = pd.Series(dtype="datetime64[ns]")
        else:
            timestamps = pd.to_datetime(price_index["timestamp"]).dt.floor("15min").drop_duplicates()
        columns = [
            "timestamp",
            "text_event_count",
            "text_sentiment_mean",
            "text_sentiment_sum",
            "text_sentiment_abs_mean",
            "text_positive_hits",
            "text_negative_hits",
            "text_macro_count",
            "text_risk_count",
            "text_crypto_count",
            "text_regulation_count",
            "text_liquidity_count",
            "text_news_count",
            "text_report_count",
            "text_sns_count",
            "text_shock_z",
            "text_sentiment_momentum_1h",
            "updated_at",
        ]
        frame = pd.DataFrame({"timestamp": timestamps})
        for column in columns:
            if column not in frame.columns:
                frame[column] = 0.0
        frame["updated_at"] = datetime.utcnow()
        return frame[columns]


def ingest_realtime_text_context(db_path: str = "upbit_data.db", max_items_per_source: int = 30) -> tuple[int, pd.DataFrame]:
    collector = TextDataCollector()
    records = collector.collect_all(max_items_per_source=max_items_per_source)
    builder = TextFeatureBuilder(db_path=db_path)
    inserted = builder.persist_raw_records(records)
    features = builder.build_and_persist_15m_features()
    return inserted, features


def load_price_with_text_context(
    db_path: str = "upbit_data.db",
    price_table: str = "btc_15m_advance",
    limit: int | None = None,
) -> pd.DataFrame:
    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    with duckdb.connect(db_path) as con:
        price_df = con.execute(
            f"""
            SELECT timestamp, open, high, low, close, volume, value
            FROM {price_table}
            ORDER BY timestamp DESC
            {limit_sql}
            """
        ).df()
    price_df = price_df.sort_values("timestamp").reset_index(drop=True)
    return TextFeatureBuilder(db_path=db_path).enrich_price_frame(price_df)
