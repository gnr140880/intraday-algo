"""
News Aggregator – Global & India Stock Market News

Sources (no API key required):
  - Google News RSS (India business, market, NIFTY, global markets)
  - Economic Times Markets RSS
  - Moneycontrol RSS
  - LiveMint Markets RSS
  - Reuters Business RSS
  - CNBC RSS

Sources (API key optional):
  - NewsAPI.org (if NEWS_API_KEY is set)

Features:
  - Fetches today's news from all sources
  - Deduplicates by headline similarity
  - Basic sentiment analysis via TextBlob
  - Categories: india_market, global_market, nifty, sector, commodity, forex
  - Caches results for 10 minutes to avoid hammering feeds
"""
import logging
import re
import hashlib
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from datetime import datetime, date, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import feedparser
import httpx
from bs4 import BeautifulSoup
from textblob import TextBlob

from config import settings

logger = logging.getLogger(__name__)

# Indian Standard Time (UTC+5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# ── RSS Feed Sources ──────────────────────────────────────────

INDIA_RSS_FEEDS = {
    # Economic Times
    "ET Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "ET Stocks": "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",
    "ET Commodities": "https://economictimes.indiatimes.com/markets/commodities/rssfeeds/1808152898.cms",
    # Moneycontrol
    "MC Market": "https://www.moneycontrol.com/rss/marketreports.xml",
    "MC Business": "https://www.moneycontrol.com/rss/business.xml",
    # LiveMint
    "Mint Markets": "https://www.livemint.com/rss/markets",
    "Mint Companies": "https://www.livemint.com/rss/companies",
    # NDTV Profit
    "NDTV Profit": "https://feeds.feedburner.com/ndtvprofit-latest",
}

GLOBAL_RSS_FEEDS = {
    # Reuters
    "Reuters Business": "https://feeds.reuters.com/reuters/businessNews",
    "Reuters Markets": "https://feeds.reuters.com/reuters/marketsNews",
    # CNBC
    "CNBC Top": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "CNBC World": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100727362",
    "CNBC Market": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=20910258",
    # MarketWatch
    "MarketWatch Top": "http://feeds.marketwatch.com/marketwatch/topstories/",
    "MarketWatch Markets": "http://feeds.marketwatch.com/marketwatch/marketpulse/",
    # Yahoo Finance
    "Yahoo Finance": "https://finance.yahoo.com/news/rssindex",
    # Bloomberg (via Google News)
    "Bloomberg via Google": "https://news.google.com/rss/search?q=bloomberg+stock+market&hl=en-US&gl=US&ceid=US:en",
}

GOOGLE_NEWS_QUERIES = [
    # India-specific
    ("NIFTY today", "https://news.google.com/rss/search?q=NIFTY+today&hl=en-IN&gl=IN&ceid=IN:en"),
    ("Bank NIFTY", "https://news.google.com/rss/search?q=Bank+NIFTY+today&hl=en-IN&gl=IN&ceid=IN:en"),
    ("Indian stock market", "https://news.google.com/rss/search?q=Indian+stock+market+today&hl=en-IN&gl=IN&ceid=IN:en"),
    ("Sensex today", "https://news.google.com/rss/search?q=Sensex+today&hl=en-IN&gl=IN&ceid=IN:en"),
    ("RBI policy", "https://news.google.com/rss/search?q=RBI+policy+markets&hl=en-IN&gl=IN&ceid=IN:en"),
    ("FII DII flow", "https://news.google.com/rss/search?q=FII+DII+flow+India&hl=en-IN&gl=IN&ceid=IN:en"),
    # Global
    ("US stock market", "https://news.google.com/rss/search?q=US+stock+market+today&hl=en-US&gl=US&ceid=US:en"),
    ("Fed interest rate", "https://news.google.com/rss/search?q=Federal+Reserve+rate&hl=en-US&gl=US&ceid=US:en"),
    ("Crude oil price", "https://news.google.com/rss/search?q=crude+oil+price+today&hl=en-US&gl=US&ceid=US:en"),
    ("Gold price today", "https://news.google.com/rss/search?q=gold+price+today&hl=en-US&gl=US&ceid=US:en"),
    ("Dollar rupee", "https://news.google.com/rss/search?q=dollar+rupee+exchange+rate&hl=en-IN&gl=IN&ceid=IN:en"),
    ("Asian markets", "https://news.google.com/rss/search?q=Asian+stock+markets+today&hl=en-US&gl=US&ceid=US:en"),
]


# ── Data Model ────────────────────────────────────────────────

@dataclass
class NewsItem:
    title: str
    summary: str
    url: str
    source: str
    published: str                  # ISO format
    category: str = "general"       # india_market, global_market, nifty, sector, commodity, forex
    sentiment: float = 0.0          # -1.0 to 1.0
    sentiment_label: str = "neutral"  # bullish, bearish, neutral
    relevance_score: float = 0.0    # 0–100
    keywords: List[str] = field(default_factory=list)
    _hash: str = ""

    def __post_init__(self):
        self._hash = hashlib.md5(
            (self.title.lower().strip()[:80]).encode()
        ).hexdigest()

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("_hash", None)
        return d


# ── Category Detection ────────────────────────────────────────

INDIA_KEYWORDS = [
    "nifty", "sensex", "bse", "nse", "indian market", "india stock",
    "rbi", "rupee", "fii", "dii", "sebi", "banknifty", "bank nifty",
    "midcap", "smallcap", "f&o", "nifty 50", "nifty50",
]
GLOBAL_KEYWORDS = [
    "s&p 500", "nasdaq", "dow jones", "wall street", "fed", "fomc",
    "us stock", "european market", "asian market", "ftse", "dax",
    "global market", "world market", "china market", "japan market",
]
COMMODITY_KEYWORDS = ["gold", "silver", "crude oil", "brent", "wti", "copper", "natural gas", "commodity"]
FOREX_KEYWORDS = ["dollar", "rupee", "forex", "currency", "yen", "euro", "gbp", "usd"]
NIFTY_KEYWORDS = ["nifty", "banknifty", "bank nifty", "finnifty", "nifty 50"]
SECTOR_KEYWORDS = [
    "it sector", "pharma", "banking sector", "auto sector", "metal",
    "fmcg", "realty", "energy sector", "infra",
]


def _classify_category(text: str) -> str:
    t = text.lower()
    if any(k in t for k in NIFTY_KEYWORDS):
        return "nifty"
    if any(k in t for k in COMMODITY_KEYWORDS):
        return "commodity"
    if any(k in t for k in FOREX_KEYWORDS):
        return "forex"
    if any(k in t for k in SECTOR_KEYWORDS):
        return "sector"
    if any(k in t for k in INDIA_KEYWORDS):
        return "india_market"
    if any(k in t for k in GLOBAL_KEYWORDS):
        return "global_market"
    return "general"


def _extract_keywords(text: str) -> List[str]:
    t = text.lower()
    all_kw = INDIA_KEYWORDS + GLOBAL_KEYWORDS + COMMODITY_KEYWORDS + FOREX_KEYWORDS + NIFTY_KEYWORDS + SECTOR_KEYWORDS
    return list(set(k for k in all_kw if k in t))


def _compute_relevance(text: str) -> float:
    """Higher relevance if more market keywords are present."""
    t = text.lower()
    all_kw = INDIA_KEYWORDS + GLOBAL_KEYWORDS + COMMODITY_KEYWORDS + FOREX_KEYWORDS + NIFTY_KEYWORDS
    hits = sum(1 for k in all_kw if k in t)
    return min(100.0, hits * 15.0)


# ── Sentiment ─────────────────────────────────────────────────

def _analyze_sentiment(text: str) -> tuple:
    """Returns (polarity float, label str)."""
    try:
        blob = TextBlob(text)
        pol = blob.sentiment.polarity
        if pol > 0.1:
            return round(pol, 3), "bullish"
        elif pol < -0.1:
            return round(pol, 3), "bearish"
        return round(pol, 3), "neutral"
    except Exception:
        return 0.0, "neutral"


# ── RSS Parsing ───────────────────────────────────────────────

def _parse_feed(source_name: str, url: str, max_items: int = 15) -> List[NewsItem]:
    """Parse a single RSS feed, return NewsItems."""
    items = []
    try:
        feed = feedparser.parse(url)
        today_ist = datetime.now(IST).date()
        for entry in feed.entries[:max_items]:
            # Parse date
            published_str = ""
            pub_date = None
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                pub_date = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
                published_str = pub_date.isoformat()
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                pub_date = datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)
                published_str = pub_date.isoformat()
            else:
                published_str = datetime.now(timezone.utc).isoformat()
                pub_date = datetime.now(timezone.utc)

            # Filter to today's date in IST only
            if pub_date and pub_date.astimezone(IST).date() != today_ist:
                continue

            title = entry.get("title", "").strip()
            if not title:
                continue

            # Summary – strip HTML tags
            summary_raw = entry.get("summary", entry.get("description", ""))
            summary = BeautifulSoup(summary_raw, "html.parser").get_text()[:500].strip()

            link = entry.get("link", "")
            full_text = f"{title} {summary}"

            sentiment, label = _analyze_sentiment(full_text)
            category = _classify_category(full_text)
            keywords = _extract_keywords(full_text)
            relevance = _compute_relevance(full_text)

            items.append(NewsItem(
                title=title,
                summary=summary,
                url=link,
                source=source_name,
                published=published_str,
                category=category,
                sentiment=sentiment,
                sentiment_label=label,
                relevance_score=relevance,
                keywords=keywords,
            ))
    except Exception as e:
        logger.warning(f"Failed to parse feed '{source_name}': {e}")
    return items


# ── NewsAPI.org (optional paid source) ────────────────────────

def _fetch_newsapi(query: str, category_hint: str, max_items: int = 10) -> List[NewsItem]:
    """Fetch from NewsAPI.org if API key is configured."""
    if not settings.news_api_key or settings.news_api_key == "your_newsapi_key_here":
        return []
    items = []
    try:
        today_str = datetime.now(IST).date().isoformat()
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": today_str,
            "to": today_str,
            "sortBy": "publishedAt",
            "pageSize": max_items,
            "language": "en",
            "apiKey": settings.news_api_key,
        }
        resp = httpx.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        for article in data.get("articles", []):
            title = article.get("title", "").strip()
            if not title or title == "[Removed]":
                continue
            summary = (article.get("description") or "")[:500]
            full_text = f"{title} {summary}"
            sentiment, label = _analyze_sentiment(full_text)
            items.append(NewsItem(
                title=title,
                summary=summary,
                url=article.get("url", ""),
                source=f"NewsAPI: {article.get('source', {}).get('name', 'Unknown')}",
                published=article.get("publishedAt", ""),
                category=category_hint,
                sentiment=sentiment,
                sentiment_label=label,
                relevance_score=_compute_relevance(full_text),
                keywords=_extract_keywords(full_text),
            ))
    except Exception as e:
        logger.warning(f"NewsAPI fetch failed for '{query}': {e}")
    return items


# ── Deduplication ─────────────────────────────────────────────

def _deduplicate(items: List[NewsItem]) -> List[NewsItem]:
    seen_hashes = set()
    unique = []
    for item in items:
        if item._hash not in seen_hashes:
            seen_hashes.add(item._hash)
            unique.append(item)
    return unique


# ── Main Aggregator ───────────────────────────────────────────

class NewsAggregator:
    """
    Fetches, classifies, and caches today's stock market news
    from multiple RSS and API sources.
    """

    def __init__(self, cache_ttl_seconds: int = 600):
        self.cache_ttl = cache_ttl_seconds
        self._cache: Optional[Dict] = None
        self._cache_time: Optional[datetime] = None

    def _is_cache_valid(self) -> bool:
        if self._cache is None or self._cache_time is None:
            return False
        return (datetime.now() - self._cache_time).total_seconds() < self.cache_ttl

    def fetch_all(self, force_refresh: bool = False) -> Dict:
        """
        Fetch all news. Returns structured dict with categories.
        Cached for `cache_ttl` seconds.
        """
        if not force_refresh and self._is_cache_valid():
            return self._cache

        all_items: List[NewsItem] = []

        # ── Parallel RSS fetch ──
        feeds_to_fetch = []
        for name, url in INDIA_RSS_FEEDS.items():
            feeds_to_fetch.append((name, url))
        for name, url in GLOBAL_RSS_FEEDS.items():
            feeds_to_fetch.append((name, url))
        for query_name, url in GOOGLE_NEWS_QUERIES:
            feeds_to_fetch.append((f"Google: {query_name}", url))

        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(_parse_feed, name, url, 15): name
                for name, url in feeds_to_fetch
            }
            for future in as_completed(futures, timeout=30):
                try:
                    items = future.result(timeout=15)
                    all_items.extend(items)
                except Exception as e:
                    logger.warning(f"Feed fetch error: {e}")

        # ── NewsAPI (if key available) ──
        newsapi_queries = [
            ("Indian stock market NIFTY", "india_market"),
            ("global stock market S&P 500", "global_market"),
            ("crude oil gold commodity", "commodity"),
        ]
        for query, cat in newsapi_queries:
            all_items.extend(_fetch_newsapi(query, cat))

        # ── Deduplicate + sort ──
        all_items = _deduplicate(all_items)
        all_items.sort(key=lambda x: x.relevance_score, reverse=True)

        # ── Categorise ──
        by_category = {}
        for item in all_items:
            by_category.setdefault(item.category, []).append(item.to_dict())

        # ── Sentiment summary ──
        sentiments = [i.sentiment for i in all_items if i.sentiment != 0]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        bullish_count = sum(1 for i in all_items if i.sentiment_label == "bullish")
        bearish_count = sum(1 for i in all_items if i.sentiment_label == "bearish")
        neutral_count = sum(1 for i in all_items if i.sentiment_label == "neutral")

        if avg_sentiment > 0.05:
            overall_mood = "BULLISH"
        elif avg_sentiment < -0.05:
            overall_mood = "BEARISH"
        else:
            overall_mood = "NEUTRAL"

        result = {
            "total_articles": len(all_items),
            "last_updated": datetime.now().isoformat(),
            "sentiment_summary": {
                "overall_mood": overall_mood,
                "avg_sentiment": round(avg_sentiment, 4),
                "bullish_count": bullish_count,
                "bearish_count": bearish_count,
                "neutral_count": neutral_count,
            },
            "categories": {
                "india_market": by_category.get("india_market", []),
                "global_market": by_category.get("global_market", []),
                "nifty": by_category.get("nifty", []),
                "sector": by_category.get("sector", []),
                "commodity": by_category.get("commodity", []),
                "forex": by_category.get("forex", []),
                "general": by_category.get("general", []),
            },
            "top_stories": [i.to_dict() for i in all_items[:20]],
        }

        self._cache = result
        self._cache_time = datetime.now()
        logger.info(
            f"News aggregated: {len(all_items)} articles, "
            f"mood={overall_mood}, sentiment={avg_sentiment:.4f}"
        )
        return result

    def get_india_news(self, limit: int = 30) -> List[Dict]:
        data = self.fetch_all()
        combined = (
            data["categories"].get("india_market", [])
            + data["categories"].get("nifty", [])
            + data["categories"].get("sector", [])
        )
        combined.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return combined[:limit]

    def get_global_news(self, limit: int = 30) -> List[Dict]:
        data = self.fetch_all()
        combined = (
            data["categories"].get("global_market", [])
            + data["categories"].get("commodity", [])
            + data["categories"].get("forex", [])
        )
        combined.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
        return combined[:limit]

    def get_sentiment_summary(self) -> Dict:
        data = self.fetch_all()
        return data.get("sentiment_summary", {})

    def invalidate_cache(self):
        self._cache = None
        self._cache_time = None


# Singleton
news_aggregator = NewsAggregator(cache_ttl_seconds=600)
