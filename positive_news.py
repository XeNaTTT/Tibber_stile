"""Small, dependency-free RSS selector for constructive German news."""

from __future__ import annotations

import datetime as dt
import html
import json
import logging
import os
import re
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from urllib.parse import urlsplit, urlunsplit

import requests


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_FILE = os.path.join(PROJECT_DIR, "positive_news_cache.json")
RSS_TIMEOUT = (2.5, 4.0)
MAX_CANDIDATES = 50

# Official feed URLs, documented by the respective broadcasters.
RSS_SOURCES = (
    ("tagesschau/wissen", "https://www.tagesschau.de/wissen/index~rss2.xml"),
    ("tagesschau/forschung", "https://www.tagesschau.de/wissen/forschung/index~rss2.xml"),
    ("tagesschau/technologie", "https://www.tagesschau.de/wissen/technologie/index~rss2.xml"),
    ("tagesschau/klima", "https://www.tagesschau.de/wissen/klima/index~rss2.xml"),
    ("DLF Wissen", "https://www.deutschlandfunk.de/wissen-106.rss"),
)

POSITIVE_TERMS = {
    "erfolg": 3, "erfolgreich": 3, "fortschritt": 4, "gelingt": 3,
    "gelungen": 3, "verbessert": 3, "verbessern": 3, "verbesserung": 3,
    "entdeckung": 3, "entdeckt": 2, "entwickelt": 2, "entwicklung": 1,
    "durchbruch": 4, "rekord": 2, "schutz": 2, "schützen": 2,
    "geschützt": 2, "rettet": 3, "gerettet": 3, "erholt": 3,
    "erholung": 3, "therapie": 3, "behandlung": 2, "heilung": 4,
    "forschung": 2, "innovation": 3, "erneuerbar": 3, "solar": 2,
    "solarenergie": 3, "windenergie": 3, "energiewende": 2,
    "klimaneutral": 3, "emissionen gesenkt": 4, "wieder angesiedelt": 4,
    "ausbau": 2, "lösung": 4, "loesung": 4, "chance": 1,
}
NEGATIVE_TERMS = (
    "krieg", "angriff", "tote", "getötet", "getoetet", "mord",
    "katastrophe", "unfall", "gewalt", "bomben", "raketen", "terror",
    "missbrauch", "geiseln", "explosion",
)
CONSTRUCTIVE_CONTEXT = (
    "neu", "senkt", "reduziert", "verhindert", "hilft", "besser",
    "therapie", "behandlung", "schutz", "lösung", "forschung",
)
TOPICS = {
    "medizin": ("medizin", "gesundheit", "therapie", "behandlung", "krebs", "patient"),
    "umwelt": ("klima", "umwelt", "natur", "artenschutz", "energie", "solar", "windkraft"),
    "technik": ("technik", "technologie", "digital", "computer", "robot", "innovation"),
    "bildung": ("bildung", "schule", "universität", "gesellschaft", "sozial"),
    "forschung": ("forschung", "studie", "wissenschaft", "entdeckung", "weltraum"),
}


def _local_name(tag):
    return tag.rsplit("}", 1)[-1].lower()


def clean_text(value, title=""):
    """Turn an RSS HTML fragment into compact, display-safe Unicode text."""
    text = html.unescape(value or "")
    text = re.sub(r"<\s*(?:br|/p|/div)\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\b(?:mehr|weiterlesen)\s*(?:[.›»…]+)?\s*$", "", text, flags=re.I)
    text = re.sub(r"^(?:von|autor(?:in)?):?\s+[^.]{2,60}[.|–-]\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n-|•")
    if title:
        cleaned_title = clean_text(title)
        if text.casefold().startswith(cleaned_title.casefold()):
            text = text[len(cleaned_title):].lstrip(" .:–-")
    return text


def normalize_headline(title):
    value = unicodedata.normalize("NFKC", clean_text(title)).casefold()
    value = re.sub(r"[^\wäöüß]+", " ", value, flags=re.UNICODE)
    return re.sub(r"\s+", " ", value).strip()


def canonical_url(url):
    try:
        parts = urlsplit((url or "").strip())
        if not parts.netloc:
            return ""
        path = parts.path.rstrip("/") or "/"
        return urlunsplit((parts.scheme.lower() or "https", parts.netloc.lower(), path, "", ""))
    except (TypeError, ValueError):
        return ""


def parse_timestamp(value):
    if not value:
        return None
    try:
        parsed = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def parse_rss(xml_data, source):
    """Parse RSS 2.0 or Atom, skipping incomplete/malformed items."""
    root = ET.fromstring(xml_data)
    articles = []
    entries = [node for node in root.iter() if _local_name(node.tag) in ("item", "entry")]
    for entry in entries:
        fields = {}
        links = []
        for child in entry:
            name = _local_name(child.tag)
            value = "".join(child.itertext()).strip()
            if name == "link":
                links.append(child.attrib.get("href") or value)
            elif name not in fields or not fields[name]:
                fields[name] = value
        title = clean_text(fields.get("title", ""))
        if not title:
            continue
        summary = clean_text(fields.get("description") or fields.get("summary")
                             or fields.get("content") or "", title)
        published = parse_timestamp(fields.get("pubdate") or fields.get("published")
                                    or fields.get("updated") or fields.get("date"))
        articles.append({
            "title": title, "summary": summary, "source": source,
            "url": next((link for link in links if link), ""),
            "published": published.isoformat() if published else None,
        })
    return articles


def deduplicate(articles):
    output, urls, headlines = [], set(), set()
    for article in articles:
        url = canonical_url(article.get("url"))
        headline = normalize_headline(article.get("title", ""))
        if (url and url in urls) or (headline and headline in headlines):
            continue
        if not headline:
            continue
        item = dict(article)
        item["url"] = url or article.get("url", "")
        output.append(item)
        if url:
            urls.add(url)
        headlines.add(headline)
    return output


def article_topic(article):
    text = f"{article.get('title', '')} {article.get('summary', '')}".casefold()
    scores = {topic: sum(text.count(word) for word in words)
              for topic, words in TOPICS.items()}
    topic, count = max(scores.items(), key=lambda pair: pair[1])
    return topic if count else "wissen"


def recency_score(published, now=None):
    timestamp = parse_timestamp(published) if isinstance(published, str) else published
    if not timestamp:
        return 0.0
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    hours = max(0, (now.astimezone(dt.timezone.utc) - timestamp).total_seconds() / 3600)
    if hours <= 12:
        return 3.0
    if hours <= 24:
        return 2.2
    if hours <= 48:
        return 1.3
    if hours <= 96:
        return 0.4
    return 0.0


def constructive_score(article, now=None):
    text = f"{article.get('title', '')} {article.get('summary', '')}".casefold()
    positive = sum(weight * text.count(term) for term, weight in POSITIVE_TERMS.items())
    negatives = sum(text.count(term) for term in NEGATIVE_TERMS)
    context = sum(text.count(term) for term in CONSTRUCTIVE_CONTEXT)
    # Serious terms remain a strong penalty, but genuine solution context offsets
    # part of it (for example, a therapy improving cancer survival).
    penalty = negatives * (3.0 if positive + context >= 4 else 7.0)
    return positive + min(context, 3) - penalty + recency_score(article.get("published"), now)


def select_stories(articles, limit=3, now=None):
    ranked = sorted(deduplicate(articles),
                    key=lambda item: constructive_score(item, now), reverse=True)
    # Do not fill the panel with strongly tragic articles merely to reach three.
    ranked = [item for item in ranked if constructive_score(item, now) > -2]
    selected, topics = [], set()
    while ranked and len(selected) < limit:
        diverse = next((item for item in ranked if article_topic(item) not in topics), None)
        choice = diverse or ranked[0]
        ranked.remove(choice)
        choice = dict(choice)
        choice["topic"] = article_topic(choice)
        selected.append(choice)
        topics.add(choice["topic"])
    return selected


def split_sentences(summary, maximum=3):
    text = clean_text(summary)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ0-9])", text)
    return [part.strip() for part in parts if part.strip()][:maximum]


def load_cache(path=CACHE_FILE):
    try:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        stories = payload.get("stories")
        if not isinstance(stories, list):
            return []
        return [item for item in stories if isinstance(item, dict) and item.get("title")][:3]
    except (OSError, ValueError, TypeError):
        return []


def write_cache(stories, path=CACHE_FILE, now=None):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    payload = {"updated_at": (now or dt.datetime.now(dt.timezone.utc)).isoformat(),
               "stories": stories[:3]}
    fd, temporary = tempfile.mkstemp(prefix=".positive_news_", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _fetch_source(source, url, session=requests):
    response = session.get(url, timeout=RSS_TIMEOUT,
                           headers={"User-Agent": "Tibber-stile/1.0 RSS reader"})
    response.raise_for_status()
    return parse_rss(response.content, source)


def refresh_positive_news(cache_path=CACHE_FILE, sources=RSS_SOURCES,
                          session=requests, now=None):
    """Attempt all feeds on every call; cache is reliability fallback only."""
    logging.info("Positive news: refreshing RSS")
    cached = load_cache(cache_path)
    fetched = []
    with ThreadPoolExecutor(max_workers=min(5, max(1, len(sources)))) as executor:
        futures = {executor.submit(_fetch_source, name, url, session): name
                   for name, url in sources}
        for future in as_completed(futures):
            name = futures[future]
            try:
                items = future.result()
                fetched.extend(items)
                logging.info("Positive news: source %s OK", name)
            except Exception as exc:
                logging.warning("Positive news: source %s unavailable (%s)", name, exc)
    fetched = fetched[:MAX_CANDIDATES]
    unique = deduplicate(fetched)
    logging.info("Positive news: fetched %d items", len(fetched))
    logging.info("Positive news: %d unique candidates", len(unique))
    selected = select_stories(unique, 3, now)
    if selected and len(selected) < 3:
        selected = deduplicate(selected + cached)[:3]
    if selected:
        try:
            write_cache(selected, cache_path, now)
            logging.info("Positive news: updated fallback cache")
        except OSError as exc:
            logging.warning("Positive news: cache update failed (%s)", exc)
        logging.info("Positive news: selected %d stories", len(selected))
        return selected
    if cached:
        logging.warning("Positive news: RSS refresh failed, using fallback cache")
        return cached
    logging.warning("Positive news: RSS refresh failed, no fallback cache")
    return []
