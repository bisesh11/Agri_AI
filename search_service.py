"""Agricultural Search Service Module.

Provides live online information retrieval using real search engines (DuckDuckGo,
SerpApi, Tavily), URL validation, domain prioritization for trusted agricultural
institutions (ICAR, TNAU, IMD, Agmarknet, etc.), caching, and transparent fallback.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv

from source_fetcher import is_valid_http_url

load_dotenv()
logger = logging.getLogger("search_service")

# Trusted Indian and Global Agricultural Domains prioritized in ranking
TRUSTED_AGRI_DOMAINS = [
    "icar.gov.in",
    "agritech.tnau.ac.in",
    "india.gov.in",
    "imdagrimet.gov.in",
    "agmarknet.gov.in",
    "farmer.gov.in",
    "agricoop.nic.in",
    "vikaspedia.in",
    "pau.edu",
    "iari.res.in",
    "fao.org",
    "cimmyt.org",
    "irri.org",
    "kiran.nic.in"
]

# Official agricultural fallback portals when search services are unavailable
OFFICIAL_STATIC_PORTALS = [
    {
        "title": "Indian Council of Agricultural Research (ICAR)",
        "url": "https://icar.gov.in/",
        "snippet": "Apex body for coordinating, guiding, and managing research and education in agriculture in India.",
        "source": "icar.gov.in",
        "is_static_fallback": True
    },
    {
        "title": "TNAU Agritech Portal",
        "url": "https://www.agritech.tnau.ac.in/",
        "snippet": "Comprehensive agricultural knowledge hub covering crop production, plant protection, and farm advisories.",
        "source": "agritech.tnau.ac.in",
        "is_static_fallback": True
    },
    {
        "title": "India Meteorological Department - Agricultural Meteorology",
        "url": "https://imdagrimet.gov.in/",
        "snippet": "Official weather forecasting, agromet advisories, and monsoon bulletins for all Indian districts.",
        "source": "imdagrimet.gov.in",
        "is_static_fallback": True
    },
    {
        "title": "Agmarknet - Directorate of Marketing and Inspection",
        "url": "https://agmarknet.gov.in/",
        "snippet": "Government portal providing daily market arrivals and wholesale prices of agricultural commodities.",
        "source": "agmarknet.gov.in",
        "is_static_fallback": True
    }
]


class SearchCache:
    """In-memory cache with Time-To-Live (TTL) to prevent duplicate API hits."""

    def __init__(self, ttl_seconds: int = 900):
        self.ttl = ttl_seconds
        self.cache: Dict[str, Dict[str, Any]] = {}

    def get(self, key: str) -> Optional[List[Dict[str, Any]]]:
        record = self.cache.get(key)
        if record:
            if time.time() - record["timestamp"] < self.ttl:
                return record["results"]
            else:
                del self.cache[key]
        return None

    def set(self, key: str, results: List[Dict[str, Any]]) -> None:
        self.cache[key] = {
            "results": results,
            "timestamp": time.time()
        }


# Global search cache instance
_SEARCH_CACHE = SearchCache(ttl_seconds=900)


class SearchProvider(ABC):
    """Abstract Base Class for search providers."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        pass


class DuckDuckGoProvider(SearchProvider):
    """Real live search integration via DuckDuckGo (zero API key required)."""

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        results = []
        try:
            # Try ddgs first, then duckduckgo_search
            try:
                from ddgs import DDGS
            except ImportError:
                from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                raw_results = list(ddgs.text(query, max_results=max_results))
                for r in raw_results:
                    url = r.get("href") or r.get("link", "")
                    title = r.get("title", "")
                    snippet = r.get("body") or r.get("snippet", "")
                    if url and is_valid_http_url(url):
                        domain = urlparse(url).netloc
                        results.append({
                            "title": title or domain,
                            "url": url,
                            "snippet": snippet,
                            "source": domain,
                            "published_date": "Online Source",
                            "is_static_fallback": False
                        })
        except Exception as e:
            logger.warning(f"DuckDuckGo search error: {e}")
        return results


class SerpApiProvider(SearchProvider):
    """SerpApi Google search integration (activated when SERPAPI_API_KEY is configured)."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        import requests
        results = []
        try:
            params = {
                "engine": "google",
                "q": query,
                "api_key": self.api_key,
                "num": max_results,
                "hl": "en",
                "gl": "in"
            }
            resp = requests.get("https://serpapi.com/search.json", params=params, timeout=8)
            if resp.status_code == 200:
                data = resp.json()
                for org in data.get("organic_results", [])[:max_results]:
                    url = org.get("link", "")
                    if url and is_valid_http_url(url):
                        results.append({
                            "title": org.get("title", ""),
                            "url": url,
                            "snippet": org.get("snippet", ""),
                            "source": urlparse(url).netloc,
                            "published_date": org.get("date", "Online Source"),
                            "is_static_fallback": False
                        })
        except Exception as e:
            logger.warning(f"SerpApi search error: {e}")
        return results


class TavilyProvider(SearchProvider):
    """Tavily search provider (activated when TAVILY_API_KEY is configured)."""

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        import requests
        results = []
        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json={"query": query, "api_key": self.api_key, "max_results": max_results},
                timeout=8
            )
            if resp.status_code == 200:
                data = resp.json()
                for item in data.get("results", [])[:max_results]:
                    url = item.get("url", "")
                    if url and is_valid_http_url(url):
                        results.append({
                            "title": item.get("title", ""),
                            "url": url,
                            "snippet": item.get("content", ""),
                            "source": urlparse(url).netloc,
                            "published_date": item.get("published_date", "Online Source"),
                            "is_static_fallback": False
                        })
        except Exception as e:
            logger.warning(f"Tavily search error: {e}")
        return results


def get_search_provider() -> SearchProvider:
    """Factory selecting the best configured search provider."""
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if serpapi_key and len(serpapi_key.strip()) > 5:
        return SerpApiProvider(serpapi_key.strip())

    tavily_key = os.getenv("TAVILY_API_KEY")
    if tavily_key and len(tavily_key.strip()) > 5:
        return TavilyProvider(tavily_key.strip())

    return DuckDuckGoProvider()


def generate_search_query(user_question: str, predicted_topic: Optional[str] = None) -> str:
    """Refines the user's raw question into an effective agricultural search query."""
    cleaned = user_question.strip()

    # Topic keyword hints to boost relevant agricultural extension content
    topic_keywords = {
        "crop_cultivation": "agriculture cultivation practices advisory",
        "plant_disease": "crop disease symptoms management treatment",
        "pest_control": "crop pest control insect management",
        "fertilizer": "fertilizer dosage NPK application recommendations",
        "irrigation": "irrigation water management schedule",
        "soil_health": "soil fertility soil test health management",
        "seed_selection": "certified seed variety selection agriculture",
        "rice_cultivation": "paddy rice cultivation practices advisory",
        "vegetable_cultivation": "vegetable farming package of practices",
        "organic_farming": "organic farming natural bio-fertilizer practices",
        "weather": "agricultural weather agromet advisory forecast",
        "market_prices": "commodity mandi wholesale price Agmarknet",
        "government_schemes": "government agriculture scheme subsidy eligibility",
        "weed_management": "herbicide weed management agriculture",
        "harvesting": "harvesting storage post harvest management",
        "general_agriculture": "agricultural extension guidelines ICAR"
    }

    # Only append if query is short or needs domain context
    query = cleaned
    if predicted_topic and predicted_topic in topic_keywords:
        # Check if query already has agricultural keywords
        hint = topic_keywords[predicted_topic]
        if not any(k in cleaned.lower() for k in ["crop", "agriculture", "farmer", "plant", "field"]):
            query = f"{cleaned} agriculture"

    return query


def rank_and_deduplicate_sources(sources: List[Dict[str, Any]], max_results: int = 5) -> List[Dict[str, Any]]:
    """Deduplicates URLs, validates links, and ranks trusted agricultural institutions higher."""
    seen_urls = set()
    cleaned_sources = []

    for src in sources:
        url = src.get("url", "").strip()
        if not url or not is_valid_http_url(url):
            continue

        # Canonicalize URL (remove fragments and trailing slashes)
        canonical = url.split("#")[0].rstrip("/")
        if canonical.lower() in seen_urls:
            continue
        seen_urls.add(canonical.lower())

        # Calculate authority score
        domain = src.get("source", "").lower()
        authority_boost = 0
        for trusted in TRUSTED_AGRI_DOMAINS:
            if trusted in domain:
                authority_boost = 5
                break

        src["authority_boost"] = authority_boost
        src["canonical_url"] = canonical
        cleaned_sources.append(src)

    # Sort sources: trusted institutional domains first, then original order
    cleaned_sources.sort(key=lambda s: s.get("authority_boost", 0), reverse=True)
    return cleaned_sources[:max_results]


def search_agricultural_web(
    question: str,
    predicted_topic: Optional[str] = None,
    max_results: int = 4
) -> Dict[str, Any]:
    """Retrieves live online agricultural information with caching, ranking, and error handling.

    Returns:
        Dict with keys:
            'query': search query used
            'results': list of source dicts
            'status': 'ONLINE_SUCCESS', 'CACHE_HIT', or 'OFFLINE_FALLBACK'
            'provider': name of search provider
            'error_message': error description if applicable
    """
    search_query = generate_search_query(question, predicted_topic)
    cache_key = search_query.lower()

    # Check Cache
    cached_results = _SEARCH_CACHE.get(cache_key)
    if cached_results is not None and len(cached_results) > 0:
        return {
            "query": search_query,
            "results": cached_results,
            "status": "CACHE_HIT",
            "provider": "Cache",
            "error_message": None
        }

    provider = get_search_provider()
    provider_name = provider.__class__.__name__

    try:
        raw_results = provider.search(search_query, max_results=max_results + 3)
        ranked = rank_and_deduplicate_sources(raw_results, max_results=max_results)

        if ranked:
            _SEARCH_CACHE.set(cache_key, ranked)
            return {
                "query": search_query,
                "results": ranked,
                "status": "ONLINE_SUCCESS",
                "provider": provider_name,
                "error_message": None
            }
        else:
            # Honest fallback when zero online results returned
            return {
                "query": search_query,
                "results": OFFICIAL_STATIC_PORTALS[:max_results],
                "status": "OFFLINE_FALLBACK",
                "provider": provider_name,
                "error_message": "No online search results were returned by the provider. Displaying official static portals."
            }

    except Exception as e:
        logger.error(f"Search retrieval failed: {e}")
        return {
            "query": search_query,
            "results": OFFICIAL_STATIC_PORTALS[:max_results],
            "status": "OFFLINE_FALLBACK",
            "provider": provider_name,
            "error_message": f"Online search provider encountered an error: {str(e)}. Displaying official static portals."
        }

