"""Test Agricultural Search Integration Module."""

import pytest
from unittest.mock import patch, MagicMock
from search_service import (
    generate_search_query,
    rank_and_deduplicate_sources,
    SearchCache,
    search_agricultural_web,
    OFFICIAL_STATIC_PORTALS
)
from source_fetcher import is_valid_http_url


def test_is_valid_http_url():
    assert is_valid_http_url("https://icar.gov.in/content/crop-science") is True
    assert is_valid_http_url("http://agritech.tnau.ac.in/") is True
    assert is_valid_http_url("ftp://files.example.com") is False
    assert is_valid_http_url("invalid-url") is False
    assert is_valid_http_url("http://localhost:8000") is False
    assert is_valid_http_url("http://127.0.0.1/test") is False
    assert is_valid_http_url(None) is False


def test_generate_search_query():
    q = "control yellow rust in wheat"
    query = generate_search_query(q, "plant_disease")
    assert "yellow rust" in query
    assert "wheat" in query


def test_rank_and_deduplicate_sources():
    raw_sources = [
        {"title": "Dup 1", "url": "https://example.com/page1", "source": "example.com"},
        {"title": "Dup 2", "url": "https://example.com/page1#section", "source": "example.com"},
        {"title": "ICAR Advisory", "url": "https://icar.gov.in/wheat-rust", "source": "icar.gov.in"},
        {"title": "Invalid URL", "url": "not_a_valid_url", "source": "unknown"}
    ]
    ranked = rank_and_deduplicate_sources(raw_sources, max_results=5)
    # Deduplication check
    assert len(ranked) == 2
    # ICAR is an official trusted domain, so it should be ranked first
    assert "icar.gov.in" in ranked[0]["url"]


def test_search_cache_ttl():
    cache = SearchCache(ttl_seconds=1)
    sample_res = [{"title": "Test", "url": "https://icar.gov.in"}]
    cache.set("wheat query", sample_res)
    assert cache.get("wheat query") == sample_res
    # Expired entry check
    import time
    time.sleep(1.1)
    assert cache.get("wheat query") is None


def test_search_provider_offline_fallback():
    # Mock search provider to raise exception and verify honest fallback
    with patch("search_service.get_search_provider") as mock_provider_factory:
        mock_provider = MagicMock()
        mock_provider.search.side_effect = Exception("Simulated network failure")
        mock_provider_factory.return_value = mock_provider

        res = search_agricultural_web("How to treat blight?")
        assert res["status"] == "OFFLINE_FALLBACK"
        assert len(res["results"]) > 0
        assert res["results"][0]["is_static_fallback"] is True

