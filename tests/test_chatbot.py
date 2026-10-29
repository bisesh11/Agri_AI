"""Test Chatbot Core Workflow and Topic Routing."""

import pytest
from unittest.mock import patch
from chatbot import answer_question, extract_location_hint


def test_extract_location_hint():
    assert extract_location_hint("What is the weather in Coimbatore today?") == "Coimbatore"
    assert extract_location_hint("Forecast for Nashik district") == "Nashik"
    assert extract_location_hint("How to irrigate soil in summer?") is None


def test_answer_question_invalid_input():
    res = answer_question("")
    assert res["search_status"] == "INVALID_INPUT"
    assert res["predicted_topic"] == "unknown"

    res_spaces = answer_question("   ")
    assert res_spaces["search_status"] == "INVALID_INPUT"


def test_answer_question_weather_routing():
    # Test weather routing with mocked weather service
    with patch("chatbot.get_live_weather") as mock_weather:
        mock_weather.return_value = {
            "status": "LIVE_SUCCESS",
            "location": "Coimbatore, Tamil Nadu, India",
            "current_temperature": 28.5,
            "humidity": 65,
            "wind_speed_kmh": 10.2,
            "current_condition": "Partly cloudy",
            "agricultural_advisory": ["Favorable conditions for field operations."],
            "source": "Open-Meteo Meteorological Service",
            "source_url": "https://open-meteo.com/"
        }
        res = answer_question("What is the weather in Coimbatore?", location="Coimbatore")
        assert res["predicted_topic"] == "weather"
        assert "Coimbatore" in res["answer"]
        assert "28.5" in res["answer"]
        assert res["search_status"] == "LIVE_SUCCESS"


def test_answer_question_mandi_routing():
    res = answer_question("What is the current market price of onions in Lasalgaon mandi?")
    assert res["predicted_topic"] == "market_prices"
    assert "Onion" in res["answer"]
    assert "Lasalgaon" in res["answer"]
    assert res["search_status"] == "LIVE_SUCCESS"


def test_answer_question_general_flow():
    # Test normal question processing with mocked search service
    with patch("chatbot.search_agricultural_web") as mock_search:
        mock_search.return_value = {
            "query": "How to control aphids in mustard crop? agriculture",
            "results": [
                {
                    "title": "Aphid Control in Mustard - TNAU",
                    "url": "https://agritech.tnau.ac.in/mustard-pest",
                    "snippet": "Spray dimethoate 30 EC or neem seed kernel extract for mustard aphids.",
                    "source": "agritech.tnau.ac.in",
                    "published_date": "Recent",
                    "is_static_fallback": False
                }
            ],
            "status": "ONLINE_SUCCESS",
            "provider": "DuckDuckGoProvider",
            "error_message": None
        }
        res = answer_question("How to control aphids in mustard crop?")
        assert res["predicted_topic"] in ("pest_control", "crop_cultivation")
        assert res["confidence"] > 0
        assert "agritech.tnau.ac.in" in res["answer"] or "dimethoate" in res["answer"]

