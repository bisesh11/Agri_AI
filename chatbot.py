"""Agricultural Chatbot Core Module.

Coordinates question preprocessing, Multinomial Naive Bayes topic classification,
confidence estimation, query routing (specialized live services vs online search),
web content fetching, evidence-based answer generation, and source citations.
"""

import os
import re
import logging
from typing import Dict, Any, Optional
import joblib
import numpy as np

from preprocessing import TextPreprocessor
from search_service import search_agricultural_web, generate_search_query, OFFICIAL_STATIC_PORTALS
from source_fetcher import fetch_page_content
from answer_service import generate_answer
from weather_service import get_live_weather
from market_service import get_mandi_prices, extract_commodity_from_text, format_mandi_price_markdown

logger = logging.getLogger("chatbot")

MODEL_PATH = os.path.join("models", "trained_pipeline.joblib")
DEFAULT_CONFIDENCE_THRESHOLD = 0.12  # Benchmark threshold across 16 classes (random chance = 0.0625)

_CACHED_PIPELINE = None


def load_classifier():
    """Loads the serialized Scikit-learn Pipeline (TextPreprocessor + Vectorizer + MultinomialNB)."""
    global _CACHED_PIPELINE
    if _CACHED_PIPELINE is not None:
        return _CACHED_PIPELINE

    if not os.path.exists(MODEL_PATH):
        logger.info(f"Model artifact not found at {MODEL_PATH}. Initiating automatic training...")
        from train_model import train
        _CACHED_PIPELINE = train()
    else:
        _CACHED_PIPELINE = joblib.load(MODEL_PATH)

    return _CACHED_PIPELINE


def extract_location_hint(text: str) -> Optional[str]:
    """Attempts to extract common Indian city or district names mentioned in a query."""
    common_cities = [
        "delhi", "mumbai", "pune", "nashik", "nagpur", "coimbatore", "chennai", "madurai",
        "ludhiana", "amritsar", "karnal", "hisar", "jaipur", "bharatpur", "indore", "bhopal",
        "lucknow", "varanasi", "patna", "ranchi", "hyderabad", "nizamabad", "bengaluru",
        "mysuru", "kolkata", "ahmedabad", "rajkot", "surat", "guntur", "vijayawada"
    ]
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    for w in words:
        if w in common_cities:
            return w.title()
    # Check for "in <Location>" pattern
    match = re.search(r"\bin\s+([a-zA-Z]{3,}(?:\s+[a-zA-Z]{3,})?)", text, re.I)
    if match:
        candidate = match.group(1).strip()
        if candidate.lower() not in ("soil", "field", "wheat", "rice", "plants", "crop", "winter", "summer"):
            return candidate.title()
    return None


def answer_question(
    question: str,
    location: Optional[str] = None,
    user_mode: str = "auto",
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
) -> Dict[str, Any]:
    """End-to-End Question Answering Pipeline.

    Args:
        question: Farmer's natural language question.
        location: Optional location context (town, district, state).
        user_mode: 'auto', 'extractive', or 'llm'.
        confidence_threshold: Cutoff below which classifier is deemed uncertain.

    Returns:
        Structured dictionary containing answer, predicted_topic, confidence,
        sources, search_query, and diagnostic metadata.
    """
    if not question or not isinstance(question, str) or not question.strip():
        return {
            "question": "",
            "predicted_topic": "unknown",
            "confidence": 0.0,
            "is_confident": False,
            "answer": "Please enter a valid agricultural question.",
            "sources": [],
            "search_query": "",
            "search_status": "INVALID_INPUT",
            "provider": "None",
            "retrieval_error": "Input question was empty or invalid."
        }

    raw_question = question.strip()
    if len(raw_question) > 1000:
        raw_question = raw_question[:1000]

    # Load Pipeline
    pipeline = load_classifier()

    # Predict Topic & Estimate Confidence
    predicted_topic = pipeline.predict([raw_question])[0]
    probabilities = pipeline.predict_proba([raw_question])[0]
    confidence = float(np.max(probabilities))
    is_confident = confidence >= confidence_threshold

    # Fallback to general agriculture if confidence is too low
    effective_topic = predicted_topic if is_confident else "general_agriculture"

    # ROUTING 1: Weather Inquiries
    if effective_topic == "weather" or any(w in raw_question.lower() for w in ("weather", "rainfall", "forecast", "monsoon", "temperature")):
        loc = location or extract_location_hint(raw_question)
        if loc:
            weather_res = get_live_weather(loc)
            if weather_res.get("status") == "LIVE_SUCCESS":
                lines = [
                    f"### 🌦️ Live Agricultural Weather Report: {weather_res['location']}",
                    f"- **Current Condition:** {weather_res['current_condition']}",
                    f"- **Current Temperature:** {weather_res['current_temperature']}°C",
                    f"- **Relative Humidity:** {weather_res['humidity']}%",
                    f"- **Wind Speed:** {weather_res['wind_speed_kmh']} km/h\n",
                    "**🌾 Agricultural Weather Advisory:**"
                ]
                for adv in weather_res.get("agricultural_advisory", []):
                    lines.append(f"- {adv}")

                lines.append(f"\n*Source: [{weather_res['source']}]({weather_res['source_url']})*")
                return {
                    "question": raw_question,
                    "predicted_topic": "weather",
                    "confidence": confidence,
                    "is_confident": is_confident,
                    "answer": "\n".join(lines),
                    "sources": [{
                        "title": f"Live Weather Data for {weather_res['location']}",
                        "url": weather_res["source_url"],
                        "source": "open-meteo.com",
                        "snippet": f"Temperature: {weather_res['current_temperature']}°C, Condition: {weather_res['current_condition']}",
                        "published_date": "Live Observation"
                    }],
                    "search_query": f"Live weather API for {loc}",
                    "search_status": "LIVE_SUCCESS",
                    "provider": "Open-Meteo Meteorological Service",
                    "retrieval_error": None
                }

    # ROUTING 2: Mandi Market Prices
    if effective_topic == "market_prices" or any(w in raw_question.lower() for w in ("mandi", "market price", "rate per quintal", "msp")):
        comm = extract_commodity_from_text(raw_question)
        if comm:
            price_res = get_mandi_prices(commodity=comm)
            if price_res.get("status") == "SUCCESS":
                answer_md = format_mandi_price_markdown(price_res)
                sources = [{
                    "title": f"Agmarknet Wholesale Mandi Prices ({comm})",
                    "url": price_res["source_url"],
                    "source": "agmarknet.gov.in",
                    "snippet": f"Official daily modal price records for {comm} across major APMC markets.",
                    "published_date": "Current Daily Bulletin"
                }]
                return {
                    "question": raw_question,
                    "predicted_topic": "market_prices",
                    "confidence": confidence,
                    "is_confident": is_confident,
                    "answer": answer_md,
                    "sources": sources,
                    "search_query": f"Mandi rates for {comm}",
                    "search_status": "LIVE_SUCCESS",
                    "provider": "Agmarknet Registry",
                    "retrieval_error": None
                }

    # ROUTING 3: General Online Information Retrieval
    search_topic = effective_topic if is_confident else None
    search_result = search_agricultural_web(
        raw_question,
        predicted_topic=search_topic,
        max_results=4
    )

    sources = search_result.get("results", [])
    search_status = search_result.get("status", "UNKNOWN")
    provider = search_result.get("provider", "SearchProvider")
    error_msg = search_result.get("error_message")

    # Attempt page content extraction for top 2 online sources to enrich evidence
    if search_status in ("ONLINE_SUCCESS", "CACHE_HIT") and sources:
        for s in sources[:2]:
            url = s.get("url")
            if url and not s.get("is_static_fallback", False):
                page_data = fetch_page_content(url, timeout=4, max_chars=3000)
                if page_data and page_data.get("text"):
                    s["text"] = page_data["text"]
                    if page_data.get("published_date") and page_data["published_date"] != "Not specified":
                        s["published_date"] = page_data["published_date"]

    # Synthesize evidence-grounded answer
    answer_text, cited_sources = generate_answer(
        raw_question,
        effective_topic,
        sources,
        mode=user_mode
    )

    # If classification was uncertain, append honest clarification advice
    if not is_confident:
        answer_text += (
            f"\n\n*Note on Classification: The topic prediction ('{predicted_topic}') had low statistical confidence "
            f"({confidence * 100:.1f}%). A general agricultural search was conducted.*"
        )

    return {
        "question": raw_question,
        "predicted_topic": predicted_topic,
        "confidence": confidence,
        "is_confident": is_confident,
        "answer": answer_text,
        "sources": sources,
        "search_query": search_result.get("query", raw_question),
        "search_status": search_status,
        "provider": provider,
        "retrieval_error": error_msg
    }

