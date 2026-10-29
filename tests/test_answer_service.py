"""Test Agricultural Answer Generation Service."""

import pytest
from answer_service import clean_sentence, score_sentence_relevance, extractive_summarization, generate_answer


def test_clean_sentence():
    raw = "  This   is an    agricultural advisory.   \n\n "
    assert clean_sentence(raw) == "This is an agricultural advisory."


def test_score_sentence_relevance():
    query_tokens = {"wheat", "rust", "yellow"}
    sent1 = "Yellow rust in wheat is caused by Puccinia striiformis."
    sent2 = "The tractor needs routine engine oil maintenance."
    score1 = score_sentence_relevance(sent1, query_tokens)
    score2 = score_sentence_relevance(sent2, query_tokens)
    assert score1 > score2


def test_extractive_summarization_with_sources():
    sources = [
        {
            "title": "Wheat Disease Guide",
            "url": "https://icar.gov.in/wheat-rust",
            "snippet": "Yellow rust in wheat appears as yellow pustules on leaves. Spray propiconazole at 0.1 percent.",
            "source": "icar.gov.in"
        }
    ]
    answer, cited = extractive_summarization("How to treat yellow rust in wheat?", "plant_disease", sources)
    assert "Direct Finding" in answer
    assert "icar.gov.in" in answer or "https://icar.gov.in/wheat-rust" in answer
    assert len(cited) >= 1


def test_extractive_summarization_empty_sources():
    answer, cited = extractive_summarization("Unknown question?", "general_agriculture", [])
    assert "No online agricultural evidence could be retrieved" in answer
    assert cited == []


def test_generate_answer_mode_fallback():
    sources = [
        {
            "title": "Organic Compost Advisory",
            "url": "https://agritech.tnau.ac.in/org-farming",
            "snippet": "Jeevamrut is prepared using fresh cow dung, cow urine, jaggery, and pulse flour.",
            "source": "agritech.tnau.ac.in"
        }
    ]
    # Without valid LLM key in test environment, 'auto' should default safely to extractive
    ans, cited = generate_answer("How to make jeevamrut?", "organic_farming", sources, mode="auto")
    assert "Direct Finding" in ans or "Key Advisory" in ans
    assert len(cited) > 0

