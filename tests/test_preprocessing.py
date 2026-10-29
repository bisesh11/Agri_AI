"""Test Text Preprocessing Module."""

import pytest
from preprocessing import clean_text, TextPreprocessor, AGRICULTURAL_KEYWORDS_PRESERVE


def test_clean_text_normal_case():
    text = "What is the best fertilizer dose for wheat cultivation?"
    cleaned = clean_text(text)
    assert "fertilizer" in cleaned
    assert "wheat" in cleaned
    # Common stop words like 'what', 'is', 'the' should be stripped
    assert not cleaned.startswith("what")


def test_clean_text_preserves_agri_keywords():
    for word in ["npk", "urea", "dap", "rice", "blight", "rust", "drip", "pesticides"]:
        cleaned = clean_text(f"How to use {word} properly?")
        assert word in cleaned, f"Expected domain term '{word}' to be preserved in '{cleaned}'"


def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\t  ") == ""


def test_clean_text_punctuation_handling():
    text = "Pest-control in tomato/chili crop: how to spray?"
    cleaned = clean_text(text)
    assert "-" not in cleaned
    assert "/" not in cleaned
    assert ":" not in cleaned
    assert "?" not in cleaned
    assert "pest" in cleaned
    assert "tomato" in cleaned


def test_clean_text_max_length_truncation():
    very_long = "wheat " * 500
    cleaned = clean_text(very_long, max_len=100)
    assert len(cleaned) <= 100


def test_text_preprocessor_transformer():
    preprocessor = TextPreprocessor(max_len=500)
    input_list = [
        "How to prepare jeevamrut?",
        "When to irrigate wheat?"
    ]
    transformed = preprocessor.transform(input_list)
    assert len(transformed) == 2
    assert isinstance(transformed, list)
    assert "jeevamrut" in transformed[0]
    assert "wheat" in transformed[1]

