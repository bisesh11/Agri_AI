"""Test Data Validation and Schema Integrity."""

import os
import pandas as pd
import pytest


def test_dataset_exists():
    assert os.path.exists("agriculture_dataset.csv"), "agriculture_dataset.csv must exist"


def test_dataset_schema():
    df = pd.read_csv("agriculture_dataset.csv")
    assert "question" in df.columns, "Column 'question' missing in dataset"
    assert "topic" in df.columns, "Column 'topic' missing in dataset"


def test_dataset_sufficiency():
    df = pd.read_csv("agriculture_dataset.csv")
    assert len(df) >= 300, f"Expected at least 300 questions, found {len(df)}"
    num_topics = df["topic"].nunique()
    assert num_topics >= 12, f"Expected at least 12 unique topics, found {num_topics}"


def test_dataset_no_nulls():
    df = pd.read_csv("agriculture_dataset.csv")
    assert df["question"].isnull().sum() == 0, "Found null questions in dataset"
    assert df["topic"].isnull().sum() == 0, "Found null topics in dataset"
    # Ensure no empty strings
    empty_q = df[df["question"].str.strip() == ""]
    assert len(empty_q) == 0, "Found empty string questions in dataset"


def test_model_artifact_exists():
    model_path = os.path.join("models", "trained_pipeline.joblib")
    assert os.path.exists(model_path), f"Trained model artifact missing at {model_path}"

