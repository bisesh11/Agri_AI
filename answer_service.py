"""Agricultural Answer Generation Module.

Transforms retrieved agricultural evidence into a coherent, practical answer.
Supports:
  - Mode A: Deterministic, high-precision Extractive Summarization (Zero LLM needed).
  - Mode B: Optional Grounded LLM Summarization with fallback to Mode A.
"""

import os
import re
import logging
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("answer_service")


def clean_sentence(sentence: str) -> str:
    """Normalizes whitespace and removes trailing punctuation artifacts."""
    s = re.sub(r"\s+", " ", sentence).strip()
    return s


def score_sentence_relevance(sentence: str, query_tokens: set) -> float:
    """Scores sentence relevance based on keyword match and information density."""
    sent_tokens = set(re.findall(r"\b\w{3,}\b", sentence.lower()))
    if not sent_tokens:
        return 0.0

    overlap = sent_tokens.intersection(query_tokens)
    score = len(overlap) / (len(sent_tokens) ** 0.5)

    # Prefer informative sentences of reasonable length (between 8 and 35 words)
    words = sentence.split()
    if 10 <= len(words) <= 35:
        score *= 1.2
    elif len(words) < 6:
        score *= 0.3

    return score


def extractive_summarization(
    question: str,
    predicted_topic: str,
    sources: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """Generates an evidence-grounded answer using extractive ranking of retrieved content.

    Requires NO generative AI / LLM API. Works 100% locally and deterministically.
    """
    if not sources:
        msg = (
            "No online agricultural evidence could be retrieved for this inquiry. "
            "Please verify your internet connection or check the recommended official "
            "agricultural portals below."
        )
        return msg, []

    # Extract informative query tokens
    query_tokens = set(re.findall(r"\b\w{3,}\b", question.lower()))

    # Gather candidate passages from all sources
    candidate_passages = []
    active_sources = []

    for idx, src in enumerate(sources, start=1):
        # Gather text from snippet and full text (if available)
        combined_text = (src.get("snippet", "") + " " + src.get("text", "")).strip()
        if not combined_text:
            continue

        active_sources.append(src)
        # Split into sentences
        sentences = re.split(r"(?<=[.!?])\s+", combined_text)

        for s in sentences:
            cleaned = clean_sentence(s)
            if len(cleaned.split()) >= 6:
                score = score_sentence_relevance(cleaned, query_tokens)
                if score > 0.15:
                    candidate_passages.append({
                        "text": cleaned,
                        "score": score,
                        "source_idx": idx,
                        "source_title": src.get("title", f"Source {idx}"),
                        "url": src.get("url", "")
                    })

    # Sort passages by relevance score
    candidate_passages.sort(key=lambda x: x["score"], reverse=True)

    # Deduplicate similar passages
    selected_passages = []
    seen_texts = set()

    for p in candidate_passages:
        # Check text similarity
        simple_text = "".join(p["text"].lower().split()[:6])
        if simple_text in seen_texts:
            continue
        seen_texts.add(simple_text)
        selected_passages.append(p)
        if len(selected_passages) >= 4:
            break

    # Build Structured Output
    if not selected_passages:
        # If specific sentence extraction yielded low overlap, use top snippets directly
        bullet_points = []
        for i, src in enumerate(sources[:3], start=1):
            snip = clean_sentence(src.get("snippet", ""))
            if snip:
                bullet_points.append(f"- {snip} [[Source {i}]({src.get('url')})]")

        if bullet_points:
            answer_text = (
                f"Based on agricultural sources retrieved for **{predicted_topic.replace('_', ' ').title()}**:\n\n"
                + "\n".join(bullet_points)
                + "\n\n*Advisory: Always cross-check symptoms and dosages with your local Krishi Vigyan Kendra (KVK) or agriculture officer before applying treatments.*"
            )
            return answer_text, sources[:3]
        else:
            return (
                "The retrieved online sources did not contain enough specific evidence to formulate a complete answer. "
                "Please consult the verified agricultural portals cited below.",
                sources
            )

    # Format synthesized response
    summary_sentence = selected_passages[0]["text"]
    key_points = []
    for p in selected_passages[1:]:
        key_points.append(f"- {p['text']} [[{p['source_idx']}]({p['url']})]")

    # If only 1 passage was extracted
    if not key_points:
        key_points.append(f"- Consult the cited resource for complete application schedules and localized recommendations.")

    answer_parts = [
        f"**Direct Finding ({predicted_topic.replace('_', ' ').title()}):**",
        f"{summary_sentence} [[{selected_passages[0]['source_idx']}]({selected_passages[0]['url']})]\n",
        "**Key Advisory & Management Practices:**",
        "\n".join(key_points),
        "\n*Agricultural Advisory: Dosages and schedules can vary based on local agro-climatic conditions and soil test reports. Always verify with official extension services.*"
    ]

    return "\n\n".join(answer_parts), active_sources


def llm_grounded_summarization(
    question: str,
    predicted_topic: str,
    sources: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]]]:
    """Generates an answer using an LLM strictly grounded in retrieved evidence.

    Falls back to extractive_summarization if no API key is set or on error.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    generic_key = os.getenv("LLM_API_KEY")

    if not (gemini_key or openai_key or generic_key):
        # LLM not configured -> use extractive mode
        return extractive_summarization(question, predicted_topic, sources)

    # Prepare evidence context
    evidence_blocks = []
    for idx, s in enumerate(sources, start=1):
        title = s.get("title", "")
        url = s.get("url", "")
        content = s.get("snippet", "") + " " + s.get("text", "")
        evidence_blocks.append(f"[{idx}] Title: {title}\nURL: {url}\nContent: {content[:1000]}")

    evidence_text = "\n\n".join(evidence_blocks)

    prompt = f"""You are an expert Agricultural Extension Officer assistant.
Answer the farmer's question strictly and ONLY using the provided retrieved evidence.

Farmer's Question: {question}
Predicted Topic: {predicted_topic}

RETRIEVED EVIDENCE:
{evidence_text}

STRICT INSTRUCTIONS:
1. Ground every claim directly in the evidence. Include citations like [1], [2] referencing the corresponding evidence item.
2. DO NOT hallucinate, extrapolate, or invent chemical names, dosages, yields, or statistics not present in the evidence.
3. If the evidence is insufficient or does not contain a conclusive answer, explicitly state: "The retrieved agricultural evidence does not provide enough detail on this aspect."
4. Format the answer with clear bullet points and a practical advisory tone.
5. End with an agricultural advisory note reminding the farmer to follow local agricultural university package of practices.
"""

    try:
        if gemini_key:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model_name = os.getenv("LLM_MODEL", "gemini-1.5-flash")
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            if response and response.text:
                return response.text.strip(), sources

        elif openai_key or generic_key:
            import requests
            api_key = openai_key or generic_key
            api_url = os.getenv("LLM_API_BASE", "https://api.openai.com/v1/chat/completions")
            model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

            payload = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful, evidence-grounded agricultural advisor."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2
            }
            resp = requests.post(
                api_url,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json=payload,
                timeout=12
            )
            if resp.status_code == 200:
                answer = resp.json()["choices"][0]["message"]["content"]
                return answer.strip(), sources

    except Exception as e:
        logger.warning(f"LLM summarization failed: {e}. Falling back to extractive mode.")

    # Seamless fallback
    return extractive_summarization(question, predicted_topic, sources)


def generate_answer(
    question: str,
    predicted_topic: str,
    sources: List[Dict[str, Any]],
    mode: str = "auto"
) -> Tuple[str, List[Dict[str, Any]]]:
    """Master entry point for agricultural answer synthesis.

    Args:
        question: Farmer's question.
        predicted_topic: Topic classified by Naive Bayes.
        sources: Retrieved online search results and snippets.
        mode: 'auto' (detects if LLM key is configured, else extractive), 'extractive', or 'llm'.

    Returns:
        Tuple of (formatted_answer_string, list_of_cited_sources)
    """
    if mode == "extractive":
        return extractive_summarization(question, predicted_topic, sources)
    elif mode == "llm":
        return llm_grounded_summarization(question, predicted_topic, sources)
    else:
        # Auto mode: use LLM if key exists, otherwise extractive
        if any(os.getenv(k) for k in ("GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_API_KEY")):
            return llm_grounded_summarization(question, predicted_topic, sources)
        return extractive_summarization(question, predicted_topic, sources)

