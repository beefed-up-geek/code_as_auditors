#!/usr/bin/env python3
"""
Normalize all violated article identifiers in PIPA case analyses using GPT-4o.

For every `violated_articles` entry in the dataset, the script calls GPT-4o with
the original `id` string and replaces it with a JSON object containing the law
name and normalized article reference while keeping the rest of the record
untouched.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "pipa_cases_analysis.json"
OUTPUT_FILE = BASE_DIR / "pipa_cases_analysis_normalized.json"
MAX_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2
MODEL_NAME = os.getenv("OPENAI_GPT_MODEL", "gpt-4o")

SYSTEM_PROMPT = (
    "You are a Korean legal analyst. Normalize article identifiers so that each "
    "entry clearly separates the law name (e.g., 개인정보보호법, 시행령, or other "
    "relevant statutes) from the article reference (e.g., '제24조의2 제1항', "
    "'제30조 제2항 제4호'). Return valid JSON only."
)


def load_env() -> None:
    """Load environment variables and ensure an OpenAI API key is present."""
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env file.")


def collect_all_articles(records: List[dict]) -> List[dict]:
    """Return a flat list of every violated_articles entry for processing."""
    articles: List[dict] = []
    for record in records:
        analysis = record.get("analysis")
        if not isinstance(analysis, dict):
            continue
        violated_articles = analysis.get("violated_articles")
        if not isinstance(violated_articles, list):
            continue
        for article in violated_articles:
            if isinstance(article, dict):
                articles.append(article)
    return articles


def build_id_prompt(article_id: str) -> str:
    """
    Build the user prompt for a single ID.

    The LLM must return JSON with `normalized_article` containing `original`,
    `law`, and `id`.
    """
    payload = json.dumps({"id": article_id}, ensure_ascii=False)
    return (
        "Normalize the following Korean law article identifier. "
        "Return ONLY JSON formatted exactly as:\n"
        '{\n'
        '  "normalized_article": {\n'
        '    "original": "<original id string>",\n'
        '    "law": "<law name in Korean>",\n'
        '    "id": "<normalized article reference in Korean>"\n'
        "  }\n"
        "}\n"
        "Rules:\n"
        "1. Preserve the entry exactly once.\n"
        "2. `law` must be the statute name (예: 개인정보보호법, 시행령, or other relevant law).\n"
        "3. `id` must contain only the article reference (예: 제24조의2 제1항, 제30조 제2항 제4호).\n"
        "4. Respond with valid JSON and nothing else.\n"
        "Input ID:\n"
        f"{payload}"
    )


def normalize_single_id(client: OpenAI, article_id: str) -> Dict[str, str]:
    """Call GPT-4o to normalize a single ID string."""
    prompt = build_id_prompt(article_id)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            parsed = json.loads(content)
            entry = parsed.get("normalized_article")
            if not isinstance(entry, dict):
                raise ValueError("Response JSON missing `normalized_article` object.")

            original = entry.get("original")
            law = entry.get("law")
            normalized_id = entry.get("id")
            if not isinstance(original, str) or original.strip() != article_id.strip():
                raise ValueError("Response `original` does not match input ID.")
            if not isinstance(law, str) or not isinstance(normalized_id, str):
                raise ValueError("Response missing `law` or `id` strings.")

            return {"law": law, "id": normalized_id}
        except (ValueError, json.JSONDecodeError) as exc:
            if attempt == MAX_RETRIES:
                raise RuntimeError("Failed to normalize article IDs.") from exc
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        except Exception as exc:  # tolerate transient API errors
            if attempt == MAX_RETRIES:
                raise RuntimeError("Failed to normalize article IDs.") from exc
            time.sleep(RETRY_BACKOFF_SECONDS * attempt)

    raise RuntimeError("Failed to normalize article IDs after retries.")


def normalize_articles_in_place(client: OpenAI, articles: List[dict]) -> int:
    """Normalize every violated article entry one by one, updating in place."""
    processed = 0
    for article in tqdm(articles, desc="Normalizing violated_articles", total=len(articles)):
        original_id = article.get("id")
        if not isinstance(original_id, str) or not original_id.strip():
            continue
        normalized = normalize_single_id(client, original_id)
        article["id"] = normalized
        processed += 1
    return processed


def main() -> int:
    """Entry point."""
    load_env()
    client = OpenAI()

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    with INPUT_FILE.open("r", encoding="utf-8") as handle:
        records: List[dict] = json.load(handle)

    articles = collect_all_articles(records)
    if not articles:
        raise RuntimeError("No violated_articles entries found in the input data.")

    processed_count = normalize_articles_in_place(client, articles)

    with OUTPUT_FILE.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    print(
        "Normalization complete. "
        f"Processed {processed_count} violated_articles entries across {len(records)} records. "
        f"Output written to {OUTPUT_FILE}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
