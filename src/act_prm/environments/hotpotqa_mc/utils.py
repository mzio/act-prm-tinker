"""
Helper functions for the HotpotQA multiple choice environment
"""

from typing import Any
import json

import numpy as np

from .prompts import render_prompt


def _process_sample_for_multiple_choice(
    sample: dict[str, Any],
    ambiguous_titles: bool = False,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """
    Collect titles and documents from sample.

    Args:
    - sample (dict[str, Any]): Sample from HotpotQA HF dataset
    - ambiguous_titles (bool): If True, we make titles ambiguous by replacing with "Title <idx>"

    Returns:
    - all_docs_dict (dict[str, str]): Dictionary of titles and their corresponding documents
    - all_titles (list[str]): List of all titles
    - all_docs (list[str]): List of all documents

    sample.keys() are:
    ['id', 'question', 'answer', 'type', 'level', 
     'supporting_facts', 'context', 'correctness', 'explanation'],
    """
    # Create lookup dictionary for all documents by title
    all_docs_dict = {}
    for title_idx, title in enumerate(sample["context"]["title"]):
        doc_text = "".join(sample["context"]["sentences"][title_idx])
        # Make ambiguous title if specified
        title = f"Title {title_idx:02d}" if ambiguous_titles else title
        all_docs_dict[title] = doc_text

    all_titles = list(all_docs_dict.keys())
    all_docs = list(all_docs_dict.values())
    
    return all_docs_dict, all_titles, all_docs


def process_sample(
    sample: dict[str, Any],
    ambiguous_titles: bool = False,
    include_titles_in_prompt: bool = True,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """
    Process HotpotQA HF dataset samples into our format
    """
    query_id = sample["id"]
    question = sample["question"]
    answer = sample["answer"]
    _, all_titles, all_docs = _process_sample_for_multiple_choice(
        sample,
        ambiguous_titles=ambiguous_titles,
    )
    sort_idx = np.argsort(all_titles)  # sort titles alphabetically
    all_titles = [all_titles[i] for i in sort_idx]
    all_docs = [all_docs[i] for i in sort_idx]
    prompt = render_prompt(question, all_titles, include_titles_in_prompt)
    return {
        "query_id": query_id,
        "question": question,
        "answer": answer,
        "all_titles": all_titles,
        "all_docs": all_docs,
        "prompt": prompt,
    }


def process_sample_from_gen_dataset(
    sample: dict[str, Any],
    ambiguous_titles: bool = False,
    include_titles_in_prompt: bool = True,
) -> tuple[dict[str, Any], list[str], list[dict[str, Any]]]:
    """
    Process generated HotpotQA HF dataset samples into consistent format
    """
    query_id = f"{sample["sample_idx"]}_{sample["generation_idx"]}"
    question = sample["final_question"]
    answer = sample["final_answer"]
    all_docs_dict = json.loads(sample["all_docs_dict"])
    all_titles = list(all_docs_dict.keys())
    all_docs = list(all_docs_dict.values())
    
    if ambiguous_titles:
        all_titles = [f"Title {i:02d}" for i in range(len(all_titles))]
    sort_idx = np.argsort(all_titles)  # sort titles alphabetically
    all_titles = [all_titles[i] for i in sort_idx]
    all_docs = [all_docs[i] for i in sort_idx]
    
    prompt = render_prompt(question, all_titles, include_titles_in_prompt)
    return {
        "query_id": query_id,
        "question": question,
        "answer": answer,
        "all_titles": all_titles,
        "all_docs": all_docs,
        "prompt": prompt,
    }
