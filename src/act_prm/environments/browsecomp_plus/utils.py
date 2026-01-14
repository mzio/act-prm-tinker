"""
Helper functions for the multiple choice environment
"""

import json
from typing import Any

import numpy as np
from transformers import PreTrainedTokenizerBase


def get_title(sample_text: str) -> str | None:
    """
    Get the title from the sample text
    """
    title_splits = sample_text.split("---\ntitle: ")[-1].split("\n")
    if len(title_splits) > 1:
        return title_splits[0]
    return None


def process_batch_for_search(
    batch: dict[str, list[str]],
    tokenizer: PreTrainedTokenizerBase,
    doc_chunk_size: int = 512,
    **kwargs: Any,
) -> dict[str, list[str]]:
    """
    Process HF Dataset batches for search
    -> Split text into chunks, add title, add preview text
    """
    new_batch = {
        "doc_id": [],
        "url": [],
        "title": [],
        "text": [],
        "title_and_text": [],
        "past_scroll_id": [],
        "next_scroll_id": [],
    }
    for text_idx, text in enumerate(batch["text"]):
        doc_id = f"{batch["docid"][text_idx]}_0"
        url = batch["url"][text_idx]
        title = get_title(text) or f"{text.split('\n')[0][:32]} [...]"
        # Split text into chunks
        tokens = tokenizer.encode(text)
        token_chunks = [
            tokens[i:i + doc_chunk_size]
            for i in range(0, len(tokens), doc_chunk_size)
        ]
        if len(token_chunks[-1]) < doc_chunk_size:
            # Make the last chunk overlap with the previous one
            token_chunks[-1] = tokens[-doc_chunk_size:]
        # Decode back into text
        text_chunks = tokenizer.batch_decode(
            token_chunks, skip_special_tokens=True,
        )
        for chunk_idx, text_chunk in enumerate(text_chunks):
            new_batch["title"].append(title)
            new_batch["text"].append(text_chunk)
            new_batch["title_and_text"].append(f"## {title}\n\n{text_chunk}")
            # new_batch["past_scroll_id"].append(idx - 1)
            # new_batch["next_scroll_id"].append(idx + 1)
            new_batch["past_scroll_id"].append(f"{batch["docid"][text_idx]}_{chunk_idx - 1}")
            new_batch["next_scroll_id"].append(f"{batch["docid"][text_idx]}_{chunk_idx + 1}")
            if chunk_idx == 0:
                new_batch["past_scroll_id"][-1] = None
            elif chunk_idx == len(text_chunks) - 1:
                new_batch["next_scroll_id"][-1] = None
            # Add back old metadata
            doc_id = f"{batch["docid"][text_idx]}_{chunk_idx}"
            new_batch["doc_id"].append(doc_id)
            new_batch["url"].append(url)

    return new_batch


def process_sample_for_multiple_choice(
    sample: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_distractors: int | None = None,
    max_docs: int | None = None,
    max_preview_chars: int = 32,
    doc_chunk_size: int = 512,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Process a sample into our multiple choice format
    """
    search_results = []  # list of docs initially shown
    doc_dict = {}        # where we store all docs
    num_docs = 0
    num_distractors = 0
    # max_distractors = max_distractors or # len(sample["negative_docs"])
    if max_distractors is None and max_docs is not None:
        max_distractors = max(0, max_docs - len(sample["gold_docs"]))
    else:
        max_distractors = max_distractors or len(sample["negative_docs"])
    _sample = {
        k: v for k, v in sample.items() if k in ["gold_docs", "negative_docs"]
    }
    # Get potential docs
    for k, v in _sample.items():
        np.random.shuffle(v)
        for _, doc in enumerate(v):
            if k == "gold_docs" or num_distractors < max_distractors:
                # Only show preview and split on new word
                preview_text = doc["text"][:max_preview_chars]
                preview_text = f"{preview_text.rsplit(" ", 1)[0]} [...]"
                search_results.append({
                    "doc_id": f"{doc["docid"]}_0",
                    "text_preview": preview_text,
                })
                num_docs += 1
                if k != "gold_docs":
                    num_distractors += 1

                # Split text into chunks
                tokens = tokenizer.encode(doc["text"])
                token_chunks = [
                    tokens[i:i+doc_chunk_size]
                    for i in range(0, len(tokens), doc_chunk_size)
                ]
                if len(token_chunks[-1]) < doc_chunk_size:  # make last chunk same size
                    token_chunks[-1] = tokens[-doc_chunk_size:]
                # Decode back into text
                text_chunks = tokenizer.batch_decode(
                    token_chunks, skip_special_tokens=True,
                )
                # Add chunks to doc_dict
                for idx, text_chunk in enumerate(text_chunks):
                    _doc_id = f"{doc["docid"]}_{idx}"
                    doc_dict[_doc_id] = {
                        "doc_id": _doc_id,
                        "url": doc["url"],
                        "text": text_chunk,
                        "past_scroll_id": f"{doc["docid"]}_{idx - 1}",
                        "next_scroll_id": f"{doc["docid"]}_{idx + 1}",
                    }
                    if idx == 0:
                        doc_dict[_doc_id]["past_scroll_id"] = None
                    if idx == len(text_chunks) - 1:
                        doc_dict[_doc_id]["next_scroll_id"] = None

    doc_ids = list([d["doc_id"] for d in search_results])
    sorting_idx = sorted(range(len(doc_ids)), key=lambda x: doc_ids[x])
    search_results = [search_results[i] for i in sorting_idx]
    return search_results, doc_dict


def render_prompt(
    query: str,
    search_results: list[dict[str, Any]] | None = None,
) -> str:
    """
    Render prompt for BrowseComp-Plus

    Returns:
    - prompt (str): The prompt to be used for the model
    """
    initial_msg = (
        "# Instructions\n\nGiven a list of search results, think and use the available"
        f" tools to answer the following question:\n'''\n{query}\n'''\n"
        "\nYou **MUST** use the available tools to answer the question."
        " Otherwise, you'll most likely fail."
        " Once you have found the answer, provide your final answer as a concise sentence,"
        " in the following format: 'Final Answer: <put your answer here>'."
    )
    # We initialize without search results, so just show initial instruction prompt
    if search_results is None:
        return initial_msg
    # Otherwise, add search results and instruction reminder (ala lost-in-the-middle)
    search_msg = f"# Search Results:\n\n{json.dumps(search_results, indent=2)}"
    final_msg = (
        "# Instructions (again)\n\nNow answer the original question."
        f" Recall the question is:\n'''\n{query}\n'''\n\n"
        "Once you have found the answer, provide your final answer as a concise sentence,"
        " in the following format: 'Final Answer: <put your answer here>'."
    )
    return f"{initial_msg}\n\n{search_msg}\n\n{final_msg}"
