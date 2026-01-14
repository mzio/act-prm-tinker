"""
Utility functions for LongBench environment
- Mostly around chunking and tokenizing documents
"""

from typing import Any

from transformers import PreTrainedTokenizerBase


def chunk_tokens(
    tokens: list[int],
    chunk_size: int = 2048,
    overlap: int = 256,
) -> list[list[int]]:
    """
    Split a list of tokens into overlapping chunks.

    Example:
      chunk_size = 2048
      overlap    = 256
      stride     = 1792

    So chunk 0: tokens[0:2048]
       chunk 1: tokens[1792:3840]
       etc.
    """
    assert overlap < chunk_size
    stride = chunk_size - overlap

    chunks = []
    for start in range(0, len(tokens), stride):
        end = start + chunk_size
        chunk = tokens[start:end]
        if not chunk:
            break
        chunks.append(chunk)
        if end >= len(tokens):
            break

    return chunks


def chunk_text_by_tokens(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = 2048,
    overlap: int = 256,
) -> tuple[list[list[int]], list[str]]:
    """
    Split text into chunks by tokens, handling long paragraphs
    """
    # Split into paragraphs using blank lines
    paragraphs = [f"{p}\n" for p in text.split("\n") if p.strip()]

    chunks = []
    current_tokens = []

    for para in paragraphs:
        para_tokens = tokenizer.encode(para, add_special_tokens=False)

        # If this single paragraph is huge, fall back to plain token chunking
        if len(para_tokens) > chunk_size:
            # flush current chunk first
            if current_tokens:
                chunks.append(current_tokens)
                current_tokens = []

            big_para_chunks = chunk_tokens(para_tokens, chunk_size, overlap)
            chunks.extend(big_para_chunks)
            continue

        # If adding this paragraph would overflow, finalize current chunk
        if len(current_tokens) + len(para_tokens) > chunk_size:
            chunks.append(current_tokens)
            # start new chunk, possibly with overlap from previous
            if overlap > 0 and len(current_tokens) > 0:
                current_tokens = current_tokens[-overlap:]
            else:
                current_tokens = []

        current_tokens.extend(para_tokens)

    if current_tokens:
        chunks.append(current_tokens)

    text_chunks = tokenizer.batch_decode(chunks, skip_special_tokens=False)
    return chunks, text_chunks


def convert_text_chunks_to_dicts(text_chunks: list[str]) -> list[dict[str, Any]]:
    """
    Convert a list of text chunks to a (linked) list of `doc_dict` dictionaries
    """
    return [
        {
            "text": chunk,
            "chunk_idx": chunk_idx,
            "next_chunk_idx": chunk_idx + 1 if chunk_idx < len(text_chunks) - 1 else None,
            "prev_chunk_idx": chunk_idx - 1 if chunk_idx > 0 else None,
        }
        for chunk_idx, chunk in enumerate(text_chunks)
    ]
