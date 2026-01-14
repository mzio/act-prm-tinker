"""
Search tool for SlackChat QA Environment
"""

from os.path import join
from typing import Any
import json
import logging

import numpy as np
import bm25s
from bm25s import BM25
from Stemmer import Stemmer
from rich import print as rich_print
from transformers import PreTrainedTokenizerBase

from ...base import BaseTool


logger = logging.getLogger(__name__)


RESULT_TEMPLATE = """## Document View:
'''
{document}
'''{scroll_message}
"""


def build_retriever(
    retriever_name: str,
    corpus: list[dict[str, Any]],
    stemmer: Stemmer | None = None,
    retriever_config: dict[str, Any] | None = None,
    stopwords: str = "en",
    save_path: str | None = None,
) -> BM25 | None:
    """
    Build BM25 retriever from corpus and stemmer
    """
    try:
        save_path = save_path or ""
        retriever = BM25.load(save_path, load_corpus=True)
        rich_print(f"-> Loaded retriever from [bright_blue]{save_path}[/bright_blue]!")
        return retriever
    except FileNotFoundError:
        rich_print(
            f"-> No retriever found at [bright_blue]{save_path}[/bright_blue]. "
            "Building new retriever..."
        )

    if retriever_name.startswith("bm25"):
        # Lowercase corpus for BM25
        corpus = [c["text"].lower() for c in corpus]
        corpus_tokens = bm25s.tokenize(corpus, stopwords=stopwords, stemmer=stemmer)
        # Create the BM25 model and index the corpus
        retriever = (
            bm25s.BM25(**retriever_config) if retriever_config is not None else bm25s.BM25()
        )
    else:
        raise NotImplementedError(f"Sorry, retriever {retriever_name} not implemented")

    # Index and save corpus
    retriever.index(corpus_tokens)
    if save_path is not None:
        retriever.save(save_path, corpus=corpus)
    return retriever


class SearchTool(BaseTool):
    """
    Search tool using BM25 ranking function (lexical matching)
    """

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizerBase,
        retriever_name: str = "bm25_index",
        use_stemmer: bool = True,
        retriever_config: dict[str, Any] | None = None,
        top_k: int = 5,
        max_preview_tokens: int = 204,  # about 1024 tokens overall
        save_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.corpus = corpus
        self.tokenizer = tokenizer

        self.retriever_name = retriever_name
        self.stemmer = Stemmer("english") if use_stemmer else None  
        self.retriever_config = retriever_config
        self.top_k = top_k
        self.max_preview_tokens = max_preview_tokens
        
        if save_path is not None:
            save_path = join(save_path, f"{retriever_name}")
        self.retriever = build_retriever(
            retriever_name=retriever_name,
            corpus=corpus,
            stemmer=self.stemmer,
            retriever_config=self.retriever_config,
            stopwords="en",
            save_path=save_path,
        )

    def __call__(
        self,
        query: str,
        **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], str]:
        """
        Search the corpus for top document based on the given query

        Returns:
        - top-k document dicts
        - string representation of the top-k document dicts
        """
        try:
            # Query the corpus
            bm25_query_tokens = bm25s.tokenize(query.lower(), self.stemmer)
            results, scores = self.retriever.retrieve(bm25_query_tokens, k=self.top_k)
            
            # For each top-k result, get the preview text
            llm_query_input_ids = self.tokenizer(query)["input_ids"]
            topk_result_preview: list[dict] = []

            for i in range(results.shape[1]):
                result_i, _ = results[0, i], scores[0, i]  # for now, don't show score
                doc_dict = self.corpus[result_i["id"]]
                result_preview = self._format_result_preview(
                    doc_dict, i, llm_query_input_ids
                )
                topk_result_preview.append(result_preview)
            result_str = json.dumps(topk_result_preview, indent=2)

        except Exception as e:
            result_str = f"error: {str(e)}"
            topk_result_preview = {}
            logger.error(f"SearchTool error: {e}")
            breakpoint()

        return topk_result_preview, result_str

    def _format_result_preview(
        self,
        retrieved_doc: dict[str, Any],
        retrieve_rank: int,
        query_input_ids: list[int],
        retriever_score: float | None = None,
    ) -> dict:
        """
        Format the result preview for a message
        """
        doc_input_ids = self.tokenizer(retrieved_doc["text"])["input_ids"]
        doc_title = retrieved_doc["title"]
        if len(doc_input_ids) > self.max_preview_tokens:
            _start_idx, _end_idx, tokens_span = best_window_total_hits_np(
                doc_input_ids, query_input_ids, window=self.max_preview_tokens
            )
            doc_content = self.tokenizer.decode(
                tokens_span, skip_special_tokens=True
            ).strip()
            # Decorate (split on words, add ellipsis prefix / suffix)
            _doc_content_delim = doc_content.split(" ")
            _doc_prefix, _doc_suffix = "", ""
            if _end_idx < len(doc_input_ids):
                _doc_content_delim = _doc_content_delim[:-1]
                _doc_suffix = " [...Expand for more]"
            if _start_idx > 0:
                _doc_content_delim = _doc_content_delim[1:]
                _doc_prefix = "[Expand for more...] "
            doc_content = f"{_doc_prefix}{' '.join(_doc_content_delim)}{_doc_suffix}"
        else:
            doc_content = retrieved_doc["text"]

        # Make retriever score JSON serializable as string
        retriever_score = str(retriever_score) if retriever_score is not None else ""
        return {
            "doc_id": retrieved_doc["doc_id"],
            "title": doc_title,
            "text_preview": doc_content,
            "retriever_rank": retrieve_rank + 1,  # 1-indexed
            "retriever_score": retriever_score,
            # "url": retrieved_doc["url"],
            # "past_scroll_id": retrieved_doc["past_scroll_id"],
            # "next_scroll_id": retrieved_doc["next_scroll_id"],
        }

    def get_tool_desc(self) -> dict:
        """
        Get the description of the search tool
        """
        return {
            "type": "function",
            "name": "search",
            "description": "Search the corpus for relevant information based on the given query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The query to search for.",
                    },
                },
                "required": ["query"],
            },
        }


def best_window_total_hits_np(
    doc: list[int],
    query: list[int],
    window: int = 256,
) -> tuple[int, int, list[int]]:
    """
    Helper to find window in the document that matches the query best
    - Use for slightly more intelligent preview of document text vs just
      returning the first window tokens (i.e., doc[:window])

    Returns: 
    - start_idx: index of the start of the best window
    - score: number of query tokens in the best window
    - best_window_slice: slice of the document that matches the query best
    """
    if not doc or not query or window <= 0:
        return (0, 0, [])
    n = len(doc)
    if window >= n:
        qset = set(query)
        score = sum(1 for x in doc if x in qset)
        return (0, score, doc)

    arr = np.asarray(doc)
    quniq = np.unique(np.asarray(query))
    marks = np.isin(arr, quniq).astype(np.int32)  # 1 if arr[i] in query, else 0
    sums = np.convolve(marks, np.ones(window, dtype=np.int32), mode="valid")
    best_start = int(np.argmax(sums))
    best_score = int(sums[best_start])
    return best_start, best_score, doc[best_start : best_start + window]
