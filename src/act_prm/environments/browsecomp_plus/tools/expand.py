"""
Expand tool for BrowseComp-Plus

For context, our search tool returns a list of search results, which each contain:
1. the title of the document
2. a preview of the document's text
3. an identifying `doc_id`

This is for brevity and mirrors real settings, where an LLM (and human) must *expand* on the result
to get more information (e.g., visiting a web page based on the search engine preview).
"""

from typing import Any

from datasets import Dataset

from act_prm.environments.base import BaseTool


class ExpandTool(BaseTool):
    """
    Expand a search result given its `doc_id`
    """
    def __init__(
        self,
        doc_dict: dict[str, dict[str, Any]] | None = None,
        doc_dataset: Dataset | None = None,
        ds_corpus_index: dict[str, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.doc_dict = doc_dict
        self.doc_dataset = doc_dataset
        self.ds_corpus_index = ds_corpus_index
        assert self.doc_dict or self.doc_dataset, "Either doc_dict or doc_dataset must be provided"
        if self.doc_dataset is not None:
            assert self.ds_corpus_index is not None, (
                "ds_corpus_index must be provided if doc_dataset is provided"
            )

    def __call__(
        self,
        doc_id: str,
        doc_dict: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """
        Expand a search result given its `doc_id`
        """
        if self.doc_dataset is not None:
            doc_idx = self.ds_corpus_index[doc_id]
            doc = self.doc_dataset[doc_idx]
        else:
            doc_dict = doc_dict or self.doc_dict
            doc: dict[str, Any] = doc_dict[doc_id]
        
        result_str = f"# Document {doc_id}:\n\n"
        # Give some minor indication of where the document is in the context
        if doc["past_scroll_id"] is not None:
            if not (
                self.ds_corpus_index is not None
                and doc["past_scroll_id"] not in self.ds_corpus_index
            ):
                result_str += "[Scroll up for more...] "
        # Add document text
        result_str += f"{doc["text"]}"
        # Give some minor indication of where the document is in the context
        if doc["next_scroll_id"] is not None:
            if not (
                self.ds_corpus_index is not None
                and doc["next_scroll_id"] not in self.ds_corpus_index
            ):
                result_str += " [...Scroll down for more]"
        return doc, result_str

    def get_tool_desc(self) -> dict:
        """
        Get the description of the expand tool
        """
        return {
            "type": "function",
            "name": "expand",
            "description": "Expand and visit a search result given its result ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {
                        "type": "string",
                        "description": (
                            "The `doc_id` of the search result to expand."
                            " Returns more information about the result."
                        ),
                    },
                },
                "required": ["doc_id"],
            },
        }
