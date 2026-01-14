"""
Scroll tool for BrowseComp-Plus
"""

from typing import Any

from datasets import Dataset
from act_prm.environments.base import BaseTool


class BaseScrollTool(BaseTool):
    """
    Base class for scroll tools
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
        current_doc_id: str,
        doc_dict: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """
        Scroll in the document context -> implemented in child classes
        """
        raise NotImplementedError

    def get_tool_desc(self) -> dict:
        """
        Get the description of the scroll tool -> implemented in child classes
        """
        raise NotImplementedError


class ScrollUpTool(BaseScrollTool):
    """
    Scroll up in the document context
    """
    def __call__(
        self,
        current_doc_id: str,
        doc_dict: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        """
        Scroll up in the document context
        """
        # Get the current document object
        if self.doc_dataset is not None:
            doc_idx = self.ds_corpus_index[current_doc_id]
            doc: dict[str, Any] = self.doc_dataset[doc_idx]
        else:
            doc_dict = doc_dict or self.doc_dict
            assert doc_dict is not None, "doc_dict not defined in ScrollUpTool function or init"
            doc: dict[str, Any] = doc_dict[current_doc_id]
        
        # Use pointer of past_scroll_id to get the previous document object
        if doc["past_scroll_id"] is not None:
            new_doc_id = doc["past_scroll_id"]
            new_doc_idx = self.ds_corpus_index[new_doc_id]
            new_doc = self.doc_dataset[new_doc_idx]

            result_str = "# Prior Text:\n\n"
            if new_doc["past_scroll_id"] is not None:
                result_str += "[...] "
            result_str += f"{new_doc["text"]}"
            if new_doc["next_scroll_id"] is not None:
                result_str += " [...]"
        else:
            result_str = "No prior text available."
        return doc, result_str

    def get_tool_desc(self) -> dict:
        """
        Get the description of the scroll up tool
        """
        return {
            "type": "function",
            "name": "scroll_up",
            "description": "Scroll up in the document context to view prior text.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }


class ScrollDownTool(BaseScrollTool):
    """
    Scroll down in the document context
    """
    def __call__(
        self,
        current_doc_id: str,
        doc_dict: dict[str, dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], str]:
        """
        Scroll down in the document context
        """
        # Get the current document object
        if self.doc_dataset is not None:
            doc_idx = self.ds_corpus_index[current_doc_id]
            doc: dict[str, Any] = self.doc_dataset[doc_idx]
        else:
            doc_dict = doc_dict or self.doc_dict
            assert doc_dict is not None, "doc_dict not defined in ScrollDownTool function or init"
            doc: dict[str, Any] = doc_dict[current_doc_id]
        
        # Use pointer of next_scroll_id to get the next document object
        if doc["next_scroll_id"] is not None:
            new_doc_id = doc["next_scroll_id"]
            new_doc_idx = self.ds_corpus_index[new_doc_id]
            new_doc = self.doc_dataset[new_doc_idx]
            result_str = "# Next Text:\n\n"
            if new_doc["past_scroll_id"] is not None:
                result_str += "[Scroll up for more...] "
            result_str += f"{new_doc["text"]}"
            if new_doc["next_scroll_id"] is not None:
                result_str += " [...Scroll down for more]"
        else:
            result_str = "No next text available."
        return doc, result_str

    def get_tool_desc(self) -> dict:
        """
        Get the description of the scroll up tool
        """
        return {
            "type": "function",
            "name": "scroll_down",
            "description": "Scroll down in the document context to view next text.",
            "parameters": {
                "type": "object",
                "properties": {},
            }
        }
