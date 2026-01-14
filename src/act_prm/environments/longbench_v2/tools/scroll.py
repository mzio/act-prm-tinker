"""
Scroll tool
"""

from typing import Any

from ...base import BaseTool


class ScrollUpTool(BaseTool):
    """
    Scroll up in the document context
    """
    def __call__(
        self,
        current_doc_id: int,
        all_doc_dicts: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """
        Scroll up in the document context
        """
        doc_dict: dict[str, Any] = all_doc_dicts[current_doc_id]

        if doc_dict["prev_chunk_idx"] is not None:
            new_doc_id = doc_dict["prev_chunk_idx"]
            new_doc_dict = all_doc_dicts[new_doc_id]

            _count_str = f"{new_doc_id + 1} of {len(all_doc_dicts)}"
            result_str = f"# Prior Text ({_count_str}):\n\n"
            if new_doc_dict["prev_chunk_idx"] is not None:
                result_str += "[Scroll up for more...] "
            result_str += f"{new_doc_dict["text"]}"
            if new_doc_dict["next_chunk_idx"] is not None:
                result_str += " [...Scroll down for more]"
        else:
            result_str = "No prior text available."
        return new_doc_dict, result_str

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


class ScrollDownTool(BaseTool):
    """
    Scroll down in the document context
    """
    def __call__(
        self,
        current_doc_id: int,
        all_doc_dicts: list[dict[str, Any]],
        **kwargs: Any,
    ) -> tuple[Any, str]:
        """
        Scroll down in the document context
        """
        doc_dict: dict[str, Any] = all_doc_dicts[current_doc_id]

        if doc_dict["next_chunk_idx"] is not None:
            new_doc_id = doc_dict["next_chunk_idx"]
            new_doc_dict = all_doc_dicts[new_doc_id]

            _count_str = f"{new_doc_id + 1} of {len(all_doc_dicts)}"
            result_str = f"# Next Text ({_count_str}):\n\n"
            if new_doc_dict["prev_chunk_idx"] is not None:
                result_str += "[Scroll up for more...] "
            result_str += f"{new_doc_dict["text"]}"
            if new_doc_dict["next_chunk_idx"] is not None:
                result_str += " [...Scroll down for more]"
        else:
            result_str = "No next text available."
        return new_doc_dict, result_str

    def get_tool_desc(self) -> dict:
        """
        Get the description of the scroll down tool
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
