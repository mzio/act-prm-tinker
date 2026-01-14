"""
Tools for HotpotQA Multiple Choice environment
"""

from ..base import BaseTool


class VisitTool(BaseTool):
    """
    Visit a title
    """
    def __call__(self, title: str, all_docs_dict: dict[str, str]) -> tuple[None, str]:
        """
        Visit a title
        """
        if title not in all_docs_dict:
            return f"Title '{title}' not found in valid titles."
        return all_docs_dict[title]

    def get_tool_desc(self) -> dict:
        """
        Get the tool description
        """
        return {
            "type": "function",
            "name": "visit",
            "description": "Visit a given title and expand for more information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "The title to visit"},
                },
            },
            "required": ["title"],
        }
