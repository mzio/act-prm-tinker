"""
Helper functions for processing actions from HuggingFace Transformer and Tinker LLM handlers
"""

from typing import Any

import json
from json import JSONDecodeError
from rich import print as rich_print

from .types import ActionFromLLM


def get_actions(
    response: list[dict[str, Any]],
    tool_call_argname: str = "arguments",
    **tool_call_parse_kwargs: Any,
) -> list[ActionFromLLM]:
    """
    Parse chat response into list of actions, where 
    response is a (singleton) list: [{"role": "assistant", "content": <response_text>}].
    """
    action_list = []
    # Split thoughts and tool_calls into different messages
    try:
        response = get_messages_from_text(
            response[0]["content"],
            tool_call_argname=tool_call_argname,
            **tool_call_parse_kwargs,
        )
    except Exception as e:
        rich_print(f"[red]Error in get_messages_from_text: {e}[/red]")
        breakpoint()

    for message in response:
        if message.get("tool_calls", None) is not None:
            for tool_call in message["tool_calls"]:
                output = tool_call["function"]
                name = output.get("name", "invalid_tool_call")
                arguments = output.get(tool_call_argname, {})
                if not isinstance(arguments, dict):
                    arguments = {"arguments": json.dumps(arguments)}
                elif len(arguments) == 0:
                    arguments = {"arguments": json.dumps({})}
                text_repr = json.dumps(output)
                action_list.append(
                    ActionFromLLM(
                        role="assistant",
                        type="function_call",
                        text=text_repr,
                        call_id=None,
                        name=name,
                        arguments=arguments,
                    )
                )
        else:
            # Parse as regular message
            action_list.append(
                ActionFromLLM(
                    role="assistant",
                    type="message",
                    text=message["content"],
                    call_id=None,
                    name=None,
                    arguments=None,
                )
            )
    return action_list


def get_messages_from_text(
    text: str,
    tool_call_bos: str = "<tool_call>",
    tool_call_eos: str = "</tool_call>",
    tool_call_argname: str = "arguments",
) -> list[dict[str, Any]]:
    """
    Convert text to LLM chat messages
    """
    messages = []
    try:
        tool_call_str = text.split(tool_call_bos)[-1].split(tool_call_eos)[0]
        tool_call = json.loads(tool_call_str)
        valid_tool_call = True
    except JSONDecodeError:
        valid_tool_call = False

    if valid_tool_call:
        if isinstance(tool_call, str):
            valid_tool_call = False
        else:
            try:
                assert tool_call.get("name", None) is not None
                assert tool_call.get(tool_call_argname, None) is not None
            except AssertionError:
                if tool_call.get("arguments", None) is not None:
                    tool_call_argname = "arguments"
                else:
                    valid_tool_call = False

    # Convert any text before to regular message
    message = text.split(tool_call_bos)[0].strip()
    if len(message) > 0:
        messages.append({"role": "assistant", "content": message})

    if valid_tool_call:
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": tool_call}],
            }
        )
    elif tool_call_bos in text:  # Invalid tool call
        try:
            _invalid_tool_call_text = text.split(tool_call_bos)[-1].strip()
            assert _invalid_tool_call_text != "", "Invalid tool call"
            try:
                _invalid_tool_call_text = _invalid_tool_call_text.split(tool_call_eos)[0].strip()
                assert _invalid_tool_call_text != "", "Invalid tool call"
            except AssertionError:
                pass
        except AssertionError:
            _invalid_tool_call_text = text.strip()

        _invalid_tool_call = {
            "name": "invalid_tool_call",
            "arguments": _invalid_tool_call_text,
        }
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [{"type": "function", "function": _invalid_tool_call}],
            }
        )
    return messages
