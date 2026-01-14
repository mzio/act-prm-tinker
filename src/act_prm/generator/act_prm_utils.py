"""
Helper functions for processing Act-PRM samples
"""

from copy import copy, deepcopy
from typing import Any

from tinker import SamplingClient
from tinker.types import ModelInput
from transformers import PreTrainedTokenizerBase

from ..trainer.utils import gather_with_progress


def get_thought_and_actions(
    msg: dict[str, Any],
    action_bos: str = "<tool_call>",
    action_eos: str = "</tool_call>",
    final_answer_bos: str = "Final Answer: ",
) -> tuple[dict[str, Any], dict[str, Any] | None]:

    """
    From a given chat message, return two messages where the first only contains "thought text"
    and the second only contains "actions text".

    For example, if `msg` is of the form:
    {"role": "assistant", "content": "(thought text...)<tool_call>(actions text...)</tool_call>"}

    Then we return two message dicts:
    - {"role": "assistant", "content": "(thought text...)"}
    - {"role": "assistant", "content": "<tool_call>(actions text...)</tool_call>"}
    """
    thought_msg, actions_msg = msg, msg

    if msg["role"] == "assistant":  # Otherwise, we'll just return the original message
        thought_msg, actions_msg = copy(msg), copy(msg)
        content = msg["content"]
        if final_answer_bos is not None and final_answer_bos in content:
            _thought_action_splits = content.split(final_answer_bos)
            thought_str = final_answer_bos.join(_thought_action_splits[:-1]).strip()
            actions_str = _thought_action_splits[-1].strip()
            thought_msg["content"] = thought_str
            actions_msg["content"] = f"{final_answer_bos}{actions_str}"

        elif action_bos not in content:  # No action in this message, e.g., because invalid tool call
            return thought_msg, None

        else:
            # Extract content from tool calls, e.g., f"<tool_call>{content}</tool_call>"
            # -> Assumes only one action per message
            assert action_bos in content, f"No explicit action found in message: {content}"
            thought_str, actions_str = content.split(action_bos, maxsplit=1)
            assert action_eos in actions_str, f"No closing action tag found in message: {actions_str}"
            # Extract the action text content
            actions_str = actions_str.split(action_eos, maxsplit=1)[0].strip()
            # Add back the action tags (maybe redundant)
            actions_str = f"{action_bos}{actions_str}{action_eos}"
            thought_msg["content"] = f"{thought_str.strip()}"
            actions_msg["content"] = f"{actions_str}"

    return thought_msg, actions_msg


def get_latent_completion(
    messages: list[dict[str, Any]],
    continue_final_message: bool = True,
    **get_thought_actions_kwargs: Any,
) -> tuple[list[dict[str, Any]], str]:
    """
    From standard chat messages, return tuple of (latent_inputs, action_target)
    where latent_inputs is a list of chat messages of the form:
    [
        {"role": "user", "content": <user_message>},
        {"role": "assistant", "content": "<action_message_0><thought_message_0>"},
        ...,
        {"role": "assistant", "content": "<action_message_last><thought_message_last>"},
    ]
    
    For training convenience, if `continue_final_message` is True, then we return `action_target`
    as the last assistant action message, e.g., "<action_message_last>"
    - We also change messages to prompt for the last thought, e.g., "<thought_message_last>"
    """
    latent_inputs = []
    action_target = ""

    assistant_indices = []
    for msg_idx, msg in enumerate(messages):
        if msg["role"] != "assistant":
            latent_inputs.append(msg)
        else:
            assistant_indices.append(msg_idx)
            msg_thought, msg_actions = get_thought_and_actions(msg, **get_thought_actions_kwargs)
            # Extract content from message dicts
            c_thought, c_actions = msg_thought["content"], msg_actions["content"]
            latent_content = f"{c_actions}\n\n<thought>\n{c_thought}\n</thought>"
            latent_content = latent_content.replace("\n\n\n", "\n\n")  # tidy a bit
            msg_latent = {"role": msg["role"], "content": latent_content}
            latent_inputs.append(msg_latent)

    if continue_final_message:
        # Make the last assistant message prompt for the latent thought
        last_assistant_idx = assistant_indices[-1]
        last_assistant_msg = latent_inputs[last_assistant_idx]
        msg_content = last_assistant_msg["content"].split("<thought>\n")[0]
        action_target = copy(msg_content).strip()
        msg_content = f"{msg_content}<thought>\n"
        latent_inputs[last_assistant_idx] = {
            "role": last_assistant_msg["role"],
            "content": msg_content,
        }
        latent_inputs = latent_inputs[:last_assistant_idx + 1]
    return latent_inputs, action_target


def process_state_messages_for_act_prm(
    state_messages: list[dict[str, str]],
    system_prompt: dict[str, str],
) -> list[dict[str, str]]:
    """
    Ensure state messages have 1 system prompt and no last assistant message
    """
    state_messages = deepcopy(state_messages)
    if state_messages[0]["role"] == "system":
        state_messages = state_messages[1:]
    if state_messages[-1]["role"] == "assistant":
        state_messages = state_messages[:-1]
    return [system_prompt] + state_messages


async def compute_single_thought_action_metrics(
    state_messages: list[dict[str, str]],
    generated_thought: str,
    target_action: str,
    hf_tokenizer: PreTrainedTokenizerBase,
    sampling_client: SamplingClient,
    tools: list[dict[str, str]] | None = None,
) -> dict[str, float | list[int] | list[float] | int, list[dict[str, str]]]:
    """
    Compute logprob and likelihood metrics for a given state, thought, and target action

    These include artifacts useful for SFT training, e.g., (state, thought, action) tokens

    Returns a dictionary with the following keys:
    - action_probs:                  p(action | state, generated_thought)
    - action_logprobs:               logprobs of the action tokens
    - state_thought_action_tokens:   tokens of the current (state, thought, action) trajectory
    - state_thought_action_logprobs: logprobs of the current (state, thought, action) trajectory
    - state_thought_len:             number of (state, thought) tokens
    - action_len:                    number of action tokens
    - thought_action_messages:       thought-action model response
    """
    _tokenize_kwargs = {
        "add_generation_prompt": False,
        "tokenize": True,
        "tools": tools,
    }
    generated_thought = generated_thought.split("</thought>")[0].strip()
    thought_msgs = [{"role": "assistant", "content": generated_thought}]
    thought_action_msgs = [{
        "role": "assistant",
        "content": f"{generated_thought}\n\n{target_action}",
    }]
    prefix_tokens = hf_tokenizer.apply_chat_template(
        state_messages + thought_msgs,
        continue_final_message=True,
        **_tokenize_kwargs,
    )
    full_tokens = hf_tokenizer.apply_chat_template(
        state_messages + thought_action_msgs,
        continue_final_message=False,
        **_tokenize_kwargs,
    )
    action_token_len = len(full_tokens) - len(prefix_tokens)

    # Compute length-normalized joint probabilities of action tokens as reward metrics
    tinker_prompt = ModelInput.from_ints(full_tokens)
    logprobs = await sampling_client.compute_logprobs_async(tinker_prompt)
    action_logprobs = np.array(logprobs[-action_token_len:])
    action_probs = np.exp(action_logprobs.sum() / len(action_logprobs)).item()  # length-normalize

    return {
        "action_probs": action_probs,
        "action_logprobs": action_logprobs.tolist(),
        "state_thought_action_tokens": full_tokens,
        "state_thought_action_logprobs": logprobs,
        "state_thought_len": len(prefix_tokens),
        "action_len": action_token_len,
        "thought_action_messages": thought_action_msgs,
    }


async def compute_group_thought_action_metrics(
    state_messages: list[dict[str, str]],
    generated_thoughts: list[str],
    target_action: str,
    system_prompt: dict[str, str],
    tools: list[dict[str, str]],
    hf_tokenizer: PreTrainedTokenizerBase,
    sampling_client: SamplingClient,
) -> dict[str, list[Any]]:
    """
    Compute thought-action metrics for a group of generated thoughts

    Returns a dictionary with the same keys as compute_single_thought_action_metrics, but also:
    - state_len: number of state tokens
    """
    state_messages = process_state_messages_for_act_prm(state_messages, system_prompt)
    state_len = len(
        hf_tokenizer.apply_chat_template(
            state_messages,
            add_generation_prompt=True,
            tokenize=True,
            tools=tools,
        )
    )
    metrics_in_group: list[dict[str, Any]] = await gather_with_progress(
        [
            compute_single_thought_action_metrics(
                state_messages=state_messages,
                generated_thought=gen_thought,
                target_action=target_action,
                hf_tokenizer=hf_tokenizer,
                sampling_client=sampling_client,
                tools=tools,
            ) for gen_thought in generated_thoughts
        ],
        desc="Computing thought-action metrics, p(action | state, thought)",
        colour="green",
    )
    # Convert list of dicts to dict of lists
    metrics_by_key: dict[str, Any] = {}
    for k in metrics_in_group[0].keys():
        metrics_by_key[k] = [getattr(m, k) for m in metrics_in_group]
    metrics_by_key["state_len"] = [state_len] * len(generated_thoughts)
    return metrics_by_key
