"""
Tinker Generator with Action Process Reward Models
"""

import asyncio
from copy import copy, deepcopy
from typing import Any

import numpy as np
from tinker.types import ModelInput
from transformers import PreTrainedTokenizerBase

from ..environments.base import Environment
from ..environments.act_prm import ActionProcessRewardState
from ..environments.types import EnvironmentStepResult
from ..llm_handlers.action_utils import get_actions
from ..llm_handlers.tinker import TinkerCompleter, TokensWithLogprobsAndText
from ..llm_handlers.types import ActionFromLLM
from ..replay_buffer.types import TrajectoryGroup, MeanCenteredTrajectoryGroup

from .tinker_act_prm import (
    compute_group_thought_action_metrics,
    process_state_messages_for_metrics,
    TinkerActPrmGenerator,
)


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


def get_action_prompted_completion(
    messages: list[dict[str, Any]],
    continue_final_message: bool = True,
    thought_bos: str = "<thought>",
    thought_eos: str = "</thought>",
    **get_thought_actions_kwargs: Any,
) -> tuple[list[dict[str, Any]], str]:
    """
    Convert standard messages (state, thought, action) to messages that *condition on the action*
    to prompt for the thought (state, action, ...)

    In more detail: from standard chat messages, returns a tuple of (latent_inputs, action_target)
    where latent_inputs is a list of chat messages of the form:
    [
        {"role": "user", "content": <user_message>},
        {"role": "assistant", "content": "<action_message_0><thought_message_0>"},
        ...,
        {"role": "assistant", "content": "<action_message_last><thought_message_last>"},
    ]
    
    If `continue_final_message` is True, then we include the last `action_target` as the last
    assistant message, i.e., to prompt a continuation for the thought that leads to it.
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
            latent_content = f"{c_actions}\n\n{thought_bos}\n{c_thought}\n{thought_eos}"
            latent_content = latent_content.replace("\n\n\n", "\n\n")  # tidy a bit
            msg_latent = {"role": msg["role"], "content": latent_content}
            latent_inputs.append(msg_latent)

    if continue_final_message:
        # Make the last assistant message prompt for the latent thought
        last_assistant_idx = assistant_indices[-1]
        last_assistant_msg = latent_inputs[last_assistant_idx]
        msg_content = last_assistant_msg["content"].split(f"{thought_bos}\n")[0]
        action_target = copy(msg_content).strip()
        latent_inputs[last_assistant_idx] = {
            "role": last_assistant_msg["role"],
            "content": f"{msg_content}{thought_bos}\n",
        }
        latent_inputs = latent_inputs[:last_assistant_idx + 1]
    return latent_inputs, action_target


class TinkerActionPromptActPrmGenerator(TinkerActPrmGenerator):
    """
    Tinker Generator with Action Process Reward Models
    """

    def _get_thought_prompt(
        self,
        state_messages: list[dict[str, Any]],
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
    ) -> tuple[list[dict[str, Any]], list[int]]:
        """
        Get the thought prompt messages for the given state messages.
        We expect `state_messages` as a list of the form:
        ```
        [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
            ...
            {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        ]
        ```
        Here, we use `get_action_prompted_completion` to convert these messages to those that
        prompt for the thought
        """
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        # Prompt for thoughts, e.g., of the form <action_bos>(action)</action_eos><thought_bos>
        state_messages = deepcopy(state_messages)
        state_messages, action_target = get_action_prompted_completion(
            state_messages, 
            continue_final_message=True,
            action_bos=self.action_bos,
            action_eos=self.action_eos,
            thought_bos=self.thought_bos,
            thought_eos=self.thought_eos,
            final_answer_bos=self.final_answer_bos,
        )
        input_ids: list[int] = hf_tokenizer.apply_chat_template(
            state_messages,
            add_generation_prompt=False,
            continue_final_message=True,  # don't add eos_token to final message
            tokenize=True,
        )
        return state_messages, input_ids

    async def do_act_prm_group_rollout(
        self,
        num_return_sequences: int,
        llm: TinkerCompleter | None = None,
        env: Environment | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        split: str = "train",
        batch_id: int = 0,
        unique_data_sample_id: int = 0,
        try_step: int = 0,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, list[TrajectoryGroup]]:
        """
        Generate thought-action trajectories given observed actions in an Act-PRM environment.

        Unlike typical group-rollouts, at *each* step, we:
        1. Generate `num_return_sequences` thoughts,
        2. Compute the per-step reward for each generation,
        3. Pick the highest-reward thought to continue for the next step.

        This results in *1* full trajectory (from start to workflow completion).

        However for training, we still save each (state, action', thought', reward') tuple 
        for all `num_return_sequences` thoughts as a TrajectoryGroup. This results in returning
        `num_steps` TrajectoryGroups.

        Currently returns both samples of the form (state, thought, action) and (state, action, thought) for RL training
        - (state, thought, action) can be used for SFT training, as it's the standard (state, action, next_obs) tuple
        - (state, action, thought) can be used for RL training for action-prompted generation

        MZ 1/13/26: We may update this to return only one, and have another class implement the other
        """
        llm = llm or self.llm
        env = env or self.env
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        max_tokens = max_tokens or llm.max_tokens
        temperature = temperature or llm.temperature

        # Initialize list of all trajectory groups to return
        # -> We return (1) (state, thought, action) and (2) (state, action, thought) trajectories
        #    (1) is used for SFT or TinkerActPrmGenerator RL
        #    (2) is used for TinkerActionPromptActPrmGenerator RL
        all_trajectory_groups: list[TrajectoryGroup] = []
        all_act_prompt_trajectory_groups: list[TrajectoryGroup] = []

        state: ActionProcessRewardState = await env.reset_async(
            sample_idx=unique_data_sample_id,
            generation_idx=0,
            try_step=try_step,
        )
        max_turns = len(state.assistant_indices)

        done = False
        while not done:
            # Generate model responses and step through the environment
            # state_messages should be [obs, action, ..., obs, action]
            state_messages: list[dict[str, Any]] = self._get_messages_from_state(state)
            # Prompt for thoughts
            act_prompt_state_messages, input_ids = self._get_thought_prompt(state_messages)
            tinker_input = ModelInput.from_ints(input_ids)
            # Generate `num_return_sequences` thoughts
            responses_in_group: list[TokensWithLogprobsAndText] = await asyncio.gather(*[
                llm.generate(
                    tinker_input,
                    max_tokens=max_tokens,
                    temperature=temperature
                ) for _ in range(num_return_sequences)
            ])
            # Parse thoughts from responses
            thoughts_in_group: list[str] = [  # demonstrations may have <thought>...</thought>
                self._parse_thoughts(response.text) for response in responses_in_group
            ]
            # Compute per-step rewards for each thought
            # -> Get current state without last action
            standard_chat = process_state_messages_for_metrics(state_messages, state.system_prompt)
            group_metrics = await compute_group_thought_action_metrics(
                state_messages=standard_chat,
                generated_thoughts=thoughts_in_group,
                target_action=state.action_target,
                system_prompt=state.system_prompt,
                tools=state.tools,
                hf_tokenizer=hf_tokenizer,
                sampling_client=llm.sampling_client,
            )
            # Get artifacts for RL training
            rewards_in_group = group_metrics["action_probs"]
            thought_action_messages = group_metrics["thought_action_messages"]
            # Visualize generated thoughts
            if self.verbose:
                for i in range(num_return_sequences):
                    header_text = (
                        f"Batch {batch_id}, Try {try_step}, "
                        f"Sample {unique_data_sample_id}, Generation {i}, "
                        f"Step {state.timestep} / {max_turns - 1}, "
                        f"Reward {rewards_in_group[i]:.4f}"
                    )
                    self.display_state_action_next_obs(
                        state_messages=standard_chat,
                        action_messages=thought_action_messages[i],
                        next_obs_messages=[],
                        hf_tokenizer=hf_tokenizer,
                        tools=state.tools,
                        header_text=header_text,
                        generation_id=i,
                    )

            # Pick the highest-reward thought to continue for the next step
            best_thought_idx = np.argmax(rewards_in_group)
            best_thought = thoughts_in_group[best_thought_idx]
            model_messages = [{"role": "assistant", "content": best_thought}]
            parsed_actions: list[ActionFromLLM] = [get_actions(model_messages)]

            env_step_result: EnvironmentStepResult = await env.step_async(
                parsed_actions=parsed_actions,
                # model_response=model_messages,
                current_state=state,
                current_messages=state_messages,
            )
            next_state = env_step_result.state
            truncated  = env_step_result.truncated
            done       = env_step_result.done
            next_obs = [
                {
                    "role": msg["role"],
                    "content": msg["output"] if msg.get("output", None) else msg["content"]
                } for msg in next_state.new_messages
            ]

            # ---------- Save episode steps for each generation ----------
            shared_kwargs = {
                "next_obs": next_obs,
                "tools": state.tools,
                "temperature": temperature,
                "done": done,
                "truncated": truncated,
                "timestep": state.timestep,
                "try_step": try_step,
                "batch_id": batch_id,
                "unique_data_sample_id": unique_data_sample_id,
                "split": split,
            }
            # Save (state, thought, action) steps
            trajectory_group = self._get_trajectory_group_from_generations(
                state_messages=standard_chat,
                actions_in_group=[msg[0] for msg in thought_action_messages],  # list[dict[str, str]]
                state_len=group_metrics["state_len"][0],
                state_action_tokens_in_group=group_metrics["state_thought_action_tokens"],
                old_logprobs_in_group=group_metrics["action_logprobs"],
                rewards_in_group=rewards_in_group,
                generation_ids_in_group=range(num_return_sequences),
                try_step=try_step,
                **shared_kwargs,
            )
            all_trajectory_groups.append(trajectory_group)

            # Save action-prompted (state, action, thought) steps            
            # 1. Get state-action-thought artifacts for RL training
            state_action_thought_messages_G = [
                deepcopy(act_prompt_state_messages) for _ in range(num_return_sequences)
            ]
            state_action_thought_input_ids_G = []
            for i, response in enumerate(responses_in_group):
                state_action_thought_messages_G[i][-1]["content"] += response.text
                state_action_thought_input_ids_G.append(
                    hf_tokenizer.apply_chat_template(
                        state_action_thought_messages_G[i],
                        add_generation_prompt=False,
                        tokenize=True,
                        tools=state.tools,
                    )
                )
            action_token_lens_G = [
                len(state_action_thought_input_ids_G[i]) - len(input_ids)
                for i in range(num_return_sequences)
            ]
            state_action_thought_tinker_input_G = [
                ModelInput.from_ints(input_ids) for input_ids in state_action_thought_input_ids_G
            ]
            logprobs_G = await asyncio.gather(*[
                llm.compute_logprobs_async(tinker_input)[-action_token_lens_G[i]:]
                for i, tinker_input in enumerate(state_action_thought_tinker_input_G)
            ])
            actions_in_group = [
                {"role": "assistant", "content": responses_in_group[i].text}
                for i in range(num_return_sequences)
            ]
            
            # 2. Actually save state-action-thought steps to a TrajectoryGroup
            act_prompt_trajectory_group = self._get_trajectory_group_from_generations(
                state_messages=act_prompt_state_messages,
                actions_in_group=actions_in_group,  # list[dict[str, str]]
                state_len=group_metrics["state_len"][0],
                state_action_tokens_in_group=state_action_thought_input_ids_G[i],
                old_logprobs_in_group=logprobs_G[i],
                rewards_in_group=rewards_in_group,
                generation_ids_in_group=range(num_return_sequences),
                try_step=try_step,
                **shared_kwargs,
            )
            all_act_prompt_trajectory_groups.append(act_prompt_trajectory_group)
            # ---------- End saving episode steps for each generation ----------
            
            # Transition to next state
            state = next_state

        return {
            "policy": all_trajectory_groups,
            "act_prompt": all_act_prompt_trajectory_groups,
        }
