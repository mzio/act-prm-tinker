"""
Tinker Generator with Action Process Reward Models
"""

import asyncio
from copy import deepcopy
from typing import Any

import numpy as np
from tinker.types import ModelInput
from transformers import PreTrainedTokenizerBase

from ..environments.base import Environment
from ..environments.act_prm import ActionProcessRewardState
from ..environments.types import EnvironmentStepResult
from ..llm_handlers.action_utils import get_actions
from ..llm_handlers.tinker import SamplingClient, TinkerCompleter, TokensWithLogprobsAndText
from ..llm_handlers.types import ActionFromLLM
from ..replay_buffer.types import (
    EpisodeStep, Trajectory, TrajectoryGroup, MeanCenteredTrajectoryGroup,
)
from ..trainer.tinker.utils import gather_with_progress

from .tinker import TinkerGenerator


def process_state_messages_for_metrics(
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
    state_messages = process_state_messages_for_metrics(state_messages, system_prompt)
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
    for k, v in metrics_in_group[0].items():
        metrics_by_key[k] = [getattr(m, k) for m in metrics_in_group]
    metrics_by_key["state_len"] = [state_len] * len(generated_thoughts)
    return metrics_by_key


class TinkerActPrmGenerator(TinkerGenerator):
    """
    Tinker Generator with Action Process Reward Models
    """
    def __init__(
        self,
        action_bos: str = "<tool_call>",
        action_eos: str = "</tool_call>",
        thought_bos: str = "<thought>",
        thought_eos: str = "</thought>",
        final_answer_bos: str = "Final Answer: ",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Delimiters to parse thoughts and actions from response text
        self.action_bos  = action_bos
        self.action_eos  = action_eos
        self.thought_bos = thought_bos
        self.thought_eos = thought_eos
        self.final_answer_bos = final_answer_bos

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
        So by default, we remove the last assistant message
        -> TinkerActionFirstActPrmGenerator will do something different
        """
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer
        state_messages = deepcopy(state_messages)[:-1]
        input_ids: list[int] = hf_tokenizer.apply_chat_template(
            state_messages,
            add_generation_prompt=True,
            tokenize=True,
        )
        return state_messages, input_ids

    def _parse_thoughts(self, response_text: str) -> str:
        """
        Extract only the thought text from a response text
        """
        # Extract thoughts if explicitly tagged
        response_text = response_text.split(self.thought_bos)[-1].strip()
        response_text = response_text.split(self.thought_eos)[0].strip()
        # Extract thought as text before action or final answer
        response_text = self.action_bos.join(response_text.split(self.action_bos)[:-1]).strip()
        response_text = self.final_answer_bos.join(response_text.split(self.final_answer_bos)[:-1])
        return response_text.strip()

    async def do_group_rollout(
        self,
        num_return_sequences: int,
        **single_rollout_kwargs: Any,
    ) -> dict[str, list[TrajectoryGroup]]:
        """
        Wrapper for do_act_prm_group_rollout
        """
        return self.do_act_prm_group_rollout(
            num_return_sequences=num_return_sequences,
            **single_rollout_kwargs,
        )

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
        all_trajectory_groups: list[TrajectoryGroup] = []

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
            _, input_ids = self._get_thought_prompt(state_messages)
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
            # ---------- End saving episode steps for each generation ----------

            # Transition to next state
            state = next_state

        return {"policy": all_trajectory_groups}

    def _get_trajectory_group_from_generations(
        self,
        state_messages: list[dict[str, str]],
        actions_in_group: list[dict[str, str]],
        state_len: int,
        state_action_tokens_in_group: list[list[int]],
        old_logprobs_in_group: list[list[float]],
        rewards_in_group: list[float],
        generation_ids_in_group: list[int],
        try_step: int,
        **shared_kwargs: Any,
    ) -> TrajectoryGroup:
        """
        Save generations to a TrajectoryGroup
        """
        episode_steps_in_group: list[EpisodeStep] = [
            EpisodeStep(
                state=state_messages,
                action=action,  # dict[str, str]
                state_len=state_len,
                state_action_tokens=state_action_tokens_in_group[i],
                old_logprobs=old_logprobs_in_group[i],
                reward=rewards_in_group[i],
                generation_id=generation_ids_in_group[i],
                **shared_kwargs,
            ) for i, action in enumerate(actions_in_group)
        ]
        trajectories_in_group: list[Trajectory] = [
            Trajectory(
                episode_steps=[episode_step],
                try_step=try_step,
                discount_factor=self.discount_factor,
                final_state=state_messages,
                final_obs=[],
                final_reward=rewards_in_group[i],
            ) for i, episode_step in enumerate(episode_steps_in_group)
        ]
        return self._get_trajectory_group(
            trajectories=trajectories_in_group,
            final_rewards=rewards_in_group,
            discount_factor=self.discount_factor,
        )
