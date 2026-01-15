"""
Base generation with Tinker SamplingClient

Implements similar functions to the following Tinker Cookbook methods:
* `do_single_rollout`
* `do_group_rollout`
* `do_group_rollout_and_filter_constant_reward`
"""

import asyncio
from copy import copy
from typing import Any

from tinker.types import ModelInput
from tinker_cookbook.utils import ml_log
from transformers import PreTrainedTokenizerBase

from ..environments.base import Environment, EnvironmentState
from ..environments.types import EnvironmentStepResult
from ..llm_handlers.action_utils import get_actions
from ..llm_handlers.tinker import TinkerCompleter, TokensWithLogprobsAndText
from ..llm_handlers.types import ActionFromLLM
from ..replay_buffer.types import (
    EpisodeStep, Trajectory, TrajectoryGroup, MeanCenteredTrajectoryGroup,
)

from .utils_display import display_state_action_next_obs


class TinkerGenerator:
    """
    Compute rollouts using Tinker Completer (SamplingClient)
    """
    def __init__(
        self,
        llm: TinkerCompleter,
        env: Environment,
        hf_tokenizer: PreTrainedTokenizerBase,
        discount_factor: float = 0.9,
        verbose: bool = False,
        ml_logger: ml_log.Logger | None = None,
    ) -> None:
        self.llm = llm
        self.env = env
        self.hf_tokenizer = hf_tokenizer

        self.discount_factor = discount_factor
        self.verbose = verbose
        self.run_url = ml_logger.get_logger_url() if ml_logger is not None else None

    def _get_trajectory_group(self, **kwargs: Any) -> TrajectoryGroup:
        """
        Return trajectory group class
        - Override in subclasses, e.g., to return MeanCenteredTrajectoryGroup
        """
        return TrajectoryGroup(**kwargs)

    def _get_messages_from_state(
        self,
        state: EnvironmentState,
        default_context: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get messages from the environment state, in the form of
        [{"role": <role>, "content": <content>}, ...]
        """
        # First add prior observations + model's last response
        messages = (
            (state.prior_messages or []) 
            + (state.model_response or [])
            + state.new_messages
        )
        # Additional preprocessing into {"role": <role>, "content": <content>} format
        # -> See `act_prm.environments` classes for environment responses
        messages = [
            {"role": msg["role"], "content": msg["output"]}
            if msg.get("type", "") == "function_call_output"
            else msg
            for msg in messages
        ]
        # Add default context (few-shot examples) if provided
        default_context = copy(default_context or [])
        # Remove system prompt (will add it back after default context)
        if messages[0].get("role", "") == "system":
            messages = messages[1:]
        # Return final messages list
        return [
            {"role": "system", "content": state.system_prompt},
            *default_context,
            *messages,
        ]

    async def do_single_rollout(
        self,
        llm: TinkerCompleter | None = None,
        env: Environment | None = None,
        hf_tokenizer: PreTrainedTokenizerBase | None = None,
        split: str = "train",
        batch_id: int = 0,
        unique_data_sample_id: int = 0,
        generation_id: int = 0,
        try_step: int = 0,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> Trajectory:
        """
        Generate a full rollout in the environment, and return the trajectory
        """
        llm = llm or self.llm
        env = env or self.env
        hf_tokenizer = hf_tokenizer or self.hf_tokenizer

        max_tokens = max_tokens or llm.max_tokens
        temperature = temperature or llm.temperature

        episode_steps: list[EpisodeStep] = []
        state: EnvironmentState = await env.reset_async(
            sample_idx=unique_data_sample_id,
            generation_idx=generation_id,
            try_step=try_step,
        )
        done = False
        while not done:
            # Generate model responses and step through the environment
            state_messages: list[dict[str, Any]] = self._get_messages_from_state(state)
            input_ids: list[int] = hf_tokenizer.apply_chat_template(
                state_messages,
                add_generation_prompt=True,
                tokenize=True,
                tools=state.tools,
            )
            tinker_input: ModelInput = ModelInput.from_ints(input_ids)
            # 1. Generate model responses (thoughts + actions)
            response: TokensWithLogprobsAndText = await llm.generate(
                tinker_input,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            # print(hf_tokenizer.decode(input_ids))
            # breakpoint()
            model_messages = [{"role": "assistant", "content": response.text}]
            parsed_actions: list[ActionFromLLM] = get_actions(model_messages)

            # Recompute logprobs over parsed actions
            state_action_input_ids: list[int] = hf_tokenizer.apply_chat_template(
                state_messages + model_messages,
                add_generation_prompt=False,
                tokenize=True,
                tools=state.tools,
            )
            state_action_tinker_input = ModelInput.from_ints(state_action_input_ids)
            logprobs = await llm.compute_logprobs_async(state_action_tinker_input)
            action_token_len = len(state_action_input_ids) - len(input_ids)
            logprobs = logprobs[-action_token_len:]

            # 2. Step through the environment
            env_step_result: EnvironmentStepResult = await env.step_async(
                parsed_actions=parsed_actions,
                model_response=model_messages,
                current_state=state,
                # Set next_state.prior_messages to all messages
                current_messages=state_messages,
            )
            next_state = env_step_result.state
            reward     = env_step_result.reward
            done       = env_step_result.done
            truncated  = env_step_result.truncated
            # 3. Save EpisodeStep
            next_obs = [
                {
                    "role": msg["role"],
                    "content": msg["output"] if msg.get("output", None) else msg["content"]
                }
                for msg in next_state.new_messages
            ]
            episode_step = EpisodeStep(
                state=state_messages,  # state,
                action=model_messages[0],  # parsed_actions,
                next_obs=next_obs,
                tools=state.tools,
                state_len=len(input_ids),
                state_action_tokens=state_action_input_ids,
                # old_logprobs=response.logprobs,
                old_logprobs=logprobs,
                temperature=temperature,
                reward=reward,
                done=done,
                truncated=truncated,
                timestep=state.timestep,
                try_step=state.try_step,
                batch_id=batch_id,
                unique_data_sample_id=unique_data_sample_id,
                generation_id=generation_id,
                # is_train=split == "train",
                split=split,
            )
            episode_steps.append(episode_step)

            # If verbose, rich print the state and action
            if self.verbose:
                _header_text = (
                    f"Batch {batch_id}, Try {try_step}, "
                    f"Sample {unique_data_sample_id}, Generation {generation_id}, "
                    f"Step {state.timestep} (Max {env.max_turns - 1})"
                )
                self.display_state_action_next_obs(  # slightly coded for Qwen models for now
                    state_messages=state_messages,
                    action_messages=model_messages,
                    next_obs_messages=next_obs,
                    hf_tokenizer=hf_tokenizer,
                    tools=state.tools,
                    header_text=_header_text,
                    generation_id=generation_id,
                )
            # Transition to next state
            state = env_step_result.state

        return Trajectory(
            episode_steps=episode_steps,
            try_step=try_step,
            discount_factor=self.discount_factor,
            final_state=state,
            final_obs=next_obs,
            final_reward=reward,
        )

    async def do_group_rollout(
        self,
        num_return_sequences: int,
        **single_rollout_kwargs: Any,
    ) -> dict[str, list[TrajectoryGroup]]:
        """
        Generate a group of trajectories in the environment, and return a list of the trajectory
        group(s).

        By default, we should just return a singleton with 1 TrajectoryGroup. However, there may 
        be cases for >1 TrajectoryGroups, e.g., if we're generating multiple actions per step,
        and we want advantages over each (state, action, next_obs) tuple across generations
        """
        trajectories_in_group: list[Trajectory] = await asyncio.gather(
            *[
                self.do_single_rollout(generation_id=gen_idx, **single_rollout_kwargs)
                for gen_idx in range(num_return_sequences)
            ],
        )
        # final_rewards_in_group: list[float] = [t.final_reward for t in trajectories_in_group]
        all_trajectory_groups = [
            self._get_trajectory_group(
                trajectories=trajectories_in_group, 
                # final_rewards=final_rewards_in_group,
                discount_factor=self.discount_factor,
            )
        ]
        return {"policy": all_trajectory_groups}

    def display_state_action_next_obs(self,
        state_messages: list[dict[str, Any]],
        action_messages: list[dict[str, Any]],
        next_obs_messages: list[dict[str, Any]],
        hf_tokenizer: PreTrainedTokenizerBase,
        tools: list[dict[str, Any]],
        header_text: str,
        generation_id: int,
        # Rich colors
        system_color: str = "bold bright_yellow",
        tool_call_color: str = "bold bright_blue",
        tool_response_color: str = "bright_green",
        **other_rich_colors: Any,
    ) -> None:
        """
        Display the state, action, and next observations in a rich format
        """
        # Silly coloring to differentiate between generations
        _base_color = f"color({(generation_id + 1) % 8 + 8})"
        _bold_color = f"bold color({(generation_id + 1) % 8 + 8})"
        _rich_colors = {
            "system_color": system_color,
            "user_color": _base_color,
            "assistant_color": _bold_color,
            "tool_call_color": tool_call_color,
            "tool_response_color": tool_response_color,
        }
        _rich_colors.update(other_rich_colors)
        display_state_action_next_obs(
            state_messages=state_messages,
            action_messages=action_messages,
            next_obs_messages=next_obs_messages,
            hf_tokenizer=hf_tokenizer,
            tools=tools,
            header_text=header_text,
            run_url=self.run_url,
            **_rich_colors,
        )


class TinkerGRPOGenerator(TinkerGenerator):
    """
    Tinker Generator with Mean-Centered Return Rollouts
    """
    def _get_trajectory_group(self, **kwargs: Any) -> MeanCenteredTrajectoryGroup:
        """
        Returns trajectory group where we compute advantages by:
        1. Computing mean-centered final rewards: final_reward - mean(final_rewards)
        2. Optionally apply step-wise discounting to these values
        """
        return MeanCenteredTrajectoryGroup(**kwargs)
