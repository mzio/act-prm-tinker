"""
LongBench environment
"""

from copy import copy
from os.path import join
from typing import Any

import numpy as np
from datasets import DatasetDict, load_dataset
from pydantic import InstanceOf
from rich import print as rich_print
from transformers import AutoTokenizer

from ...llm_handlers import ActionFromLLM
from ..base import BaseTool, Environment
from ..types import EnvironmentStateWithAnswer, EnvironmentStepResult
from .prompts import INITIAL_PROMPT_TEMPLATE
from .tools import SearchTool, ScrollUpTool, ScrollDownTool
from .utils import chunk_text_by_tokens, convert_text_chunks_to_dicts


RESULT_TEMPLATE = """## Document View:
'''
{document}
'''{scroll_message}
"""

class LongBenchState(EnvironmentStateWithAnswer):
    """
    State of the LongBench environment
    """
    doc_dict: dict[str, Any] | None
    all_doc_dicts: list[dict[str, Any]]
    doc_chunk_idx: int
    tool_registry: dict[str, InstanceOf[BaseTool]]  # callable tools, by name
    tools: list[dict[str, Any]]                     # tool descriptions


class LongBenchStepResult(EnvironmentStepResult):
    """
    Step result of the LongBench environment
    """
    state: LongBenchState


class LongBenchEnvironment(Environment):
    """
    LongBench environment
    """
    def __init__(
        self,
        dataset_config: dict[str, Any],
        search_tool_config: dict[str, Any],
        # grader_model_config: dict[str, Any],  # or just match on exact match?
        tokenizer_config: dict[str, Any],
        doc_chunk_size: int = 2048,
        doc_chunk_overlap: int = 256,
        num_train_samples: int = 400,
        num_val_samples: int = 16,
        num_test_samples: int = 60,
        max_turns: int = 20,
        num_tries: int = 1,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = "You are a helpful assistant that can answer questions and call tools.",
        **kwargs: Any,
    ) -> None:
        super().__init__(max_turns=max_turns, num_tries=num_tries, seed=seed, **kwargs)
        self.dataset_config = dataset_config
        self.search_tool_config = search_tool_config
        self.tokenizer = AutoTokenizer.from_pretrained(**tokenizer_config)

        # Build environment
        self.doc_chunk_size = doc_chunk_size
        self.doc_chunk_overlap = doc_chunk_overlap

        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples

        self.max_turns = max_turns
        self.num_tries = num_tries
        self.seed = seed
        self.split = split

        # Initialize data
        self.system_prompt = system_prompt
        self.datasets = self.init_data()

        # # Initialize tools -> maybe should be done per reset call?
        # self.tool_registry = {
        #     "search": SearchTool(**search_tool_config),
        # }

    def __len__(self) -> int:
        """
        Get number of samples in dataset
        """
        return len(self.datasets[self.split])

    def init_data(self) -> DatasetDict:
        """
        Load raw data from HF dataset hub and split into train/val/test
        """
        ds = load_dataset(**self.dataset_config)
        # Get splits
        trainval_test_dict = ds.train_test_split(
            test_size=self.num_test_samples, shuffle=True, seed=self.seed
        )
        train_val_dict = trainval_test_dict["train"].train_test_split(
            test_size=self.num_val_samples, shuffle=True, seed=self.seed
        )
        # return DatasetDict(
        #     train=train_val_dict["train"],
        #     eval=train_val_dict["test"],
        #     test=trainval_test_dict["test"],
        # )
        return DatasetDict({
            "train": train_val_dict["train"],
            "eval": train_val_dict["test"],
            "test": trainval_test_dict["test"],
        })

    def shuffle(self, seed: int | None = None) -> None:
        """
        Shuffle dataset
        """
        seed = seed or self.seed
        np.random.seed(seed)
        indices = np.arange(len(self.datasets[self.split]))
        np.random.shuffle(indices)
        self.datasets[self.split] = self.datasets[self.split][indices]
        
    def reset(
        self,
        sample_idx: int,
        generation_idx: int,
        try_step: int = 0,
        batch_idx: int = 0,
    ) -> LongBenchState:
        """
        Reset environment (starting new episode + loading a new task)
        """
        sample_idx_adj = self.adjust_sample_idx(sample_idx)  # Wrap around if out of bounds
        sample = self.datasets[self.split][sample_idx_adj]
        document = sample["context"]
        # Split document into chunks
        _, text_chunks = chunk_text_by_tokens(document, self.tokenizer)
        all_doc_dicts = convert_text_chunks_to_dicts(text_chunks)
        current_chunk_id = 0  # start with the first chunk
        doc_dict = all_doc_dicts[current_chunk_id]
        # Get answer choices
        choices = []
        for k in ["choice_A", "choice_B", "choice_C", "choice_D"]:
            _letter = k[len("choice_"):].upper()
            choices.append(f"{_letter}: {sample[k]}")
        choices = "\n".join(choices)

        # Build prompt and save answer
        question = sample["question"]
        answer = sample["answer"].upper()  # convert to uppercase letter for matching
        prompt = INITIAL_PROMPT_TEMPLATE.format(
            question=question,
            choices=choices,
            document=doc_dict["text"],
        )
        _search_save_path = join(
            self.dataset_config["cache_dir"],
            f"longbench_v2-{self.split}-{sample_idx:04d}"
        )
        tool_registry = {
            "search": SearchTool(
                **self.search_tool_config,
                corpus=all_doc_dicts,
                save_path=_search_save_path,
            ),
            "scroll_down": ScrollDownTool(),
            "scroll_up": ScrollUpTool(),
        }
        tools = [tool.get_tool_desc() for tool in tool_registry.values()]
        messages = [
            {"role": "user", "content": prompt},
        ]

        return LongBenchState(
            system_prompt=self.system_prompt,
            new_messages=messages,
            model_response=None,
            prior_messages=[],
            tool_registry=tool_registry,
            tools=tools,
            # LongBench-specific fields
            question=question,
            answer=answer,
            doc_dict=doc_dict,
            all_doc_dicts=all_doc_dicts,
            doc_chunk_idx=current_chunk_id,
            # Step-wise metadata
            sample_id=sample_idx,
            generation_id=generation_idx,
            batch_id=batch_idx,
            try_step=try_step,
            timestep=0,
            # Track for accuracy eval
            metadata={"correct": 0, "total": 1},
        )

    def step(self, **kwargs: Any) -> LongBenchStepResult:
        """
        Step through the environment; see _step_impl for details
        """
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: LongBenchState,
        current_messages: list[dict[str, Any]] | None = None,
    ) -> LongBenchStepResult:
        """
        Step through the environment
        """
        question = str(current_state.question)
        answer   = str(current_state.answer)
        # prior_messages = current_state.prior_messages
        
        done = False
        truncated = False
        reward = 0
        updated_try_step = False
        
        metadata = copy(current_state.metadata)
        timestep = copy(current_state.timestep)
        try_step = copy(current_state.try_step)
        sample_id = current_state.sample_id
        generation_id = current_state.generation_id
        batch_id = current_state.batch_id

        # Create environment response
        env_messages = []
        new_doc_dict = copy(current_state.doc_dict)

        # Parse actions (messages and tool calls)
        for action_idx, action in enumerate(parsed_actions):
            if action.type == "function_call":  # handle tool call
                fc_name = action.name
                fc_args = action.arguments
                try:  # Execute tool call
                    tool = current_state.tool_registry[fc_name]
                    if "scroll" in fc_name:
                        fc_args.update({
                            "current_doc_id": current_state.doc_chunk_idx,
                            "all_doc_dicts": current_state.all_doc_dicts,
                        })
                    new_doc_dict, maybe_result_str = tool(**fc_args)
                    # maybe_doc_result = result_str
                except Exception as e:
                    new_doc_dict = None
                    maybe_result_str = str(e)
                    raise e
                    breakpoint()

                if new_doc_dict is not None:
                    scroll_msg = ""
                    try:
                        if new_doc_dict["next_chunk_idx"] is not None:
                            scroll_msg += "\n- Scroll down for more..."
                        if new_doc_dict["prev_chunk_idx"] is not None:
                            scroll_msg += "\n- Scroll up for more..."
                        stdout = RESULT_TEMPLATE.format(
                            document=new_doc_dict["text"],
                            scroll_message=scroll_msg,
                        )
                    except Exception as e:
                        print(e)
                        print(new_doc_dict.keys())
                        breakpoint()
                else:
                    stdout = f"Error: {maybe_result_str}"

                env_response = {
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": action.call_id,
                    "output": stdout,  # or "content" for HF transformers
                }
                env_messages.append(env_response)

            elif action.type in ["message", "reasoning"]:
                text = action.text or ""
                if (
                    action.type == "message"
                    and action_idx + 1 == len(parsed_actions)
                ):
                    done = True
                    if "Final Answer: " in text:  # Last action was an answer submission
                        ans_pred = text.split("Final Answer: ")[1].strip().lower()
                        ans_true = answer.lower()
                        reward = float(ans_pred == ans_true)  # convert bool to float for reward
                    else:
                        reward = 0
                    
                    if reward == 1:
                        user_content = "# RESULT: CORRECT!"
                    else:
                        user_content = "# RESULT: INCORRECT!"

                    env_messages.append({"role": "user", "content": user_content})
                    metadata["correct"] = reward
                    metadata["total"] = 1

            # Update timesteps, fail if too many turns
            timestep = timestep + 1
            if timestep >= self.max_turns:
                truncated = True
                done = True
                env_messages.append({"role": "user", "content": self.truncation_message})
                if not updated_try_step:
                    try_step += 1
                    updated_try_step = True

            # Handle badness (environment should always respond to LLM response)
            if len(env_messages) == 0:
                env_messages.append({
                    "role": "user",
                    "content": "No tool calls or final answers were parsed. Please try again",
                })
            if new_doc_dict is None:
                new_doc_dict = copy(current_state.doc_dict)

            metadata.update(
                {"reward": reward, "done": done, "truncated": truncated},
            )
            new_state = LongBenchState(
                system_prompt=current_state.system_prompt,
                new_messages=env_messages,
                model_response=model_response,
                prior_messages=current_messages or [],
                tool_registry=current_state.tool_registry,
                tools=current_state.tools,
                # LongBench-specific fields
                question=question,
                answer=answer,
                doc_dict=new_doc_dict,
                all_doc_dicts=current_state.all_doc_dicts,
                doc_chunk_idx=new_doc_dict["chunk_idx"],
                # Step-wise metadata
                sample_id=sample_id,
                generation_id=generation_id,
                batch_id=batch_id,
                try_step=try_step,
                timestep=timestep,
                # Track for accuracy eval
                metadata=metadata,
            )
            return LongBenchStepResult(
                state=new_state,
                reward=reward,
                done=done,
                truncated=truncated,
                info=new_state.metadata,  # alternative access
            )


class AsyncLongBenchEnvironment(LongBenchEnvironment):
    """
    Asynchronous LongBench environment
    """
    async def reset_async(self, **kwargs: Any) -> LongBenchState:
        """
        Asynchronous reset -> assumes super().reset() is fast and non-blocking
        """
        return super().reset(**kwargs)

    async def step_async(self, **kwargs: Any) -> LongBenchStepResult:
        """
        Asynchronous step -> assumes super().step() is fast and non-blocking
        """
        return super().step(**kwargs)
