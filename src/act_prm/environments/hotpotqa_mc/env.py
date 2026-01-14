"""
HotpotQA Multiple Choice Environment
"""

from copy import copy
from typing import Any

import numpy as np
from datasets import Dataset,DatasetDict, load_dataset
from pydantic import InstanceOf
from rich import print as rich_print

from ...graders.qa import LLMGraderForQA
from ...llm_handlers import ActionFromLLM
from ..base import Environment
from ..types import EnvironmentStateWithAnswer, EnvironmentStepResult

from .prompts import FEWSHOT_PROMPTS
from .tools import VisitTool
from .utils import process_sample, process_sample_from_gen_dataset


class HotpotQAMultipleChoiceState(EnvironmentStateWithAnswer):
    """
    State for HotpotQA multiple choice tasks (Pydantic object)
    """
    all_docs_dict: dict[str, str]                    # e.g., {title: doc_text}
    tool_registry: dict[str, InstanceOf[VisitTool]]  # callable tools, by name
    tools: list[dict[str, Any]]                      # tool descriptions


class HotpotQAMultipleChoiceStepResult(EnvironmentStepResult):
    """
    Step result for HotpotQA multiple choice tasks
    """
    state: HotpotQAMultipleChoiceState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


class HotpotQAMultipleChoiceEnv(Environment):
    """
    Multiple choice environment for HotpotQA
    """
    
    def __init__(
        self,
        dataset_config: dict[str, Any],
        grader_model_config: dict[str, Any] | None = None,
        grader_model_samples: int = 1,
        grader_model_verbose: bool = False,
        qa_are_generated: bool = False,
        ambiguous_titles: bool = False,
        next_obs_feedback: bool = False,
        num_fewshot_prompts: int = 0,
        num_train_samples: int = 1000,
        num_val_samples: int = 64,
        num_test_samples: int = 100,
        max_turns: int = 20,
        num_tries: int = 1,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = "You are a helpful assistant that can answer questions and call tools.",
        truncation_message: str = "Sorry, you have reached the maximum number of steps. Please try again.",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset_config = dataset_config
        self.qa_are_generated = qa_are_generated
        self.ambiguous_titles = ambiguous_titles
        # Include grader feedback after answer submission if True
        self.next_obs_feedback = next_obs_feedback
        self.num_fewshot_prompts = num_fewshot_prompts  # if > 0, add few-shot samples in prompt

        # LLM-as-a-judge for grading
        self.grader_model_config = grader_model_config
        self.grader_model_samples = grader_model_samples
        self.grader_model = LLMGraderForQA(
            grader_model_config=grader_model_config,
            num_samples=grader_model_samples,
            verbose=grader_model_verbose,
        )
        
        # Build environment
        self.num_train_samples = num_train_samples
        self.num_val_samples = num_val_samples
        self.num_test_samples = num_test_samples
        self.split = split
        
        self.max_turns = max_turns
        self.num_tries = num_tries
        self.seed = seed
        self.split = split
        
        # Load data and tools
        self.system_prompt = system_prompt
        self.truncation_message = truncation_message
        self.datasets = self.init_data()

        # Initialize tools
        self.tool_registry = {"visit": VisitTool()}

        # Initialize default context (fewshot prompts)
        self.default_context = []
        for _idx in range(num_fewshot_prompts):
            self.default_context.extend(FEWSHOT_PROMPTS[_idx])
            # Add natural conversation follow-up (not included in FEWSHOT_PROMPTS)
            self.default_context.append({
                "role": "assistant",
                "content": "Great! Do you have a follow-up request?",
            })
        
    def __len__(self) -> int:
        """
        Get the environment's number of sample tasks
        """
        return len(self.datasets[self.split])        

    def init_data(self) -> DatasetDict:
        """
        Initialize dataset (from pre-downloaded file)
        Returns:
        - datasets: DatasetDict of questions and answers by (train, val, test) splits
        """
        ds = load_dataset(**self.dataset_config)
        ds = ds.map(
            process_sample if not self.qa_are_generated else process_sample_from_gen_dataset,
            fn_kwargs={
                "ambiguous_titles": self.ambiguous_titles,
                "include_titles_in_prompt": True,
            },
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )
        return self.get_splits(ds)

    def get_splits(self, dataset: Dataset) -> DatasetDict:
        """
        Get splits from dataset
        """
        trainval_test_dict = dataset.train_test_split(
            test_size=self.num_test_samples, shuffle=True, seed=self.seed,
        )
        train_val_dict = trainval_test_dict["train"].train_test_split(
            test_size=self.num_val_samples, shuffle=True, seed=self.seed,
        )
        return DatasetDict({
            "train": train_val_dict["train"],
            "eval": train_val_dict["test"],
            "test": trainval_test_dict["test"],
        })

    def shuffle(self, seed: int | None = None) -> None:
        """
        Shuffle dataset
        """
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        indices = np.arange(len(self.datasets[self.split]))
        np.random.shuffle(indices)
        self.datasets[self.split] = self.datasets[self.split][indices]

    def reset(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
    ) -> HotpotQAMultipleChoiceState:
        """
        Reset environment (starting new episode + loading a new task)
        """
        sample_idx_adj = self.adjust_sample_idx(sample_idx)  # Wrap around if out of bounds
        sample = self.datasets[self.split][sample_idx_adj]
        # Build lookup dictionary for all documents by title
        sample["all_docs_dict"] = {
            title: doc
            for title, doc in zip(sample["all_titles"], sample["all_docs"])
        }
        messages = self.default_context + [
            {"role": "user", "content": sample["prompt"]}
        ]
        return HotpotQAMultipleChoiceState(
            system_prompt=self.system_prompt,
            new_messages=messages,
            model_response=None,
            prior_messages=[],
            tool_registry=self.tool_registry,
            tools=[self.tool_registry["visit"].get_tool_desc()],
            # HotpotQA-specific fields
            question=str(sample["question"]),
            answer=str(sample["answer"]),
            all_docs_dict=sample["all_docs_dict"],
            # Step-wise metadata
            sample_id=sample_idx,
            generation_id=generation_idx,
            batch_id=batch_idx,
            try_step=try_step,
            timestep=0,
            # Track for accuracy eval
            metadata={"correct": 0, "total": 1},
        )

    def step(
        self,
        **kwargs: Any,
    ) -> HotpotQAMultipleChoiceStepResult:
        """
        Step through the environment; see _step_impl for details
        """
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: HotpotQAMultipleChoiceState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> HotpotQAMultipleChoiceStepResult:
        """
        Subclass implementation of step
        """
        question = str(current_state.question)
        answer   = str(current_state.answer)
        # Use to retrieve documents
        all_docs_dict = current_state.all_docs_dict

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
        available_tools = copy(current_state.tools)

        # Parse actions (messages and tool calls)
        for action_idx, action in enumerate(parsed_actions):
            if action.type == "function_call":  # Handle tool call (only visit)
                fc_name = action.name
                fc_args = action.arguments
                try:
                    # Execute tool call (visit in this case)
                    title = fc_args.get("title", None)
                    assert title in all_docs_dict, f"Title '{title}' not found in available titles"
                    tool = current_state.tool_registry[fc_name]
                    result = tool(**fc_args, all_docs_dict=all_docs_dict)
                
                except Exception as e:
                    if title not in all_docs_dict and title is not None:
                        result = f"Title '{title}' not found in available titles"
                    else:
                        result = (
                            f"Invalid tool call: {action.text}. "
                            f"Error during execution:\n{str(e)}"
                        )
                env_response = {
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": action.call_id,
                    "output": result,   # have "output" for OAI Responses
                }
                env_messages.append(env_response)
            
            elif action.type in ["message", "reasoning"]:
                text = action.text or ""
                if (
                    action.type == "message"
                    and action_idx + 1 == len(parsed_actions)
                    and "Final Answer: " in action.text
                ):
                    # Last action was an answer submission
                    reward, grader_text = self.grader_model(
                        question=question,
                        correct_answer=answer,
                        response=text,
                        sample_id=sample_id,
                        generation_id=generation_id,
                        split=self.split,
                    )
                    reward = float(reward)  # Convert bool to float for reward
                    done = True
                    user_content = "# RESULT: CORRECT!" if reward == 1 else "# RESULT: INCORRECT!"
                    if self.next_obs_feedback:  # Include feedback in the next observation
                        user_content += f"\n\n{grader_text}"
                    env_messages.append({
                        "role": "user",
                        "content": user_content,
                    })
                    metadata["correct"] = reward
                    metadata["total"] = 1  # explicit here

        # Update timesteps, fail if too many turns
        timestep += 1
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

        metadata.update(
            {"reward": reward, "done": done, "truncated": truncated},
        )
        new_state = HotpotQAMultipleChoiceState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages or [],
            tool_registry=current_state.tool_registry,
            tools=available_tools,
            # HotpotQA-specific fields
            question=question,
            answer=answer,
            all_docs_dict=all_docs_dict,
            # Step-wise metadata
            sample_id=sample_id,
            generation_id=generation_id,
            batch_id=batch_id,
            try_step=try_step,
            timestep=timestep,
            # Track for accuracy eval
            metadata=metadata,
        )
        return HotpotQAMultipleChoiceStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=new_state.metadata,  # alternative access
        )


class AsyncHotpotQAMultipleChoiceEnv(HotpotQAMultipleChoiceEnv):
    """
    Asynchronous environment for HotpotQA multiple choice tasks
    """
    async def reset_async(
        self,
        sample_idx: int = 0,
        generation_idx: int = 0,
        try_step: int = 0,
        batch_idx: int = 0,
        **kwargs: Any,
    ) -> HotpotQAMultipleChoiceState:
        """
        Asynchronous reset -> assumes super().reset() is fast and non-blocking
        """
        return super().reset(
            sample_idx=sample_idx,
            generation_idx=generation_idx,
            batch_idx=batch_idx,
            try_step=try_step,
            **kwargs,
        )

    async def step_async(
        self,
        **kwargs: Any,
    ) -> HotpotQAMultipleChoiceStepResult:
        """
        Asynchronous step -> assumes super().step() is fast and non-blocking
        """
        return super().step(**kwargs)
