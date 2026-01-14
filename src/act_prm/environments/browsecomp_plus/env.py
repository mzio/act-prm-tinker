"""
Search environment for BrowseComp-Plus
"""

import json
import os
from copy import copy
from os.path import join
from typing import Any

import numpy as np
from datasets import DatasetDict, Dataset, load_from_disk, load_dataset
from pydantic import InstanceOf
from rich import print as rich_print
from tqdm import tqdm

from ...graders.qa import LLMGraderForQA
from ...llm_handlers import ActionFromLLM
from ..base import Environment, BaseTool
from ..types import EnvironmentStateWithAnswer, EnvironmentStepResult

from .data_download import load_browsecomp_plus_dataset
from .utils import process_batch_for_search, render_prompt
from .tools import ExpandTool, ScrollUpTool, ScrollDownTool, SearchTool


class BrowseCompPlusSearchState(EnvironmentStateWithAnswer):
    """
    State for BrowseComp-Plus search tasks
    """
    tool_registry: dict[str, InstanceOf[BaseTool]]  # callable tools, by name
    tools: list[dict[str, Any]]                     # tool descriptions
    doc_dict: dict[str, Any] | None                 # current document (in the corpus)
    current_doc_id: str | None                      # current document id (to retrieve from corpus)


class BrowseCompPlusSearchStepResult(EnvironmentStepResult):
    """
    Step result for BrowseComp-Plus search tasks
    """
    state: BrowseCompPlusSearchState
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any] | None = None


class BrowseCompPlusSearchEnv(Environment):
    """
    Search environment for BrowseComp-Plus
    """
    def __init__(
        self,
        dataset_config: dict[str, Any],
        search_tool_config: dict[str, Any],
        grader_model_config: dict[str, Any],
        grader_model_samples: int = 1,
        grader_model_verbose: bool = False,
        hf_repo_id: str = "browsecomp_plus-search",
        max_preview_tokens: int = 204,  # about 1024 tokens overall
        doc_chunk_size: int = 1024,
        num_train_samples: int = 750,
        num_val_samples: int = 30,
        num_test_samples: int = 50,
        max_turns: int = 20,
        num_tries: int = 1,
        seed: int = 0,
        split: str = "train",
        system_prompt: str = "You are a helpful assistant that can answer questions and call tools.",
        next_obs_feedback: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(max_turns=max_turns, num_tries=num_tries, seed=seed, **kwargs)
        self.dataset_config = dataset_config
        self.hf_repo_id = hf_repo_id

        # Search tool config
        self.search_tool_config = search_tool_config
        
        # LLM-as-a-judge for grading
        self.grader_model_config = grader_model_config
        self.grader_model_samples = grader_model_samples
        self.grader_model = LLMGraderForQA(
            grader_model_config=grader_model_config,
            num_samples=grader_model_samples,
            verbose=grader_model_verbose,
        )

        # Build environment
        self.max_preview_tokens = max_preview_tokens
        self.doc_chunk_size = doc_chunk_size

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
        self.next_obs_feedback = next_obs_feedback
        self.datasets, self.ds_corpus = self.init_data()
        # Build index based on doc_id for self.ds_corpus
        self.ds_corpus_index = {v: i for i, v in enumerate(self.ds_corpus["doc_id"])}

        # Initialize tools
        _tool_kwargs = {
            "doc_dataset": self.ds_corpus,
            "ds_corpus_index": self.ds_corpus_index,
        }
        self.tool_registry = {
            "expand": ExpandTool(**_tool_kwargs),
            "scroll_up": ScrollUpTool(**_tool_kwargs),
            "scroll_down": ScrollDownTool(**_tool_kwargs),
            "search": SearchTool(
                corpus=self.ds_corpus,
                tokenizer=self.tokenizer,  # Inherited; see _init_tokenizer(self) in ../base.py
                max_preview_tokens=self.max_preview_tokens,
                **self.search_tool_config,
            ),
        }

    def __len__(self) -> int:
        """
        Get the environment's number of sample tasks
        """
        return len(self.datasets[self.split])

    def init_data(self) -> tuple[DatasetDict, Dataset, dict[str, Any]]:
        """
        Initialize dataset (from pre-downloaded file)
        Returns:
        - datasets: DatasetDict of questions and answers by (train, val, test) splits
        - dict_corpus: Dataset of dictionary corpus of docs
        - retriever_components: dict of retriever components
        """
        cache_dir = self.dataset_config["cache_dir"]
        # Check if dataset is already downloaded, and download + decrypt if not
        if not os.path.exists(join(cache_dir, "decrypted.jsonl")):
            load_browsecomp_plus_dataset(cache_dir=cache_dir)
        datasets = self.init_datasets()
        ds_corpus = self.init_ds_corpus()
        return datasets, ds_corpus

    def init_datasets(self) -> DatasetDict:
        """
        Initialize QA datasets
        """
        cache_dir = self.dataset_config["cache_dir"]
        hf_repo_id = (
            f"{self.hf_repo_id}-preview{self.max_preview_tokens}-chunk{self.doc_chunk_size}"
        )
        _hf_repo_id_text = f"[bright_blue]{hf_repo_id}[/bright_blue]"
        try:
            datasets = load_from_disk(hf_repo_id)
            rich_print(f"-> Loaded dataset from {_hf_repo_id_text}!")
        
        except Exception as e:  # File probably doesn't exist yet
            _error_text = f"[bright_red]{e}[/bright_red]"
            rich_print(f"[red]Error loading dataset from {_hf_repo_id_text}[/red]: {_error_text}")
            rich_print("Processing and saving datasets...")
            pbar = tqdm(
                total=self.num_train_samples + self.num_val_samples + self.num_test_samples,
                desc=f"-> Loading data from {join(cache_dir, "decrypted.jsonl")}",
            )
            datasets: list[dict[str, Any]] = []
            with open(join(cache_dir, "decrypted.jsonl"), "r", encoding="utf-8") as f:
                for line in f:
                    sample = json.loads(line)
                    datasets.append({
                        "query_id": sample["query_id"],
                        "query":    sample["query"],
                        "answer":   sample["answer"],
                    })
                    pbar.update(1)
                # Convert to HF dataset and save locally
                datasets = Dataset.from_list(datasets)
                datasets = self.get_splits(datasets)
                datasets.save_to_disk(hf_repo_id)
                rich_print(
                    f"Saved datasets with splits {datasets.keys()} to {_hf_repo_id_text}!"
                )
        return datasets

    def init_ds_corpus(self) -> Dataset:
        """
        Initialize dictionary corpus of docs (retrieval corpus)
        """
        # Get dictionary corpus of docs (retrieval corpus)
        doc_corpus_hf_repo_id = (
            f"{self.hf_repo_id}-dict-corpus-preview{self.max_preview_tokens}-chunk{self.doc_chunk_size}"
        )
        _doc_corpus_hf_repo_text = f"[bright_blue]{doc_corpus_hf_repo_id}[/bright_blue]"
        try:
            ds_corpus = load_from_disk(doc_corpus_hf_repo_id)
            rich_print(f"-> Loaded dictionary corpus from {_doc_corpus_hf_repo_text}!")
        
        except Exception as e:  # File probably doesn't exist yet
            _error_text = f"[bright_red]{e}[/bright_red]"
            rich_print(
                f"[red]Error loading dictionary corpus from {_doc_corpus_hf_repo_text}[/red]: {_error_text}"
            )
            rich_print("Processing and saving datasets...")
            ds_corpus = load_dataset(**self.dataset_config)  # download from HuggingFace Hub
            map_fn_kwargs = {
                "tokenizer": self.tokenizer,
                "doc_chunk_size": self.doc_chunk_size,
            }
            ds_corpus = ds_corpus.map(
                process_batch_for_search,
                batched=True,
                fn_kwargs=map_fn_kwargs,
                remove_columns=list(ds_corpus.column_names),
            )
            # Save dictionary corpus to hub
            ds_corpus.save_to_disk(doc_corpus_hf_repo_id)
            rich_print(f"Saved dictionary corpus to {_doc_corpus_hf_repo_text}!")
        return ds_corpus

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
    ) -> BrowseCompPlusSearchState:
        """
        Reset environment (starting new episode + loading a new task)
        """
        sample_idx_adj = self.adjust_sample_idx(sample_idx)  # Wrap around if out of bounds
        sample = self.datasets[self.split][sample_idx_adj]

        # Build prompt and save answer
        query  = str(sample["query"])
        answer = str(sample["answer"])
        prompt = render_prompt(query)
        tools  = [
            self.tool_registry["search"].get_tool_desc(),  # others not available yet
            # self.tool_registry["expand"].get_tool_desc(),
            # self.tool_registry["scroll_up"].get_tool_desc(),
            # self.tool_registry["scroll_down"].get_tool_desc(),
        ]
        messages = [{"role": "user", "content": prompt}]
        return BrowseCompPlusSearchState(
            system_prompt=self.system_prompt,
            new_messages=messages,
            model_response=None,
            prior_messages=[],
            tool_registry=self.tool_registry,
            tools=tools,
            # BrowseComp-Plus-specific fields
            question=query,
            answer=answer,
            doc_dict=None,
            current_doc_id=None,
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
    ) -> BrowseCompPlusSearchStepResult:
        """
        Step through the environment; see _step_impl for details
        """
        return self._step_impl(**kwargs)

    def _step_impl(
        self,
        parsed_actions: list[ActionFromLLM],
        model_response: Any,
        current_state: BrowseCompPlusSearchState,
        current_messages: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> BrowseCompPlusSearchStepResult:
        """
        Subclass implementation of step
        """
        question = str(current_state.question)
        answer   = str(current_state.answer)

        tool_kwargs = {
            "doc_dict": current_state.doc_dict,
            "current_doc_id": current_state.current_doc_id,
        }

        done = False
        truncated = False
        reward = 0
        updated_try_step = False
        made_tool_call = False

        metadata = copy(current_state.metadata)
        timestep = copy(current_state.timestep)
        try_step = copy(current_state.try_step)
        sample_id = current_state.sample_id
        generation_id = current_state.generation_id
        batch_id = current_state.batch_id

        # Keep track of next doc_id and doc
        # -> Tool call should update, but by default will be same as current state
        next_doc_id = copy(current_state.current_doc_id)
        next_doc_dict = copy(current_state.doc_dict)

        # Create environment response
        env_messages = []
        available_tools = current_state.tools
        available_tool_registry = current_state.tool_registry

        # Parse actions (messages and tool calls)
        for action_idx, action in enumerate(parsed_actions):
            # Handle tool calls
            if action.type == "function_call":
                fc_name = action.name
                fc_args = action.arguments

                if fc_name == "invalid_tool_call":
                    stdout = f"Invalid tool call:\n\n{action.text}"
                    made_tool_call = False
                elif fc_name not in current_state.tool_registry:
                    stdout = (
                        f"Invalid tool call:\n\n{action.text}\n\n"
                        f"'{fc_name}' not currently available."
                    )
                    made_tool_call = False
                else:
                    if not isinstance(fc_args, dict):
                        raise TypeError(
                            f"Expected tool call arguments to be a mapping, got {type(fc_args)}"
                        )
                    # Execute the tool call
                    fn_call = current_state.tool_registry[fc_name]
                    try:
                        maybe_new_doc, results = fn_call(**fc_args, **tool_kwargs)
                    except Exception as e:
                        print(f"Error during tool call: {e}")
                        stdout = f"Invalid tool call: {action.text}"
                        stdout += f"\n\nError: {e}"
                        made_tool_call = False
                        results = None
                        maybe_new_doc = None

                    # Update the document and identifiers
                    if isinstance(maybe_new_doc, dict):
                        # True for all tool calls except SearchTool
                        next_doc_id = maybe_new_doc["doc_id"]
                        next_doc_dict = maybe_new_doc

                    if results is not None:
                        stdout = (
                            results if isinstance(results, str) else json.dumps(results, indent=2)
                        )
                        made_tool_call = True
                env_response = {
                    "role": "tool",
                    "type": "function_call_output",
                    "call_id": action.call_id,
                    "output": stdout,
                }
                env_messages.append(env_response)

            elif action.type in ["message", "reasoning"]:
                text = action.text or ""
                if (
                    action.type == "message"
                    and action_idx + 1 == len(parsed_actions)
                    # and "Final Answer: " in text
                ):
                    done = True
                    if "Final Answer: " in text:  # Last action was an answer submission
                        reward, grader_text = self.grader_model(
                            question=question,
                            correct_answer=answer,
                            response=text,
                            sample_id=sample_id,
                            generation_id=generation_id,
                            split=self.split,
                        )
                        reward = float(reward)  # Convert bool to float for reward
                    else:
                        reward = 0
                        grader_text = ""
                    user_content = "# RESULT: CORRECT!" if reward == 1 else "# RESULT: INCORRECT!"
                    if self.next_obs_feedback:  # Include feedback in the next observation
                        user_content += f"\n\n{grader_text}"
                    env_messages.append({
                        "role": "user",
                        "content": user_content,
                    })
                    metadata["correct"] = reward
                    metadata["total"] = 1

        # Update available tools based on what was called
        if made_tool_call:
            available_tool_names = []
            if fc_name == "search":
                available_tools = [
                    self.tool_registry["expand"].get_tool_desc(),
                    self.tool_registry["search"].get_tool_desc(),
                ]
                available_tool_names.extend(["expand", "search"])
            else:  # e.g., if fc_name == "expand":
                available_tools = [
                    self.tool_registry["search"].get_tool_desc(),
                ]
                available_tool_names.append("search")
                # Hacky patches for valid scroll up and down calls
                if (
                    isinstance(maybe_new_doc, dict) and
                    maybe_new_doc["next_scroll_id"] is not None and 
                    maybe_new_doc["next_scroll_id"] in self.ds_corpus_index
                ):
                    available_tools.append(self.tool_registry["scroll_down"].get_tool_desc())
                    available_tool_names.append("scroll_down")
                if (
                    isinstance(maybe_new_doc, dict) and
                    maybe_new_doc["past_scroll_id"] is not None and
                    maybe_new_doc["past_scroll_id"] in self.ds_corpus_index
                ):
                    available_tools.append(self.tool_registry["scroll_up"].get_tool_desc())
                    available_tool_names.append("scroll_up")
            available_tool_registry = {
                k: v for k, v in self.tool_registry.items() if k in available_tool_names
            }

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
        new_state = BrowseCompPlusSearchState(
            system_prompt=current_state.system_prompt,
            new_messages=env_messages,
            model_response=model_response,
            prior_messages=current_messages or [],
            tool_registry=available_tool_registry,
            tools=available_tools,
            # BrowseComp-Plus-specific fields
            question=question,
            answer=answer,
            doc_dict=next_doc_dict,
            current_doc_id=next_doc_id,
            # Step-wise metadata
            sample_id=sample_id,
            generation_id=generation_id,
            batch_id=batch_id,
            try_step=try_step,
            timestep=timestep,
            # Track for accuracy eval
            metadata=metadata,
        )
        return BrowseCompPlusSearchStepResult(
            state=new_state,
            reward=reward,
            done=done,
            truncated=truncated,
            info=new_state.metadata,  # alternative access
        )


class BrowseCompPlusGeneratedSearchEnv(BrowseCompPlusSearchEnv):
    """
    Search environment for BrowseComp-Plus with LLM-generated QA pairs
    - Ignore for now
    """

    def init_datasets(self) -> DatasetDict:
        """
        Initialize QA datasets
        """
        # Load from HuggingFace Hub
        datasets = load_dataset(**self.dataset_config)
        
        # Process samples for consistency (query_id, query, answer columns)
        def _process_sample(sample: dict[str, Any]) -> dict[str, Any]:
            """
            Process an LLM-generated QA pair sample for consistency
            """
            return {
                "query_id": f"{sample["sample_idx"]}_{sample["generation_idx"]}",
                "query": sample["final_question"],
                "answer": sample["final_answer"],
            }
        datasets = datasets.map(
            _process_sample,
            remove_columns=list(datasets.column_names),
        )
        # Get train, eval, test splits
        datasets = self.get_splits(datasets)
        return datasets
