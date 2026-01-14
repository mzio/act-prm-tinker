"""
Asynchronous search environment for BrowseComp-Plus
"""

import asyncio
from typing import Any

from ...graders.qa import LLMGraderForQA
from ..base import Environment

from .env import (BrowseCompPlusSearchEnv)
from .tools import ExpandTool, ScrollUpTool, ScrollDownTool, SearchTool


class AsyncBrowseCompPlusSearchEnv(BrowseCompPlusSearchEnv):
    """
    Asynchronous search environment for BrowseComp-Plus
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
        # Initialize base class attributes, as direct parent (BrowseCompPlusSearchEnv)
        # does heavy blocking work (loading data, initializing search tool)
        Environment.__init__(self, max_turns=max_turns, num_tries=num_tries, seed=seed, **kwargs)
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

        self.system_prompt = system_prompt
        self.next_obs_feedback = next_obs_feedback
        
        # Initialize data and tools as placeholders for now
        self.datasets = None
        self.ds_corpus = None
        self.ds_corpus_index = None
        self.tool_registry = None

        # async readiness
        self._ready = asyncio.Event()
        self._init_task: asyncio.Task[None] | None = None

        try:
            # If we're inside an event loop, schedule async init in the background
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # Otherwise, not inside a loop (constructed from sync code)
            # -> Fall back to blocking init
            self._sync_heavy_init()
            self._ready.set()
        else:
            self._init_task = loop.create_task(self._async_heavy_init())

    def _sync_heavy_init(self) -> None:
        """
        Synchronous initialization of data and tools
        """
        self.datasets, self.ds_corpus = self.init_data()
        self.ds_corpus_index = {v: i for i, v in enumerate(self.ds_corpus["doc_id"])}

        _tool_kwargs = {"doc_dataset": self.ds_corpus, "ds_corpus_index": self.ds_corpus_index}
        self.tool_registry = {
            "expand": ExpandTool(**_tool_kwargs),
            "scroll_up": ScrollUpTool(**_tool_kwargs),
            "scroll_down": ScrollDownTool(**_tool_kwargs),
            "search": SearchTool(
                corpus=self.ds_corpus,
                tokenizer=self.tokenizer,
                max_preview_tokens=self.max_preview_tokens,
                **self.search_tool_config,
            ),
        }

    async def _async_heavy_init(self) -> None:
        """
        Asynchronous initialization of data and tools
        """
        try:
            # init_data() is sync and can be heavy; offload to a thread so we don't block the loop
            self.datasets, self.ds_corpus = await asyncio.to_thread(self.init_data)

            # Building the index is CPU-light but fine here
            self.ds_corpus_index = {v: i for i, v in enumerate(self.ds_corpus["doc_id"])}

            # Tool construction: keep on event loop unless tools do blocking I/O
            _tool_kwargs = {"doc_dataset": self.ds_corpus, "ds_corpus_index": self.ds_corpus_index}
            # Fast bois
            expand = ExpandTool(**_tool_kwargs)
            scroll_up = ScrollUpTool(**_tool_kwargs)
            scroll_down = ScrollDownTool(**_tool_kwargs)
            # Slow boi
            search = await asyncio.to_thread(
                SearchTool,
                corpus=self.ds_corpus,
                tokenizer=self.tokenizer,
                max_preview_tokens=self.max_preview_tokens,
                **self.search_tool_config,
            )
            self.tool_registry = {
                "expand": expand,
                "scroll_up": scroll_up,
                "scroll_down": scroll_down,
                "search": search,
            }
        finally:
            # Ensure waiters get released even if an exception happens.
            # The exception will be re-raised when someone awaits ready().
            self._ready.set()

    async def ready(self) -> None:
        """
        Wait until async initialization is complete (and re-raise init errors)
        """
        await self._ready.wait()
        if self._init_task is not None:
            # Propagate exceptions from async init
            await self._init_task

    async def reset_async(self, *args: Any, **kwargs: Any):
        """
        Asynchronous reset -> auto-wait until ready before initializing new task
        """
        await self.ready()
        return super().reset(*args, **kwargs)

    async def step_async(self, *args: Any, **kwargs: Any):
        """
        Asynchronous step -> auto-wait until ready before taking a step
        """
        await self.ready()
        return super().step(*args, **kwargs)
