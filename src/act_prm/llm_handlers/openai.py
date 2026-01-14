import asyncio
import json

from copy import deepcopy
from json.decoder import JSONDecodeError
from typing import Any, Sequence

from openai import AsyncOpenAI, BadRequestError, OpenAI
from openai.types.responses import (
    Response,
    ResponseFunctionToolCall,
    ResponseOutputItem,
    ResponseOutputMessage,
)

from .base import LLM
from .types import ActionFromLLM


class OpenAIResponsesLLM(LLM):
    """
    OpenAI LLM

    Example usage:
    ```
    state = env.reset()
    done = False
    timestep = 0

    while not done:
        messages = llm.update_messages(
            messages=state.new_messages,
            model_response=state.model_response,
            prior_messages=state.prior_messages,
            interleave=False,
        )
        model_response = llm.sample(
            system_prompt=state.system_prompt,
            messages=messages,
            tools=state.tools,
            max_new_tokens=1024,
            num_return_sequences=1,
            logprobs=True,
        )
        action_list = llm.get_actions(model_response)
        step_result = env.step(action_list, model_response, timestep)
        timestep = step_result.timestep
        next_state = step_result.state
    ```
    """

    def __init__(
        self,
        model: str = "gpt-5-mini",
        use_vision: bool = False,
        generation_config: Any | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=model, generation_config=generation_config, **kwargs)
        self.use_vision = use_vision
        self.client = OpenAI(base_url=base_url)
        self.last_usage = None
        # 'temperature' is not supported with GPT-5 models
        if "gpt-5" in model and "temperature" in self.generation_config:
            self.generation_config.pop("temperature")

    def _sample_single(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> Response | None:
        """
        Generate a single response from the model
        """
        generation_kwargs = dict(self.generation_config, **generation_kwargs)
        if "temperature" in generation_kwargs and "gpt-5" in self.model:
            generation_kwargs.pop("temperature")
        if max_new_tokens is not None:
            generation_kwargs["max_output_tokens"] = max_new_tokens

        if "reasoning_effort" in generation_kwargs and (
            self.model_name.startswith("o") or "gpt-5" in self.model_name
        ):
            reasoning = {
                "effort": generation_kwargs["reasoning_effort"],
                "summary": "auto",
            }
            generation_kwargs.pop("reasoning_effort")
            generation_kwargs["reasoning"] = reasoning

        if "max_new_tokens" in generation_kwargs:
            generation_kwargs.pop("max_new_tokens")
        if "num_return_sequences" in generation_kwargs:
            generation_kwargs.pop("num_return_sequences")

        try:
            # for _idx, msg in enumerate(messages):
            #     print(f"messages {_idx} {type(msg)}: {msg}")
            print("generation_kwargs:", generation_kwargs)
            response = self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=messages,
                tools=tools,
                **generation_kwargs,
            )
            # Track token usage *before* returning
            self.last_usage = self._track_tokens(getattr(response, "usage", None))
            if self.last_usage is not None:
                print("OpenAI token usage:")
                for k, v in self.last_usage.items():
                    print(f"-> {k}: {v}")
            return response

        except BadRequestError as e:
            if e.code == "invalid_prompt":
                print("Openai invalid prompt error: %s", e)
                return None

            if e.type == "invalid_request_error":
                print("OpenAI invalid request error: %s", e)
                print(f"self.last_usage: {self.last_usage}")
                return None

            print(f"OpenAI error: {e}")
            print(f"type(e): {type(e)}")
            print(vars(e))
            print(f"self.last_usage: {self.last_usage}")
            raise e

    def _sample_batch(
        self,
        num_return_sequences: int = 1,
        **kwargs: Any,
    ) -> list[Response | None]:
        """
        Generate a batch of responses from the model
        """
        return [self._sample_single(**kwargs) for _ in range(num_return_sequences)]

    def sample(
        self,
        system_prompt: str | None,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[Response | None]:
        """
        Generate response(s) from the model
        """
        return self._sample_batch(
            num_return_sequences=num_return_sequences,
            system_prompt=system_prompt,
            messages=messages,
            tools=tools,
            max_new_tokens=max_new_tokens,
            **generation_kwargs,
        )

    def update_messages(
        self,
        messages: list[dict[str, Any]],
        model_response: Response | None,
        prior_messages: list[
            dict[str, Any] | ResponseOutputMessage | ResponseFunctionToolCall
        ],
        interleave: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Return updated messages for the model
        - If interleave, the model_response will be interleaved with the messages:
        [model_response[0], messages[0], model_response[1], messages[1], ...]
        - Otherwise, the messages will be appended to the end of the model_response:
        [model_response[0], ..., model_response[-1], messages[0], ..., messages[-1]]
        """
        if interleave or model_response is None:
            new_messages: Sequence[ResponseOutputItem | dict[str, Any]] = []
        else:
            new_messages = deepcopy(model_response.output)

        for idx, message in enumerate(messages):
            if interleave and model_response is not None:
                new_messages.append(model_response.output[idx])
            if message.get("type", None) == "function_call_output":
                # message.pop("role")
                message = {k: v for k, v in message.items() if k != "role"}
                try:
                    image_message = message.pop("image_output")
                    new_messages.extend([message, image_message])
                except KeyError:
                    new_messages.append(message)
            else:
                new_messages.append(message)
        return prior_messages + new_messages  # new copy

    def get_actions(self, response: Response | None) -> list[ActionFromLLM]:
        """
        Process response from OpenAI Responses API
        """
        action_list = []
        if response is None:
            return action_list
        for output in response.output:
            if output.type == "function_call":
                try:
                    arguments = json.loads(output.arguments)
                    # text_args = ", ".join([f'{k}="{v}"' for k, v in arguments.items()])
                    # text_repr = f"{output.name}({text_args})
                    # BETTER: Make the text_repr a JSON function call format, e.g.,
                    # <tool_call>
                    # {"name": "expand", "arguments": {"result_id": "8868"}}
                    # </tool_call>
                    text_repr = json.dumps(
                        {"name": output.name, "arguments": arguments}
                    )
                    text_repr = f"<tool_call>\n{text_repr}\n</tool_call>"
                    action_list.append(
                        ActionFromLLM(
                            role="assistant",
                            type=output.type,
                            text=text_repr,
                            call_id=output.call_id,
                            name=output.name,
                            arguments=arguments,
                        )
                    )
                except JSONDecodeError:
                    print(f"JSONDecodeError: {output.arguments}")
                    # Treat as a message
                    name_error = "(no error)"
                    arguments_error = "(no error)"
                    try:
                        name_str = output.name
                    except Exception as name_e:
                        print(f"Name Exception: {name_e}")
                        name_str = "invalid_tool_call"
                        name_error = str(name_e)
                    try:
                        arguments_str = json.loads(output.arguments)
                    except Exception as arguments_e:
                        print(f"Arguments Exception: {arguments_e}")
                        arguments_str = output.arguments
                        arguments_error = str(arguments_e)
                    text_repr = json.dumps(
                        {"name": name_str, "arguments": arguments_str}
                    )
                    action_list.append(
                        ActionFromLLM(
                            role="assistant",  # role=output.role,
                            type="message",    # type="function_call",
                            text=text_repr,
                            call_id=None,
                            name=name_str,
                            arguments={
                                "name_error": name_error,
                                "arguments_error": arguments_error,
                            },
                        )
                    )
            elif output.type == "message":  # Regular message
                action_list.append(
                    ActionFromLLM(
                        role=output.role,
                        type=output.type,
                        text=output.content[0].text,
                        call_id=None,
                        name=None,
                        arguments=None,
                    )
                )
            elif output.type == "reasoning":
                try:
                    for summary in output.summary:
                        reasoning_text = summary.text
                        action_list.append(
                            ActionFromLLM(
                                role="assistant",
                                type=output.type,
                                text=reasoning_text,
                                call_id=None,
                                name=None,
                                arguments=None,
                            )
                        )
                except Exception as e:
                    print("-> Error with OpenAIResponsesLLM.get_actions:")
                    print(f"  -> output.type: {output.type}\n  -> error: {e}")
            else:
                raise ValueError(f"Unknown output type: {output.type}")
        return action_list


class AsyncOpenAIResponsesLLM(OpenAIResponsesLLM):
    """
    Async version of OpenAI Responses API
    """

    async def _sample_single_async(
        self,
        system_prompt: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        max_new_tokens: int | None = None,
        **generation_kwargs: Any,
    ) -> Response | None:
        """
        Generate a single response from the model
        """
        generation_kwargs = dict(self.generation_config, **generation_kwargs)
        if max_new_tokens is not None:
            generation_kwargs["max_output_tokens"] = max_new_tokens
        async with AsyncOpenAI(base_url=None) as _ac:
            try:
                response = await _ac.responses.create(
                    model=self.model,
                    instructions=system_prompt,
                    input=messages,
                    tools=tools,
                    **generation_kwargs,
                )
                # Track token usage *before* returning
                self.last_usage = self._track_tokens(getattr(response, "usage", None))
                if self.last_usage is not None:
                    print("OpenAI token usage:")
                    for k, v in self.last_usage.items():
                        print(f"-> {k}: {v}")
                return response
            except BadRequestError as e:
                if e.code == "invalid_prompt":
                    # log.exception("Openai invalid prompt error: %s", e)
                    print("Openai invalid prompt error: %s", e)
                    return None
                raise e

    async def _sample_batch_async(
        self,
        num_return_sequences: int = 1,
        **kwargs: Any,
        # ) -> list[Response | None]:
    ) -> Any:
        """
        Generate a batch of responses from the model
        """
        return await asyncio.gather(
            *[self._sample_single_async(**kwargs) for _ in range(num_return_sequences)],
            return_exceptions=False,
        )

    def sample(
        self,
        system_prompt: str,
        *,
        messages: list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        max_new_tokens: int | None = None,
        num_return_sequences: int = 1,
        **generation_kwargs: Any,
    ) -> list[Response | None]:
        """
        Generate a single response from the model
        """
        try:
            return asyncio.run(
                self._sample_batch_async(
                    system_prompt=system_prompt,
                    messages=messages,
                    tools=tools,
                    max_new_tokens=max_new_tokens,
                    num_return_sequences=num_return_sequences,
                    **generation_kwargs,
                )
            )
        except Exception as e:
            print(f"asyncio.run error: {e}")
            raise e
