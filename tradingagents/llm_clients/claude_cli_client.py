"""LLM client backed by the local Claude Code CLI.

Lets the framework consume a Claude Max / Pro subscription instead of paying
per-token via the Anthropic API. The Claude Agent SDK is intentionally NOT
used because it still bills the API; only the `claude` binary itself rides on
the subscription.

Each `invoke()` shells out to `claude -p ... --output-format json --tools ""`,
which:
  * `-p` runs in print (non-interactive) mode
  * `--output-format json` returns a parsable envelope with the full response
  * `--tools ""` disables Claude's built-in tools (Read, Bash, ...). We do not
    want the CLI poking at the user's filesystem; this adapter is a pure text
    generator that LangGraph orchestrates around.

Tool calling and structured output are mapped to the CLI's `--json-schema`
flag, which constrains the model's response to a JSON Schema. LangGraph's
ToolNode handles tool execution externally, so `bind_tools` only needs to
return an ``AIMessage`` whose ``tool_calls`` field is populated when the
model wants to call a tool.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import uuid
from typing import Any, Callable, List, Optional, Sequence, Type, Union

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)


# Claude CLI takes either a model alias ("opus", "sonnet", "haiku") or a full
# model id ("claude-sonnet-4-6"). We accept both verbatim, so any catalog
# validation should be permissive.
_DEFAULT_TIMEOUT = 600  # seconds — analyst calls can be long-running


class ChatClaudeCLI(BaseChatModel):
    """LangChain ChatModel that delegates to the local `claude` CLI.

    Each generation is a single ``subprocess.run`` of ``claude -p``. Multi-turn
    conversation history is rendered into a single text prompt rather than
    using ``--resume``: simpler, stateless, and matches how LangGraph already
    feeds the full message list back on every node invocation.
    """

    model: str = "sonnet"
    cli_path: str = "claude"
    timeout: int = _DEFAULT_TIMEOUT
    permission_mode: str = "bypassPermissions"
    extra_args: List[str] = Field(default_factory=list)

    # Set via bind_tools(); not a public field.
    _bound_tools: Optional[List[BaseTool]] = PrivateAttr(default=None)

    @property
    def _llm_type(self) -> str:
        return "claude_cli"

    # ---- public LangChain entry points ---------------------------------

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_prompt, user_prompt = self._render_messages(messages)

        if self._bound_tools:
            # If the conversation has not yet observed any tool results, force
            # the model to call a tool on this turn — otherwise it tends to
            # fabricate data and emit a final_answer immediately. Once at
            # least one ToolMessage exists, both branches are allowed.
            has_tool_results = any(isinstance(m, ToolMessage) for m in messages)
            ai_message = self._invoke_with_tools(
                system_prompt, user_prompt, allow_final=has_tool_results
            )
        else:
            text = self._run_cli(system_prompt, user_prompt)
            ai_message = AIMessage(content=text)

        return ChatResult(generations=[ChatGeneration(message=ai_message)])

    def bind_tools(
        self,
        tools: Sequence[Union[BaseTool, Callable, dict, Type[BaseModel]]],
        **kwargs: Any,
    ) -> "ChatClaudeCLI":
        normalized: List[BaseTool] = []
        for t in tools:
            if isinstance(t, BaseTool):
                normalized.append(t)
            else:
                # Only BaseTool instances carry the .name/.description/.args_schema
                # we need. Other shapes are uncommon in this repo; fail loudly so
                # the caller knows to wrap them with @tool.
                raise TypeError(
                    f"ChatClaudeCLI.bind_tools only accepts BaseTool instances; "
                    f"got {type(t).__name__}. Wrap functions with langchain_core.tools.tool."
                )
        new = self.model_copy()
        new._bound_tools = normalized
        return new

    def with_structured_output(
        self,
        schema: Union[Type[BaseModel], dict],
        **kwargs: Any,
    ) -> Runnable:
        """Return a Runnable that emits a Pydantic instance of ``schema``.

        Implemented by passing the schema as ``--json-schema`` so the CLI
        guarantees JSON-conforming output, then validating with Pydantic.
        """
        if isinstance(schema, type) and issubclass(schema, BaseModel):
            json_schema = _strip_unsupported_schema_keys(schema.model_json_schema())
            pydantic_cls: Optional[Type[BaseModel]] = schema
        elif isinstance(schema, dict):
            json_schema = _strip_unsupported_schema_keys(dict(schema))
            pydantic_cls = None
        else:
            raise TypeError(
                f"with_structured_output expects a Pydantic class or JSON schema dict; "
                f"got {type(schema).__name__}"
            )

        def _invoke(input_: Any) -> Any:
            messages = _coerce_to_messages(input_)
            system_prompt, user_prompt = self._render_messages(messages)
            raw = self._run_cli(system_prompt, user_prompt, json_schema=json_schema)
            data = _parse_json_loose(raw)
            if pydantic_cls is not None:
                return pydantic_cls.model_validate(data)
            return data

        return RunnableLambda(_invoke)

    # ---- internals -----------------------------------------------------

    def _invoke_with_tools(
        self, system_prompt: str, user_prompt: str, allow_final: bool = True
    ) -> AIMessage:
        """One generation step with tools bound.

        The CLI is asked to return JSON shaped as either
        ``{"tool_call": {"name": ..., "arguments": ...}}`` or
        ``{"final_answer": "..."}``. We translate the former into an
        ``AIMessage(tool_calls=[...])`` so LangGraph's ToolNode can dispatch,
        and the latter into a plain ``AIMessage(content=...)`` so the
        analyst's "no more tools" exit condition fires.
        """
        tool_descriptions = []
        for tool in self._bound_tools:
            arg_schema = (
                _strip_unsupported_schema_keys(tool.args_schema.model_json_schema())
                if tool.args_schema is not None
                else {"type": "object"}
            )
            tool_descriptions.append(
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "arguments_schema": arg_schema,
                }
            )

        tool_names = [t.name for t in self._bound_tools]
        kind_enum = ["tool_call", "final_answer"] if allow_final else ["tool_call"]
        # Anthropic's tool-input schema rejects top-level oneOf/allOf/anyOf, so
        # we use a flat discriminated shape: a `kind` field selects the branch
        # and the model fills in the matching payload field.
        response_schema = {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": kind_enum,
                    "description": (
                        "Set to 'tool_call' to invoke a tool"
                        + (", or 'final_answer' when you have everything you need to deliver the report."
                           if allow_final else " — no tool results are present yet, so a final answer is not allowed on this turn.")
                    ),
                },
                "tool_name": {
                    "type": "string",
                    "enum": tool_names,
                    "description": "Required when kind='tool_call'. Name of the tool to invoke.",
                },
                "tool_arguments": {
                    "type": "object",
                    "description": "Required when kind='tool_call'. Arguments object for the tool.",
                },
                "final_answer": {
                    "type": "string",
                    "description": "Required when kind='final_answer'. The full text report.",
                },
            },
            "required": ["kind"],
        }

        tool_directive = (
            "You have access to the following tools. You MUST call tools to "
            "gather data — you do not have live market data, prices, news, or "
            "financial figures in your training data. Fabricating tool "
            "outputs or claiming to have used a tool without actually "
            "emitting kind='tool_call' is forbidden and will be rejected.\n\n"
            "TOOLS:\n" + json.dumps(tool_descriptions, indent=2) + "\n\n"
            "Decision procedure:\n"
            "1. If the request needs ANY external data (prices, financials, "
            "news, fundamentals, indicators) and you have not yet observed "
            "the corresponding Tool result in the conversation above: emit "
            "kind='tool_call' with `tool_name` and `tool_arguments`. Pick "
            "ONE tool per turn; you will be called again with the result.\n"
            "2. If the system prompt asks for MULTIPLE items (e.g. \"select 8 "
            "indicators\", \"a list of\"), you must call the corresponding "
            "tool ONCE PER ITEM across multiple turns. One ToolMessage above "
            "is NOT enough to finish — count the items you were asked for and "
            "make sure each one has a matching Tool result above before "
            "moving on.\n"
            "3. Tools NEVER \"fail to load\" or become \"unavailable\". If a "
            "tool result is missing from the conversation above, it is "
            "because YOU have not called it yet. Do not claim a tool errored, "
            "did not return data, is unsupported in this session, or is "
            "unavailable — instead, emit a tool_call for it. Writing "
            "phrases like \"the tool failed\", \"could not be loaded\", "
            "\"is unavailable\", or \"system error\" in final_answer is "
            "forbidden when the Tool result for that call is simply not "
            "present above.\n"
            "4. Only emit kind='final_answer' once every tool you need has "
            "already returned a result you can see above. The final_answer "
            "must cite the data from those Tool results — never invent "
            "numbers, prices, or facts you did not observe.\n\n"
            "Respond ONLY with JSON conforming to the schema."
        )
        merged_system = (system_prompt + "\n\n" + tool_directive).strip()

        raw = self._run_cli(merged_system, user_prompt, json_schema=response_schema)
        data = _parse_json_loose(raw)

        kind = data.get("kind") if isinstance(data, dict) else None

        if kind == "tool_call":
            tool_name = data.get("tool_name", "")
            tool_args = data.get("tool_arguments") or {}
            if not isinstance(tool_args, dict):
                tool_args = {}
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "id": f"call_{uuid.uuid4().hex[:16]}",
                        "type": "tool_call",
                    }
                ],
            )

        if kind == "final_answer":
            return AIMessage(content=str(data.get("final_answer", "")))

        # Fallback: schema enforcement failed; treat whole output as final text.
        logger.warning(
            "Claude CLI response did not match expected tool/answer schema; "
            "treating as free-form final answer."
        )
        return AIMessage(content=raw)

    def _render_messages(self, messages: List[BaseMessage]) -> tuple[str, str]:
        """Flatten LangChain messages into (system_prompt, user_prompt) text.

        We concatenate system messages, then render the rest of the
        conversation as labeled turns inside a single user prompt. This is
        the simplest mapping that preserves tool-call <-> tool-result
        correspondences without juggling --input-format stream-json.
        """
        system_chunks: List[str] = []
        convo: List[str] = []

        for msg in messages:
            content = _stringify(msg.content)
            if isinstance(msg, SystemMessage):
                system_chunks.append(content)
            elif isinstance(msg, HumanMessage):
                convo.append(f"User:\n{content}")
            elif isinstance(msg, AIMessage):
                if msg.tool_calls:
                    rendered_calls = json.dumps(
                        [
                            {"name": tc["name"], "arguments": tc.get("args", {})}
                            for tc in msg.tool_calls
                        ],
                        indent=2,
                    )
                    convo.append(f"Assistant (called tools):\n{rendered_calls}")
                if content:
                    convo.append(f"Assistant:\n{content}")
            elif isinstance(msg, ToolMessage):
                tool_name = getattr(msg, "name", None) or "tool"
                convo.append(f"Tool result ({tool_name}):\n{content}")
            else:
                convo.append(f"{type(msg).__name__}:\n{content}")

        system_prompt = "\n\n".join(c for c in system_chunks if c).strip()
        user_prompt = "\n\n".join(convo).strip() or "Continue."
        return system_prompt, user_prompt

    def _run_cli(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: Optional[dict] = None,
    ) -> str:
        """Spawn the CLI and return the assistant's text payload."""
        cli = shutil.which(self.cli_path) or self.cli_path
        cmd: List[str] = [
            cli,
            "-p",
            "--output-format",
            "json",
            "--model",
            self.model,
            "--tools",
            "",  # disable Claude Code's built-in tools (Read, Bash, ...)
            "--permission-mode",
            self.permission_mode,
            "--no-session-persistence",
        ]
        if system_prompt:
            cmd.extend(["--append-system-prompt", system_prompt])
        if json_schema is not None:
            cmd.extend(["--json-schema", json.dumps(json_schema)])
        cmd.extend(self.extra_args)

        try:
            proc = subprocess.run(
                cmd,
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=os.environ,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Claude CLI timed out after {self.timeout}s for model {self.model}"
            ) from exc

        if proc.returncode != 0:
            raise RuntimeError(
                f"Claude CLI exited {proc.returncode}: {proc.stderr.strip() or proc.stdout.strip()}"
            )

        try:
            envelope = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Claude CLI returned non-JSON output: {proc.stdout[:500]}"
            ) from exc

        # Print mode wraps the response in a JSON envelope. The text payload is
        # at .result; some error envelopes also have .is_error / .error.
        if envelope.get("is_error"):
            raise RuntimeError(
                f"Claude CLI reported error: {envelope.get('error') or envelope.get('result')}"
            )

        # When --json-schema is supplied, the CLI returns the schema-conforming
        # object in `structured_output` and a human-readable summary in
        # `result`. Prefer the structured field for parseable downstream use.
        if json_schema is not None and isinstance(envelope.get("structured_output"), (dict, list)):
            return json.dumps(envelope["structured_output"])

        result = envelope.get("result", "")
        if not isinstance(result, str):
            result = json.dumps(result)
        return result


# ---- module-level helpers ---------------------------------------------


def _stringify(content: Any) -> str:
    """Normalize message.content (which may be a string or a list of blocks)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(item.get("text", ""))
                else:
                    parts.append(json.dumps(item))
        return "\n".join(p for p in parts if p)
    return str(content)


def _coerce_to_messages(input_: Any) -> List[BaseMessage]:
    """Accept the same input shapes LangChain's ChatModel.invoke does."""
    if isinstance(input_, str):
        return [HumanMessage(content=input_)]
    if isinstance(input_, BaseMessage):
        return [input_]
    if isinstance(input_, list):
        out: List[BaseMessage] = []
        for item in input_:
            if isinstance(item, BaseMessage):
                out.append(item)
            elif isinstance(item, dict):
                role = item.get("role")
                content = item.get("content", "")
                if role == "system":
                    out.append(SystemMessage(content=content))
                elif role in ("user", "human"):
                    out.append(HumanMessage(content=content))
                elif role in ("assistant", "ai"):
                    out.append(AIMessage(content=content))
                else:
                    out.append(HumanMessage(content=content))
            elif isinstance(item, tuple) and len(item) == 2:
                role, content = item
                if role == "system":
                    out.append(SystemMessage(content=content))
                elif role in ("user", "human"):
                    out.append(HumanMessage(content=content))
                elif role in ("assistant", "ai"):
                    out.append(AIMessage(content=content))
                else:
                    out.append(HumanMessage(content=str(content)))
            else:
                out.append(HumanMessage(content=str(item)))
        return out
    return [HumanMessage(content=str(input_))]


def _parse_json_loose(text: str) -> Any:
    """Parse JSON, tolerating optional ```json fences from the model."""
    s = text.strip()
    if s.startswith("```"):
        # Strip ```json ... ``` fences
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:]
        s = s.strip()
        if s.endswith("```"):
            s = s[:-3].strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Last-ditch: find the first { and last } and try that slice.
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(s[start : end + 1])
        raise


def _strip_unsupported_schema_keys(schema: dict) -> dict:
    """Recursively drop JSON Schema keys that confuse the CLI's validator.

    Pydantic emits things like ``$defs`` and ``title`` that are valid JSON
    Schema but trip the CLI's stricter mode. Inline ``$ref`` references to
    sibling ``$defs`` so the schema is self-contained.
    """
    defs = schema.get("$defs") or schema.get("definitions") or {}

    def _resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str):
                ref = node["$ref"]
                # e.g. "#/$defs/Foo"
                key = ref.split("/")[-1]
                target = defs.get(key)
                if target is not None:
                    return _resolve(target)
                return {"type": "object"}
            return {
                k: _resolve(v)
                for k, v in node.items()
                if k not in ("$defs", "definitions", "title")
            }
        if isinstance(node, list):
            return [_resolve(item) for item in node]
        return node

    return _resolve(schema)


# ---- BaseLLMClient wrapper (consumed by the factory) ------------------


class ClaudeCLIClient(BaseLLMClient):
    """Adapter that fits ChatClaudeCLI into the existing factory pattern."""

    provider = "claude_cli"

    _PASSTHROUGH = ("timeout", "cli_path", "permission_mode", "extra_args", "callbacks")

    def get_llm(self) -> Any:
        self.warn_if_unknown_model()
        kwargs: dict = {"model": self.model}
        for key in self._PASSTHROUGH:
            if key in self.kwargs:
                kwargs[key] = self.kwargs[key]
        return ChatClaudeCLI(**kwargs)

    def validate_model(self) -> bool:
        # Claude CLI accepts aliases ("opus", "sonnet", "haiku") and any
        # full model id; defer validation to the CLI itself.
        return True
