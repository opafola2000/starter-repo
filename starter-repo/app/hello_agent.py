"""Day 1 H4 - minimal tool-using agent. Swap Bedrock <-> Vertex via MONK_MODEL."""
from __future__ import annotations

import os
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

DEFAULT_MODEL = "bedrock_converse:openai.gpt-oss-120b-1:0"


@tool
def get_weather(city: str) -> str:
    """Return a canned weather report for a city."""
    return f"It's 28C and sunny in {city}."


@tool
def search_news(topic: str) -> str:
    """Return a canned news headline for a topic."""
    return f"Top story on {topic}: AI agents are eating tools."


TOOLS = [get_weather, search_news]
TOOL_BY_NAME = {t.name: t for t in TOOLS}


def _chat_model():
    name = os.getenv("MONK_MODEL", DEFAULT_MODEL)
    kwargs: dict[str, Any] = {}
    if name.startswith("bedrock"):
        kwargs["region_name"] = os.getenv("AWS_REGION", "us-east-1")
        kwargs["max_tokens"] = 512
    elif name.startswith("google_vertexai"):
        kwargs["project"] = os.environ.get("GCP_PROJECT")
        kwargs["location"] = os.getenv("GCP_LOCATION", "us-central1")
        kwargs["max_output_tokens"] = 512
    return init_chat_model(name, **kwargs).bind_tools(TOOLS)


def agent_run(question: str) -> str:
    model = _chat_model()
    messages: list[BaseMessage] = [HumanMessage(content=question)]
    for _ in range(6):
        ai_msg = model.invoke(messages)
        messages.append(ai_msg)
        if not ai_msg.tool_calls:
            content = ai_msg.content
            return content if isinstance(content, str) else str(content)
        for tc in ai_msg.tool_calls:
            result = TOOL_BY_NAME[tc["name"]].invoke(tc["args"])
            messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))
    return "Max iterations reached without a final answer."


if __name__ == "__main__":
    print(
        agent_run(
            "What is the weather in Bangalore today and what's the latest AI news?"
        )
    )
