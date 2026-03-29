"""Parse stream-JSON agent logs into structured messages."""

import json
import os


def parse_agent_log(log_path: str) -> list[dict]:
    """Parse stream-JSON agent log into structured messages.

    Each line is a JSON object with a 'type' field. Returns a list of
    structured dicts suitable for rendering in the dashboard.
    """
    if not os.path.isfile(log_path):
        return []

    messages = []
    session_info = {}

    with open(log_path) as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            entry_type = entry.get("type")

            if entry_type == "system":
                msg = _parse_system(entry)
                if msg:
                    if entry.get("subtype") == "init":
                        session_info = msg.get("session_info", {})
                    messages.append(msg)

            elif entry_type == "assistant":
                for msg in _parse_assistant(entry):
                    messages.append(msg)

            elif entry_type == "user":
                for msg in _parse_user(entry):
                    messages.append(msg)

            elif entry_type == "result":
                msg = _parse_result(entry)
                if msg:
                    messages.append(msg)

    return messages


def get_log_summary(messages: list[dict]) -> dict:
    """Extract summary info from parsed messages."""
    summary = {
        "total_messages": len(messages),
        "assistant_messages": 0,
        "tool_calls": 0,
        "tool_results": 0,
        "num_turns": None,
        "total_cost_usd": None,
        "duration_ms": None,
        "session_id": None,
        "model": None,
        "result_text": None,
    }

    for msg in messages:
        if msg["type"] == "assistant_text":
            summary["assistant_messages"] += 1
        elif msg["type"] == "tool_call":
            summary["tool_calls"] += 1
        elif msg["type"] == "tool_result":
            summary["tool_results"] += 1
        elif msg["type"] == "result":
            summary["num_turns"] = msg.get("num_turns")
            summary["total_cost_usd"] = msg.get("total_cost_usd")
            summary["duration_ms"] = msg.get("duration_ms")
            summary["result_text"] = msg.get("content")
        elif msg["type"] == "system_init":
            summary["session_id"] = msg.get("session_info", {}).get("session_id")
            summary["model"] = msg.get("session_info", {}).get("model")

    return summary


def _parse_system(entry: dict) -> dict | None:
    """Parse system entries (init, notifications)."""
    subtype = entry.get("subtype", "")

    if subtype == "init":
        return {
            "type": "system_init",
            "content": "Session initialized",
            "session_info": {
                "session_id": entry.get("session_id"),
                "model": entry.get("model"),
                "cwd": entry.get("cwd"),
                "tools": entry.get("tools", []),
                "claude_code_version": entry.get("claude_code_version"),
            },
        }

    return {
        "type": "system",
        "subtype": subtype,
        "content": entry.get("message", str(entry)),
    }


def _parse_assistant(entry: dict) -> list[dict]:
    """Parse assistant messages. May contain text and/or tool_use blocks."""
    messages = []
    msg_data = entry.get("message", {})
    content_blocks = msg_data.get("content", [])

    for block in content_blocks:
        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "").strip()
            if text:
                messages.append(
                    {
                        "type": "assistant_text",
                        "content": text,
                    }
                )

        elif block_type == "tool_use":
            tool_name = block.get("name", "unknown")
            tool_input = block.get("input", {})
            tool_id = block.get("id", "")

            messages.append(
                {
                    "type": "tool_call",
                    "tool_name": tool_name,
                    "tool_id": tool_id,
                    "tool_input": tool_input,
                    "tool_input_summary": _summarize_tool_input(tool_name, tool_input),
                    "content": f"Tool call: {tool_name}",
                }
            )

    return messages


def _parse_user(entry: dict) -> list[dict]:
    """Parse user messages. May contain tool_result or text."""
    messages = []
    msg_data = entry.get("message", {})
    content = msg_data.get("content", [])

    # Content can be a string or a list of blocks
    if isinstance(content, str):
        if content.strip():
            messages.append(
                {
                    "type": "user_text",
                    "content": content.strip(),
                }
            )
        return messages

    for block in content:
        if not isinstance(block, dict):
            continue

        block_type = block.get("type")

        if block_type == "tool_result":
            tool_use_id = block.get("tool_use_id", "")
            result_content = block.get("content", "")

            # Content can be string or list of content blocks
            if isinstance(result_content, list):
                parts = []
                for part in result_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                result_text = "\n".join(parts)
            else:
                result_text = str(result_content)

            messages.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": result_text,
                    "content_truncated": _truncate(result_text, 500),
                    "content_preview": _truncate(result_text, 200),
                    "is_error": block.get("is_error", False),
                }
            )

        elif block_type == "text":
            text = block.get("text", "").strip()
            if text:
                messages.append(
                    {
                        "type": "user_text",
                        "content": text,
                    }
                )

    return messages


def _parse_result(entry: dict) -> dict | None:
    """Parse session result entry."""
    return {
        "type": "result",
        "subtype": entry.get("subtype", "unknown"),
        "is_error": entry.get("is_error", False),
        "num_turns": entry.get("num_turns"),
        "total_cost_usd": entry.get("total_cost_usd"),
        "duration_ms": entry.get("duration_ms"),
        "content": entry.get("result", ""),
        "session_id": entry.get("session_id"),
    }


def _summarize_tool_input(tool_name: str, tool_input: dict) -> str:
    """Create a short summary of tool input for display."""
    if not tool_input:
        return ""

    if tool_name == "Read":
        return tool_input.get("file_path", "")
    elif tool_name == "Write":
        path = tool_input.get("file_path", "")
        content = tool_input.get("content", "")
        return f"{path} ({len(content)} chars)"
    elif tool_name == "Edit":
        return tool_input.get("file_path", "")
    elif tool_name == "Bash":
        cmd = tool_input.get("command", "")
        return _truncate(cmd, 120)
    elif tool_name == "Glob":
        return tool_input.get("pattern", "")
    elif tool_name == "Grep":
        pattern = tool_input.get("pattern", "")
        path = tool_input.get("path", "")
        return f"{pattern}" + (f" in {path}" if path else "")
    elif tool_name == "Task":
        return tool_input.get("description", "")
    elif tool_name == "TodoWrite":
        todos = tool_input.get("todos", [])
        return f"{len(todos)} items"
    elif tool_name == "WebFetch":
        return tool_input.get("url", "")
    elif tool_name == "WebSearch":
        return tool_input.get("query", "")
    else:
        # Generic: show first key-value pair
        for k, v in tool_input.items():
            return f"{k}: {_truncate(str(v), 80)}"
    return ""


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
