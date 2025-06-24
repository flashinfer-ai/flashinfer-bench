# Agent Inspector

## Overview

This document describes a JSON schema for the Agent Inspector, which aims to capture and visualize traces of agent interactions, including:
1. LLM requests and responses (input/output)
2. Tool calls (MCP and others)
3. Multi-agent conversations (different models, different contexts)

The schema represents traces as a list of objects, where each object represents either an LLM call or a tool call in chronological order.

## JSON Schema Description

### Top-Level Structure

The agent inspector data is represented as a JSON array of trace objects:

```json
[
  {
    "type": "llm_request",
    // ... llm_request fields
  },
  {
    "type": "tool_call",
    // ... tool_call fields
  },
  {
    "type": "mcp",
    // ... mcp fields
  }
]
```

Each trace object must have a `type` field that determines its structure.

### `type`: `llm_request`

Represents an LLM request-response interaction.

| Field          | Type   | Required | Description                                    |
|----------------|--------|----------|------------------------------------------------|
| `type`         | string | Yes      | Must be `"llm_request"`                        |
| `conversation` | array  | Yes      | Complete conversation history sent to the LLM  |
| `response`     | object | Yes      | LLM response following OpenAI format           |
| `model`        | string | Yes      | Model identifier (e.g., "gpt-4", "claude-3")   |
| `annotation`   | string | No       | User-specified annotation for this LLM request |

#### `conversation` Field

The conversation field follows the [OpenAI Chat API message list format](https://platform.openai.com/docs/api-reference/chat/message-list). Each message in the array has:

| Field     | Type   | Required | Description                                   |
|-----------|--------|----------|-----------------------------------------------|
| `role`    | string | Yes      | One of: "system", "user", "assistant", "tool" |
| `content` | string | Yes      | The message content                           |

##### `role` Field

The role field is one of the following values:

- `"system"`
- `"user"`
- `"assistant"`
- `"tool"`

Example:

```json
[
    {
        "role": "system",
        "content": "You are a helpful assistant."
    },
    {
        "role": "user",
    }
]
```

#### `response` Field

The response field follows the [OpenAI Chat Completion response format](https://platform.openai.com/docs/api-reference/chat/object), containing the LLM's output and any tool calls.

SGLang/Vllm's conversation format may slightly differ from OpenAI's. We should convert their's conversations to OpenAI's format.

Now we only consider the following fields:

Example:

```json
{
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "The image shows ...",
                "tool_calls": [
                    {
                        "id": "call_123",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"location\": \"New York, NY\"}"
                        }
                    }
                ],
            },
            "finish_reason": "stop"
        }
    ],
}
```

### `type`: `tool_call`

Represents a tool call execution.

| Field        | Type   | Required | Description                                                  |
|--------------|--------|----------|--------------------------------------------------------------|
| `type`       | string | Yes      | Must be `"tool_call"`                                        |
| `tool_name`  | string | Yes      | Name of the called tool                                      |
| `arguments`  | object | Yes      | Tool arguments in json object (name-value pairs)             |
| `result`     | string | Yes      | Tool execution result in string format (potentially in list) |
| `cli_output` | string | No       | Captured stdout/stderr from tool execution                   |

### `type`: `mcp`

Represents an MCP (Model Context Protocol) interaction.

| Field  | Type   | Required | Description     |
|--------|--------|----------|-----------------|
| `type` | string | Yes      | Must be `"mcp"` |

*Note: MCP schema is to be determined (TBD).*

## Examples

### Example 1: Simple LLM Request

```json
[
  {
    "type": "llm_request",
    "conversation": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "What is the capital of France?"
      },
    ],
    "response": {
      "model": "gpt-4o",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "The capital of France is Paris."
          },
          "finish_reason": "stop"
        }
      ]
    },
    "model": "gpt-4o",
    "annotation": "Basic geography question"
  }
]
```

### Example 2: LLM Request with Tool Call

```json
[
  {
    "type": "llm_request",
    "conversation": [
      {
        "role": "user",
        "content": "What's the weather like in New York?"
      },
    ],
    "response": {
      "model": "gpt-4o",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "I'll check the weather in New York for you.",
            "tool_calls": [
              {
                "id": "call_123",
                "type": "function",
                "function": {
                  "name": "get_weather",
                  "arguments": "{\"location\": \"New York, NY\"}"
                }
              }
            ]
          },
          "finish_reason": "tool_calls"
        }
      ]
    },
    "model": "gpt-4o"
  },
  {
    "type": "tool_call",
    "tool_name": "get_weather",
    "arguments": {
      "location": "New York, NY"
    },
    "result": "Current weather in New York: 72°F, partly cloudy",
    "cli_output": "Weather API call successful\nStatus: 200 OK"
  }
]
```

### Example 3: Multi-Agent Conversation

```json
[
  {
    "type": "llm_request",
    "conversation": [
      {
        "role": "user",
        "content": "Analyze this code for security issues."
      },
      {
        "role": "assistant",
        "content": "I'll analyze the code for potential security vulnerabilities."
      }
    ],
    "response": {
      "model": "claude-3-sonnet",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "I'll analyze the code for potential security vulnerabilities."
          },
          "finish_reason": "stop"
        }
      ]
    },
    "model": "claude-3-sonnet",
    "annotation": "Security analysis agent"
  },
  {
    "type": "llm_request",
    "conversation": [
      {
        "role": "system",
        "content": "You are a code optimization specialist."
      },
      {
        "role": "user",
        "content": "Review this code for performance improvements."
      },
      {
        "role": "assistant",
        "content": "I'll review the code for optimization opportunities."
      }
    ],
    "response": {
      "model": "gpt-4o",
      "choices": [
        {
          "index": 0,
          "message": {
            "role": "assistant",
            "content": "I'll review the code for optimization opportunities."
          },
          "finish_reason": "stop"
        }
      ]
    },
    "model": "gpt-4o",
    "annotation": "Performance optimization agent"
  }
]
```

## GUI Design (Draft)

There will be three tabs: Kernel Signature, Result, Agent Inspector.

### Kernel Signature

See [kernel_signature.md](kernel_signature.md).

### Result

1. Code
2. Max Diff
3. Speed: run time, or timeout

### Agent Inspector

We show all trace elements in a list, each in a drop-down box.

Each string should be shown in a multi-line text box.

#### `type`: `llm_request`

Show the fields in the order:
1. Model Name
2. Conversation in a list of drop-down boxes
3. LLM output string
4. Tool calls in the llm output (if exists):
    1. Tool Name
    2. Arguments (key, value pairs)

#### `type`: `tool_call`

#### `type`: `mcp`

Show all the fields in the schema.
