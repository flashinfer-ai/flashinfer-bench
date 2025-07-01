# Agent Inspector

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
