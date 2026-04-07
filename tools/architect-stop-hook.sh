#!/bin/bash
# Architect Loop Stop Hook for flashinfer-bench.
# Re-injects the polling prompt after each Claude turn when a loop is active.
# Register in .claude/settings.json under hooks.Stop.

set -euo pipefail

HOOK_INPUT=$(cat)
STATE_FILE=".claude/architect-loop.local.md"

if [[ ! -f "$STATE_FILE" ]]; then
  exit 0
fi

ITERATION=$(grep '^iteration:' "$STATE_FILE" | sed 's/iteration: *//')
MAX_ITERATIONS=$(grep '^max_iterations:' "$STATE_FILE" | sed 's/max_iterations: *//')
POLL_INTERVAL=$(grep '^poll_interval:' "$STATE_FILE" | sed 's/poll_interval: *//')
POLL_INTERVAL="${POLL_INTERVAL:-30}"

if [[ ! "$ITERATION" =~ ^[0-9]+$ ]] || [[ ! "$MAX_ITERATIONS" =~ ^[0-9]+$ ]]; then
  echo "Warning: architect loop state corrupted. Stopping." >&2
  rm "$STATE_FILE"
  exit 0
fi

if [[ $MAX_ITERATIONS -gt 0 ]] && [[ $ITERATION -ge $MAX_ITERATIONS ]]; then
  echo "Architect loop: max iterations ($MAX_ITERATIONS) reached." >&2
  rm "$STATE_FILE"
  exit 0
fi

TRANSCRIPT_PATH=$(echo "$HOOK_INPUT" | jq -r '.transcript_path')
if [[ ! -f "$TRANSCRIPT_PATH" ]]; then
  echo "Warning: transcript not found. Stopping." >&2
  rm "$STATE_FILE"
  exit 0
fi

# Check if agent output contains ALL_TASKS_DONE promise
LAST_LINE=$(grep '"role":"assistant"' "$TRANSCRIPT_PATH" 2>/dev/null | tail -1)
if [[ -n "$LAST_LINE" ]]; then
  LAST_TEXT=$(echo "$LAST_LINE" | jq -r '.message.content | map(select(.type=="text")) | map(.text) | join("\n")' 2>/dev/null || echo "")
  if echo "$LAST_TEXT" | grep -q '<promise>ALL_TASKS_DONE</promise>'; then
    echo "Architect loop: all tasks done." >&2
    rm "$STATE_FILE"
    exit 0
  fi
fi

NEXT_ITERATION=$((ITERATION + 1))
TEMP_FILE="${STATE_FILE}.tmp.$$"
sed "s/^iteration: .*/iteration: $NEXT_ITERATION/" "$STATE_FILE" > "$TEMP_FILE"
mv "$TEMP_FILE" "$STATE_FILE"

PROMPT_TEXT=$(awk '/^## Polling prompt/{found=1; next} found' "$STATE_FILE")

if [[ -z "$PROMPT_TEXT" ]]; then
  echo "Warning: no prompt in state file. Stopping." >&2
  rm "$STATE_FILE"
  exit 0
fi

if [[ "$POLL_INTERVAL" =~ ^[0-9]+$ ]] && [[ "$POLL_INTERVAL" -gt 0 ]]; then
  sleep "$POLL_INTERVAL"
fi

SYSTEM_MSG="Architect polling iteration $NEXT_ITERATION / $MAX_ITERATIONS | /cancel-architect to stop | output <promise>ALL_TASKS_DONE</promise> when all definitions are done"

jq -n \
  --arg prompt "$PROMPT_TEXT" \
  --arg msg "$SYSTEM_MSG" \
  '{
    "decision": "block",
    "reason": $prompt,
    "systemMessage": $msg
  }'
