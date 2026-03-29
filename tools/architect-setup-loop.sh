#!/bin/bash
# Activate the architect polling loop by writing the state file.
# The stop hook reads this file and re-injects the polling prompt after each turn.
#
# Usage:
#   tools/architect-setup-loop.sh [--poll-interval <sec>] [--max-iterations <n>]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_FILE="$REPO_ROOT/.claude/architect-loop.local.md"

POLL_INTERVAL=30
MAX_ITERATIONS=200

while [[ $# -gt 0 ]]; do
    case "$1" in
        --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
        --max-iterations) MAX_ITERATIONS="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [ -f "$STATE_FILE" ]; then
    echo "Architect loop is already active."
    cat "$STATE_FILE"
    exit 0
fi

mkdir -p "$(dirname "$STATE_FILE")"

cat > "$STATE_FILE" <<EOF
active: true
iteration: 0
max_iterations: ${MAX_ITERATIONS}
poll_interval: ${POLL_INTERVAL}

## Polling prompt

Check status and progress of all active definition tasks:

1. Run: tools/architect status
2. Run: tools/architect progress
3. For each bench- worktree, evaluate and act:

| Condition | Action |
|-----------|--------|
| Agent stopped + all PRs open + clean | Report ready: suggest tools/architect remove <name> -y |
| Agent stopped + dirty files | Rescue: tools/architect rescue <name> --action commit -m "WIP: checkpoint" |
| Agent running | Report progress summary |
| No agent + has task spec | Report: ready to spawn, suggest tools/architect spawn <name> |

4. Report one-line summary per definition
5. When no active worktrees remain, output: <promise>ALL_TASKS_DONE</promise>
EOF

echo "Architect loop activated (poll_interval=${POLL_INTERVAL}s, max_iterations=${MAX_ITERATIONS})."
echo "State file: $STATE_FILE"
echo "To cancel: /cancel-architect"
