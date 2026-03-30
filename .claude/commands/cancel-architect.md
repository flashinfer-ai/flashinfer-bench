---
description: Cancel active architect loop
allowed-tools: Bash(test -f .claude/architect-loop.local.md*), Bash(rm .claude/architect-loop.local.md)
---

# Cancel Architect Loop

To cancel the architect loop:

1. Check if `.claude/architect-loop.local.md` exists using Bash: `test -f .claude/architect-loop.local.md && echo "EXISTS" || echo "NOT_FOUND"`

2. **If NOT_FOUND**: Say "No active architect loop found."

3. **If EXISTS**:
   - Read `.claude/architect-loop.local.md` to get the current iteration number from the `iteration:` field
   - Remove the file using Bash: `rm .claude/architect-loop.local.md`
   - Report: "Cancelled architect loop (was at iteration N)" where N is the iteration value
