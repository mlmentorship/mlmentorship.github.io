# Respawn directive: finish Task 4.1 from checkpoint 1d2e720f

The previous loop crashed before Copilot executed Task 4.1. Cadence passed an accumulated prompt larger than Linux `ARG_MAX`, causing `OSError: [Errno 7] Argument list too long: 'copilot'`. This is an orchestration/context-size failure, not an application or validation failure.

- Resume from checkpoint `1d2e720f`; Agent 5's prior audit work is committed and the worktree was clean at diagnosis.
- Work only on Task 4.1, the sole remaining task. Do not replay completed tasks or their idle/dependency history.
- Keep the initial context compact: read only Task 4.1 from the on-disk PRD. Do not inline the complete PRD, session log, shared context, all 106 modules, generated articles, or verbose command output.
- Execute the strict reviewed gate, checks, both build configurations, and browser/render/interaction audit with existing scripts. Run audit work in bounded batches and store verbose logs or capture inventories on disk.
- Open only files implicated by a reproduced failure. Fix at the owning problem or shared primitive, rerun the focused check, then rerun the required global gates.
- Avoid copying prior audit summaries into the prompt; checkpoint history is available through Git when specifically needed.
- Commit all Task 4.1 fixes and audit artifacts with a descriptive Cadence commit before marking the task complete.
