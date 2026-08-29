# Public feature flags

## Preparation subsystem

`PUBLIC_PREP_TOOLS` is a build-time flag for all preparation plans and tools.

| Value | Behavior |
|---|---|
| unset or `true` | Shows the Workbook entry point, builds `/prep/*`, and enables question Practice Mode. |
| `false` (or any value other than exact `true`) | Hides all Prep entry points and Practice Mode; `/prep/*` and legacy prep URLs redirect to `/questions/`. |

### Local build

Enabled (default):

    npm run build

Disabled:

    PUBLIC_PREP_TOOLS=false npm run build

### GitHub Pages

Set the Actions repository variable `PUBLIC_PREP_TOOLS` to `false` to disable the
subsystem on the next deploy. Delete the variable or set it to `true` to enable it.
The deploy workflow defaults to `true` so existing production behavior is stable.

The flag is intentionally build-time: disabled pages and controls do not remain
interactive in the generated site. The core Questions, Guides, and Concepts library
is unaffected in either state.