# Resolved environments

This directory contains the canonical benchmark environments used by prompt
construction and contract evaluation. Each numbered directory contains:

- `environment.json`: the complete machine-readable environment used by the
  solver.
- `prompt-context.json`: the approved natural-language substitutions used to
  render negotiation prompts.

These files intentionally represent the same environment in different forms:
executable mechanics versus controlled text exposed to agents.

`manifest.json` defines the canonical six-environment set, with environment
IDs 1 through 6. `shared/` contains the canonical components from which the
resolved snapshots were generated. Builder, validator, probability-kernel,
and solver code is intentionally excluded.
