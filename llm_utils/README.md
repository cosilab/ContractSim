# LLM utilities

`llm_utils` contains phase-neutral code used by RatConBench execution phases:

- `environments.py` loads resolved environment and prompt-context documents.
- `prompting.py` renders templates, delimits contracts, and formats system
  prompts.
- `artifacts.py` performs strict JSON conversion, atomic private writes, and
  exclusive run-status claims.
- `process_runner.py` executes bounded subprocess jobs, cleans their process
  groups on interruption, and gives leaf workers a signal-to-exception guard
  so run claims can finalize before exit.
- `llm/` contains provider configuration and adapters, Concordia agent
  construction and persistence, runtime assembly, and trace recording.

It does not own phase policy, prompt templates, contract semantics, game rules,
or experiment planning. Phase packages may depend on `llm_utils`; `llm_utils` must
not depend on an execution phase such as `negotiation`.

## Ownership

```text
llm_utils/
├── artifacts.py      strict writes and exclusive run lifecycle
├── environments.py   resolved environment and prompt-context loading
├── process_runner.py bounded subprocess lifecycle and termination guards
├── prompting.py      template rendering and prompt-safe formatting
└── llm/
    ├── config.py     provider and runtime configuration
    ├── providers.py  Gemini, OpenAI, and Anthropic adapters
    ├── agents.py     Concordia agent construction and state persistence
    ├── factory.py    ordered model/agent runtime assembly
    └── tracing.py    structured model-call traces
```

Callers that restore a collection of agents supply their own
`expected_roles`; role sets such as `customer`/`caterer` are phase policy and
are not hard-coded here.

## Migration status

Negotiation uses the shared provider adapters, model/agent assembly, agent state
I/O, tracing, atomic artifact writes, and run claiming. It also delegates
generic subprocess supervision to `process_runner.py`, while
retaining its provider matrix and summary interpretation locally. Prompt
composition, validation, protocol rules, contracts, and game mechanics remain
phase-owned.

## Tests

Run the offline tests with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s llm_utils/tests -t . -p 'test_*.py'
```
