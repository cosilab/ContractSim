# Negotiation

Runs customer–caterer negotiations between Gemini, OpenAI, and Anthropic
agents. It builds private role prompts, manages the negotiation protocol, and
saves the outcome, transcript, reasoning traces, and agent states.

## Layout

```text
negotiation/
├── domain-prompts/catering/  customer and caterer templates
├── logic/                    negotiation loop and response parsing
├── orchestration/            CLI, batch scheduler, and worker
├── prompting/                prompt builder and goal treatments
└── tests/                    negotiation tests
```

Shared provider adapters, agents, tracing, and artifact helpers live in
`llm_utils/`.

## Running

Install the repository dependencies and provide `GEMINI_API_KEY`,
`OPENAI_API_KEY`, and `ANTHROPIC_API_KEY` as needed. Gemini credentials are
currently required for agent memory embeddings in every live run.

Run one provider pairing:

```bash
.venv/bin/python negotiation/orchestration/run.py single \
  --envs 1 \
  --match openai anthropic \
  --max-rounds 50
```

Run the complete ordered provider matrix:

```bash
.venv/bin/python negotiation/orchestration/run.py batch \
  --envs 1 \
  --providers gemini openai anthropic \
  --runs 1 \
  --max-rounds 3
```

The full matrix contains nine pairings, including same-provider pairings.
Environments run in series; pairings within an environment run concurrently.
Use `--max-concurrency` to limit workers and `--dry-run` to print the plan.

Named prompt treatments are selected with `--goal-treatment`. Control runs are
written below `results/negotiation-main/`; treated runs use
`results/negotiation-abl/`.

## Outputs

Each pairing writes under its campaign, environment, and replicate:

```text
<campaign>/env-<id>/run-<n>/<customer>_vs_<caterer>/
├── agents/       saved customer and caterer states
├── config/       rendered prompts and run configuration
├── data/
│   ├── contract.txt                 accepted runs only
│   ├── llm_reasoning_traces.jsonl   always written
│   ├── negotiation_log.txt
│   └── summary.json
├── run_status.json
└── stdout.log
```

`summary.json` is the authoritative negotiation result. Reasoning traces record
every model call, although a provider may not expose a reasoning summary.

## Tests

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s negotiation -p 'test_*.py'
```
