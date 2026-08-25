# Performance

Runs the execution phase of synthetic catering contracts. A customer makes
payments while a supplier buys ingredients and produces dishes over the
environment's 11-week schedule. Either role can be an LLM or a rational
baseline.

## Layout

```text
performance/
├── baselines/        customer and supplier RC, RE, and RCC agents
├── domain-prompts/   customer and supplier performance prompts
├── engine/           environment state and execution loop
├── logic/            LLM action prompts and response parsing
├── orchestration/    public CLI, batch scheduler, and worker
├── prompting/        prompt builder and goal treatments
├── contracts.py      contract-environment validation
└── tests/            performance tests
```

Shared environment loading, provider adapters, agent construction, artifact
writing, and subprocess supervision live in `llm_utils/`.

## Contracts

The generated policy suite is under
`algorithms/synth-contracts/data/policy-suite/`:

```text
algorithms/synth-contracts/data/policy-suite/<contract-id>/
├── contract.txt
├── terms.json
├── config.json
└── baselines/
    └── rcc/
        └── policy.npz
```

`config.json` must declare an integer `environment_id` from 1 through 6. Batch
runs select contracts matching the requested environment, and every worker
validates the environment again before execution. Supplier RC and RCC use the
policy stored at `baselines/rcc/policy.npz`; supplier RE and LLM suppliers do
not read a solved policy.

Batch discovery excludes the `payment_deduction` and
`substitution_payment_deduction` variants for the current 180-contract
campaign. Their rollover variants remain included. Direct single-contract runs
are unaffected.

`algorithms/synth-contracts/generation/` contains the contract-band and
policy-suite generators, the Lambda runner, and the contract-text renderer.

## Participants

The available rational baselines are:

- `rc`: Rational Complier
- `re`: Rational Exploiter
- `rcc`: Rational Conditional Complier

The available LLM providers are `gemini`, `openai`, and `anthropic`. The
performance role named `supplier` is the caterer from the catering domain.

Live LLM runs require the corresponding provider API keys. Gemini credentials
are currently also required for agent memory embeddings.

## Running

Run one contract and matchup:

```bash
.venv/bin/python performance/orchestration/run.py single \
  --env 1 \
  --contract-dir algorithms/synth-contracts/data/policy-suite/<contract-id> \
  --customer rc \
  --supplier openai \
  --seed 42
```

Run LLM suppliers against every customer baseline with one seed:

```bash
.venv/bin/python performance/orchestration/run.py batch \
  --envs 1 2 3 4 5 6 \
  --contracts-root algorithms/synth-contracts/data/policy-suite \
  --customers rc re rcc \
  --suppliers openai anthropic gemini \
  --seeds 42 \
  --max-concurrency 60
```

Batch jobs are the Cartesian product:

```text
matching contracts × customers × suppliers × seeds
```

Requested environments run in series. RatConBench waits for every job in one
environment to finish before starting the next environment. Jobs within the
active environment run as separate worker processes, bounded by
`--max-concurrency`. If the limit is omitted, all jobs for that environment may
run concurrently.

Use `--env ID` for one batch environment, `--envs ID ...` for an ordered
multi-environment campaign, or omit both to process all six environments in
order. Use `--dry-run` to print the worker commands without creating outputs or
starting workers.

Named prompt treatments are selected with `--goal-treatment`. A named treatment
requires exactly one LLM participant. Control runs are written under
`results/performance-main/`; treated runs use `results/performance-abl/`.

## Execution

Each worker owns one contract, customer, supplier, seed, and environment. It
loads the contract and environment, builds any required LLM agents or
baselines, and executes payment and production weeks sequentially. Environment
configuration controls starting capital, private values, prices, delivery
loss, spoilage, inventory limits, and price resampling.

Parallelism exists between complete matches only. Weeks and decisions within a
single match remain sequential.

## Outputs

One campaign is organized by environment, contract, and matchup:

```text
results/performance-main/<timestamp>/
└── env-<id>/
    └── <contract-id>/
        └── customer-<name>_supplier-<name>_seed-<seed>/
            ├── agents/       saved LLM states when present
            ├── config/       rendered prompts and run configuration
            ├── data/
            │   ├── llm_reasoning_traces.jsonl   LLM runs only
            │   └── summary.json
            ├── run_status.json
            └── stdout.log
```

`summary.json` contains the payment and production histories, final balances,
deliveries, leftover inventory, contract-target fulfilment, baseline summaries,
and environment metadata. For LLM participants, the reasoning trace log records
each payment, purchase, or production model call, including retries.

## Tests

The current tests cover environment mechanics, deterministic stochastic
replay, prompting, LLM action parsing, RC/RE/RCC and contingency behavior,
contract validation, orchestration, and baseline-only worker runs.

Run the performance tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s performance/tests -t . -p 'test_*.py'
```
