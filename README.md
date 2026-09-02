# ContractSim

ContractSim is the research code for studying LLM negotiation and contract
performance in stochastic contracting environments. It contains negotiation
and execution experiments, rational baselines, an exact JAX contract solver,
synthetic-contract generation, Pareto-frontier search, domain-robustness
experiments, analysis pipelines, and publication plots.

> **Work in progress:** interfaces, paths, and generated artifacts may change
> while the research code is consolidated.

> **Terminology:** In the Catering domain, negotiation calls the producing role
> `caterer`, while performance calls it `supplier`; synthetic contracts use
> `caterer`. We have not observed this naming difference affecting LLM behavior.

> **Data availability:** Experiment results and derived analyses are currently
> too large to include in this repository. We are working on a way to share them.

## Repository map

- `negotiation/`: customer–supplier LLM negotiation protocol and orchestration.
- `performance/`: contract execution with LLM and rational participants.
- `domain-robustness/`: matched Hotel Cleaning and AI Model Hosting experiments.
- `algorithms/rc-solver/`: exact JAX solver and contingency implementations.
- `algorithms/pareto/`: F1/F2 Pareto-frontier generation and merged data.
- `algorithms/synth-contracts/`: synthetic-contract and policy-suite generation.
- `analysis/`: negotiation and performance inference pipelines.
- `paper-stuff/plots/`: reproducible publication plotting scripts and outputs.
- `llm_utils/`: shared provider, agent, prompting, tracing, and process helpers.
- `envs/`: six resolved benchmark environments and prompt contexts.
- `global/MC/`: Monte Carlo validation of exact contract values.
- `results/`: generated experiment runs.

## Setup

Python 3.12 is used for development.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Live runs use `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, and either
`GEMINI_API_KEY` or `GOOGLE_API_KEY`, as applicable. File-backed credentials
can instead be stored under the directory selected by `KEYS_DIR`. Gemini
credentials are also required by the shared agent-memory embedder.

Use `--help` on an entry point before launching a large campaign. The examples
below are intentionally small or dry-run friendly.

## Environments and contracts

`envs/manifest.json` defines benchmark environments 1–6. Each numbered
directory contains an executable `environment.json` and the corresponding
model-facing `prompt-context.json`.

Performance-ready contracts use this layout:

```text
suite/env-<evaluation>/p<band>/<contingency-variant>/
├── contract.txt
├── terms.json
├── config.json
├── result.json
└── baselines/rcc/policy.npz
```

Contract IDs retain the evaluation environment, source environment,
satisfaction-probability band, and contingency variant. Generation code lives
under `algorithms/synth-contracts/generation/`. To generate a fresh base and
policy suite:

```bash
.venv/bin/python algorithms/synth-contracts/generation/generate_all.py \
  results/synth-contracts/<run-id> --gpus 0 1
```

## Negotiation experiments

Negotiation supports ordered pairings of Gemini, OpenAI, and Anthropic agents.
A single pairing can be run with:

```bash
.venv/bin/python negotiation/orchestration/run.py single \
  --envs 1 \
  --match openai anthropic \
  --max-rounds 50
```

Run a provider matrix with:

```bash
.venv/bin/python negotiation/orchestration/run.py batch \
  --envs 1 \
  --providers gemini openai anthropic \
  --runs 1 \
  --max-rounds 3 \
  --dry-run
```

Named prompt treatments use `--goal-treatment`. Control campaigns are written
under `results/negotiation-main/`; treatments use
`results/negotiation-abl/`. Every pairing writes its configuration, agent
states, reasoning traces, negotiation log, run status, and authoritative
`data/summary.json`.

## Performance experiments

Available rational participants are Rational Complier (`rc`), Rational
Exploiter (`re`), and Rational Conditional Complier (`rcc`). A single contract
execution can be run with:

```bash
.venv/bin/python performance/orchestration/run.py single \
  --env 1 \
  --contract-dir results/synth-contracts/<run-id>/suite/env-01/p050/grim-trigger \
  --customer rc \
  --supplier openai \
  --seed 42
```

Inspect a batch without running it:

```bash
.venv/bin/python performance/orchestration/run.py batch \
  --envs 1 2 3 4 5 6 \
  --contracts-root results/synth-contracts/<run-id>/suite \
  --customers rc re rcc \
  --suppliers openai anthropic gemini \
  --seeds 42 \
  --max-concurrency 60 \
  --dry-run
```

Control campaigns are written under `results/performance-main/`; named prompt
treatments use `results/performance-abl/`. Each job writes its configuration,
status, logs, optional LLM state and traces, and authoritative
`data/summary.json`.

## Domain robustness

The fixed-Environment-5 robustness workflow reuses the main negotiation and
performance engines while translating terminology and money for Hotel
Cleaning and AI Model Hosting.

```bash
.venv/bin/python domain-robustness/run.py validate
.venv/bin/python domain-robustness/run.py plan all
.venv/bin/python domain-robustness/run.py run negotiation
.venv/bin/python domain-robustness/run.py run performance
.venv/bin/python domain-robustness/run.py run all
```

Use `--domains`, `--providers`, `--baselines`, `--seeds`, `--limit`,
`--contracts-root`, `--out-root`, and `--max-parallel` to narrow or scale a
run. Validation and dry runs do not call live model APIs.

Prepare completed domain-performance runs for analysis with:

```bash
.venv/bin/python domain-robustness/prepare_performance_analysis.py
```

## Exact RC solver

The solver enumerates delivery, spoilage, and observed-price outcomes and
solves finite contracts by backward dynamic programming; it does not use Monte
Carlo sampling. From `algorithms/rc-solver/`:

```bash
../../.venv/bin/python rc_solver.py \
  --environment tests/data/environment_09.json \
  --contract tests/data/rollover_contract.json \
  --sat-thresh 0.95 \
  --save-policy policy.npz
```

The result reports `P_sat`, `E_supplier_utility`, and `E_customer_utility`.
The base solver implements exact-production grim trigger; substitution,
payment deduction, rollover, and their combinations have separate selectors
and Bellman solvers under `contingencies/scripts/`.

## Pareto frontiers

Run both F1 orientations for an environment:

```bash
.venv/bin/python algorithms/pareto/generation/f1_frontier.py \
  --env 5 --sweep max-customer

.venv/bin/python algorithms/pareto/generation/f1_frontier.py \
  --env 5 --sweep max-supplier
```

Use `--resume` only with an unchanged generation command and compatible
checkpoint. Prepare and run targeted refinements, then merge the base and
refined sweeps with:

```bash
.venv/bin/python algorithms/pareto/generation/f1_refined_search_prereq.py
.venv/bin/python algorithms/pareto/generation/run_f1_refined_anchors.py \
  --env 5 --sweep max-customer
.venv/bin/python algorithms/pareto/generation/run_f1_refined_anchors.py \
  --env 5 --sweep max-supplier
.venv/bin/python algorithms/pareto/generation/f1_merge_base_refinement.py \
  --env 5
```

F1 base, refinement, and merged artifacts are stored under
`algorithms/pareto/data/env-XX/`. F2 generation and merged frontier artifacts
live alongside them.

## Analysis

Negotiation analysis is a three-stage pipeline: parse contract terms, build
contract-quality solves, then aggregate campaign statistics. Inspect the
available campaign and resume controls with:

```bash
.venv/bin/python analysis/negotiation/scripts/prepare_llm_data.py --help
.venv/bin/python analysis/negotiation/scripts/build_contract_qual.py --help
.venv/bin/python analysis/negotiation/scripts/analyze.py --help
```

Analyze the latest performance campaign or select one explicitly:

```bash
.venv/bin/python -m analysis.performance.scripts.analyze_campaign --latest

.venv/bin/python -m analysis.performance.scripts.analyze_campaign \
  --run-dir results/performance-main/<campaign>
```

Performance outputs contain `rows.csv` and `summary.json` under
`analysis/performance/inferences/`. Negotiation outputs contain parsed terms,
contract-quality records and solves, and `stats.json` under
`analysis/negotiation/inferences/`.

## Publication plots

Plotting programs live in `paper-stuff/plots/scripts/`, static assets in
`paper-stuff/plots/assets/`, and generated figures and audits in
`paper-stuff/plots/outputs/`. Relevant scripts include:

- `pareto-frontier-plots.py`
- `contract-quality-bar-plots.py`
- `negotiation-ablation-plots.py`
- `performance-ablation-plots.py`
- `performance-variation-plots.py`
- `domain-robustness.py`
- `ablation-guidance-figure.py`
- `merge-figure5.py`

For example, render a merged Environment-5 Pareto plot with:

```bash
.venv/bin/python paper-stuff/plots/scripts/pareto-frontier-plots.py \
  --input algorithms/pareto/data/env-05/f1-merged/f1_frontier.json \
  --frontier-output-dir algorithms/pareto/data/env-05/f1-merged \
  --f2-input algorithms/pareto/data/env-05/f2-merged/f2_frontier.json \
  --llm-stats analysis/negotiation/inferences/main/control/stats.json
```

## Validation and tests

Run the principal offline suites from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s llm_utils/tests -t . -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s negotiation -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s performance/tests -t . -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s analysis/performance/tests -t . -p 'test_*.py'
```

Run the centralized solver suite from `algorithms/rc-solver/`:

```bash
../../.venv/bin/python -m unittest discover -v -s tests
```

Independently validate exact control-campaign values with 50,000 Monte Carlo
trials per accepted contract:

```bash
.venv/bin/python global/MC/validate_contract_values.py
```

Use `--trials 5000 --limit 3` for a quick smoke check. Additional contingency
validation and parity tools are available under `global/MC/`.

## Generated data

The `results/`, analysis-inference, Pareto-data, plot-output, and paper-output
trees contain generated or imported artifacts and may be excluded by
`.gitignore`. Do not infer freshness from timestamps alone; regenerate derived
artifacts through their owning scripts when source data changes.

## Citation

```bibtex
@misc{sajja2026evaluatingrationalcontractingnatural,
  title={Evaluating Rational Contracting in Natural Language},
  author={Bhavyesh Sajja and Max Kleiman-Weiner and Roger Zimmermann and Tan Zhi-Xuan},
  year={2026},
  eprint={2608.10475},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2608.10475},
}
```
