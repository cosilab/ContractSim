# ContractSim

ContractSim is the research code for studying LLM negotiation and contract
performance in stochastic catering environments. It includes negotiation and
execution experiments, rational baselines, an exact JAX contract solver,
synthetic-contract generation, Pareto search, and analysis scripts.

> **Work in progress:** the repository is undergoing a code cleanup. Interfaces,
> paths, and documentation may change while the research code is consolidated.

## Project map

- `negotiation/`: customer–caterer LLM negotiations.
- `performance/`: contract execution with LLM and rational participants.
- `algorithms/rc-solver/`: exact JAX solver and contingency implementations.
- `algorithms/pareto/`: F1 Pareto-frontier search and plotting (F2 support to be imported soon!).
- `algorithms/synth-contracts/`: synthetic-contract generation.
- `analysis/`: negotiation and performance analysis.
- `llm_utils/`: shared provider, agent, prompting, tracing, and process helpers.

Component-specific instructions are in the README within each directory.

TODO
- F2 Frontier support
- Domain robustness
- Add results and analysis

## Setup

Python 3.12 is currently used for development.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Provider-backed runs require the relevant API keys. Gemini credentials are also
required for the shared agent-memory embedder.

## Main entry points

```bash
.venv/bin/python negotiation/orchestration/run.py --help
.venv/bin/python performance/orchestration/run.py --help
.venv/bin/python algorithms/pareto/f1_frontier.py --help
```

The RC solver is run from `algorithms/rc-solver/`; see its README for details.

## Tests

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s llm_utils/tests -t .
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s negotiation -p 'test_*.py'
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s performance/tests -t .
```

Solver tests are run separately from `algorithms/rc-solver/`.

## Data

The `results/` and `analysis/` trees are excluded by the project `.gitignore`.

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
