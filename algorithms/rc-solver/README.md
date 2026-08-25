# RC solver

This is the JAX solver for finite exact-delivery, grim-trigger resource
contracts. It enumerates delivery, spoilage, and observed price outcomes and
solves the contract by backward dynamic programming. It does not use Monte
Carlo sampling.

## Run it

Use the repository virtual environment and run from this directory:

```bash
../../.venv/bin/python rc_solver.py \
  --environment tests/data/environment_09.json \
  --contract tests/data/rollover_contract.json \
  --sat-thresh 0.95 \
  --save-policy policy.npz
```

The result contains satisfaction probability and expected utilities at zero
initial inventory and the environment's starting-capital state.
The optional `--save-policy` flag writes the selected order policy for every
round, inventory, budget, and observed price scenario.
The reported fields are `P_sat`, `E_supplier_utility`, and
`E_customer_utility`; their numeric values depend on the supplied environment
and contract.

For repeated contracts, reuse one environment-bound solver:

```python
from contract import Contract
from rc_solver import RCCSolver

solver = RCCSolver.from_json("tests/data/environment_09.json")
result = solver.solve(
    Contract.from_json("tests/data/rollover_contract.json"),
    satisfaction_threshold=0.95,
)
```

## Model

A state contains inventory for each resource and available supplier capital.
For each production round, the preceding payment is received, resource prices
are observed, and then an affordable order is selected. The resulting capital
indexes the continuation state for the next production round.

The terminal continuation has satisfaction probability one and includes the
contract's final payment. Production rounds are then solved backward, from the
final production round to the first.

For each current price scenario, the solver evaluates the expected continuation
of every candidate order over delivery, production, and spoilage outcomes, then
selects the best order. Only after that selection are the scenario-specific
results weighted by the declared price probabilities. This is a weighted sum,
not a renormalization, and means the policy depends on the observed prices.

For reproducibility, action selection matches the original ContractBench
`solver_rcc.py`. Each order receives the float32 score
`1e6 + supplier_utility` when its satisfaction probability is at least
`threshold - 1e-4`; otherwise its score is its satisfaction probability. The
order with the highest score is selected. This preserves the experiment code's
behavior, including float32 ties and extreme negative-utility edge cases.
Normally, this maximizes supplier utility among threshold-satisfying orders; if
none satisfies the threshold, it chooses the order with the highest attainable
satisfaction probability.

Resource transitions are contracted one resource at a time in environment
resource order, matching the original `X`, then `Y`, then `Z` arithmetic. The
final production round uses the same generic Bellman step as every other round.

Delivery, exact consumption, and spoilage are composed into a sub-stochastic
success transition. Missing probability mass is production failure; grim
trigger gives failed paths no continuation. The missing mass is not
renormalized away. The preceding payment is still received and the order cost
is still paid on failed paths.

The Bellman value tensor carries satisfaction probability, supplier utility,
customer value, and total paid. All four are contracted together, as in
`solver_rcc.py`; only satisfaction probability and supplier utility select the
action.

Current constraints:

- Prices and payments must be whole numbers because capital is an integer axis.
- Delivery, spoilage, and prices are independent across resources.
- Production requires the exact scheduled quantities.
- Contract product prices are validated but are not currently used by the
  Bellman calculation. Customer value uses the environment's private product
  values.
- The base `RCCSolver` uses exact-production grim. Substitution, payment
  deduction, rollover, and their combinations use separate contingency
  selectors and Bellman solvers.
- Renegotiation is not implemented.

## Code map

- `rc_solver.py`: reusable API and CLI
- `environment/`: validated environment input
- `contract/`: production and payment schedules
- `contingencies/scripts/`: clause selectors and Bellman solvers
- `contingencies/readmes/`: concise clause notes and legacy comparisons
- `transitions/`: delivery, resource, spoilage, and price kernels
- `runtime/`: transition reuse across contracts
- `bellman/`: JAX backward dynamic program
- `tests/`: centralized test suite and fixtures in `tests/data/`

## Tests

Run the complete centralized suite from `algorithms/rc-solver/`:

```bash
../../.venv/bin/python -m unittest discover -v -s tests
```

The runtime benchmark remains available at `benchmarks/benchmark_runtime.py`.
