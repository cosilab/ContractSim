# F1 Pareto frontier

This folder contains the standalone F1 workflow:

- `f1_frontier.py` is the public API and command-line entry point.
- `f1_common.py` contains shared models, configuration, and checkpoint I/O.
- `f1_base_search.py` contains the legacy epsilon-constraint search.
- `f1_refined_search.py` contains the anchor-local refined search.
- `plot_f1_frontier.py` combines sweep outputs, removes globally ineligible
  and dominated candidates, and exports the frontier.

Run both orientations from the RatConBench root:

```bash
python3 algorithms/pareto/f1_frontier.py \
  --env 5 \
  --sweep max-customer

python3 algorithms/pareto/f1_frontier.py \
  --env 5 \
  --sweep max-supplier
```

By default, each sweep writes its checkpoint, final JSON, and log under
`results/pareto-search/env-XX/`, using `<sweep>-f1` as the filename. A fresh run
replaces its log; `--resume` appends to it. Pass `--checkpoint`, `--output`, or
`--log` only to override those paths.

Base search defaults to `algorithms/pareto/examples/f1_initial_contract.json`, the same
`(2,2,2)` quantities, `(10,5,3)` prices, and six `$30` payments used by
ContractBench. Pass `--initial-contract` only to override that seed.

`--env ID` loads `envs/ID/environment.json` and checks that the file
declares the same ID. The current RatConBench manifest contains IDs 1–6.
Use the mutually exclusive `--environment path/to/environment.json` form for
an external or custom environment file.

RatConBench IDs 1–6 are the renumbered original environments 1, 4, 5, 8, 9,
and 12. To retain an original ContractBench ID when running an existing anchor,
point the numeric selector at that environment tree, for example with
`--env 8 --environment-root ../ContractBench/env-configs`.

Add `--resume` to the unchanged generation command to continue from its
checkpoint. The search validates the environment, initial contract, and all
search settings before resuming. It also fingerprints the F1 and solver source
files, so a checkpoint cannot silently combine evaluations from different
implementations. Checkpoints from older solver versions must be restarted.

## Base random starts

Base search reproduces ContractBench's fixed `(6, 8, 8)` quantity caps,
payment rounding, valid-draw budget, epsilon seed progression, and
satisfaction-based start ordering. It is defined for the canonical `A`, `B`,
`C` products with five production rounds and six payments.

`--random-start-attempts` can raise the structurally valid candidate evaluation
cap. The effective cap is at least 80 and at least 20 per requested random
start. Generation stops early after finding enough candidates, and rejected
raw draws have a separate safety limit.

With `--starts 3`, the third start is adaptive: it runs only when both of the
first two optimized starts miss the epsilon constraint.

## Targeted refinement

`F1FrontierSearchRefined` inherits `F1FrontierSearch` and ports the F1 portion
of ContractBench's `targeted_refine.py` algorithm to the modular solver. It keeps a
distinct quantity vector for every production round and searches within a
fixed box around the initial contract. The defaults are a quantity
radius of 1, price radius of 1, payment radius of 5, and 12 best-improvement
iterations. Per-dish expected-cost and private-value bounds are deliberately
not applied in this mode.

The three epsilon values are anchor-specific, so refined mode requires an
explicit grid:

```bash
python3 algorithms/pareto/f1_frontier.py \
  --env 5 \
  --initial-contract path/to/anchor-contract.json \
  --sweep max-customer \
  --epsilon-grid 35.414749,37.414749,39.414749 \
  --refined \
  --checkpoint algorithms/pareto/outputs/refined-max-customer.checkpoint.json \
  --output algorithms/pareto/outputs/refined-max-customer.json
```

Use `--sweep max-supplier` for the direction called `max-caterer` by the
ContractBench script. Radius and iteration overrides are available through
`--quantity-radius`, `--price-radius`, `--payment-radius`, and
`--refined-iterations`. Refined mode rejects the base-only random-start,
coarse/fine-iteration, pruning, and seed options.

Combine and plot the two sweeps:

```bash
python3 algorithms/pareto/plot_f1_frontier.py \
  --input results/pareto-search/env-05/max-customer-f1.json results/pareto-search/env-05/max-supplier-f1.json \
  --output-dir results/pareto-search/env-05
```

The plotter uses ContractBench's global F1 eligibility rule: finite utilities
and `P_sat + 1e-12 >= 0.95`. Whether a candidate met the epsilon constraint of
its originating subproblem remains metadata and does not exclude it from the
global frontier. The plotter writes JSON, CSV, PNG, and PDF. Pass `--no-figure`
to write only JSON and CSV when Matplotlib is unavailable.
