# Concessionary Negotiation Baseline: Design

A heuristic negotiation baseline from the multi-issue bargaining literature. It
tells us what a simple negotiator achieves when it can evaluate contracts under
its own private information, but never sees the other side's. The agreed
contracts are scored with the exact solver and plotted with the LLM contracts
and the F1/F2 Pareto frontiers.

It complements the **Honest & Optimal** baseline, where both parties reveal all
private information. One party accepts any contract that gives it at least ε,
and the other maximizes its own value under that constraint. That lands on the
Pareto frontier, which is the existing ε-constraint frontier in
`algorithms/pareto/`.

The first experiment is deliberately bare-bones: each agent proposes a contract
that meets its own target for the turn and does not model the opponent (§4.2).

## 1. Contract space

The negotiation uses the F1 contract space (`ContractCandidate` in
`algorithms/pareto/generation/f1_common.py`), with prices removed:

| Issue | Dimensions | Notes |
|---|---|---|
| Payments | 6 (weeks 1, 3, 5, 7, 9, 11) | Integer dollars |
| Quantities per production round | 3 (A, B, C) | The same vector in all R = 5 production rounds |

**Why prices are removed:** the base solver (`rc_solver.py`) never reads
`product_prices`. Only the contingency scripts use prices, through deduction
prices. Without contingencies, prices have no effect on either party's utility,
so they are dropped. That leaves 9 dimensions.

**Scope:** contingencies are excluded, so the baseline negotiates over a fixed,
predefined set of issues. That is the standard multi-issue setting. LLM agents
can also write contingency clauses (F2), and the contrast between the two is
part of the point.

## 2. Protocol

Alternating offers with a deadline:

1. At round `t = 0`, the opener proposes its opening contract (§4.4).
2. At each later round, the responder either **accepts** the last offer or
   **rejects** it and makes a counter-offer.
3. The negotiation ends when an offer is accepted or when `t = T`.
   `T = negotiation_max_rounds` (50), the same limit the LLMs have.
4. **No deal:** both parties get their outside options.

Which side opens can bias the result, so the opener alternates across seeds.

## 3. Target utility (time-dependent tactic)

This follows Faratin, Sierra & Jennings (1998). At round `t`, agent `i` demands

```
u_i(t) = u_min + (u_max − u_min) · (1 − F(t)),    F(t) = k + (1 − k) · (t/T)^(1/β)
```

| Term | Meaning |
|---|---|
| `t`, `T` | Current round and deadline; `t/T` is the fraction of time used |
| `u_max` | Opening demand: the agent's utility for its opening contract (§4.4) |
| `u_min` | Reservation value: the agent never goes below it |
| `k` | Fraction conceded immediately. **Set k = 0**, so the agent opens at `u_max` |
| `β` | Concession shape: β < 1 is Boulware (holds firm, concedes late), β = 1 is linear, β > 1 is Conceder (concedes early) |

**Both agents use β = 3 (Conceder).** There is one shared value and no sweep.

- With a shared β, both agents have conceded the same fraction of their range
  at every round. Their pair of targets therefore moves along one straight line
  from the openings to the outside options, whatever β is. The split at which a
  deal first becomes possible is set by the openings; β only sets when.
- β = 3 concedes 27% of the range in round 1, 46% by round 5, 58% by round 10
  and 79% by round 25.
- β = 3 was chosen so the environments with small feasible regions have time to
  find a deal before the deadline (table below). With β = 1, env 5 has only 5
  rounds left once a deal first becomes possible; with β = 3 it has 15.
- The literature gives no recommended value. Genius, the ANAC reference
  platform, uses e = 0.2 (Boulware), 1 (linear) and 2 (Conceder); e is the same
  parameter as β. β = 3 is within the Conceder range.
- The cost: in envs 1, 2 and 4 a deal is possible by round 5–10 and targets keep
  falling quickly after that, so agreements will tend to land below the
  frontier, closer to the outside options.

**First round at which some F1 frontier contract meets both targets**, with
the openings of §4.4 and the reservation values of §5.4 and §6.4. The pair in
brackets is (`U_c`, `U_s`) at that round.

| Env | Individually rational frontier points | β = 1 | β = 3 |
|---|---|---|---|
| 1 | 122 / 122 | 28 (246, 110) | 9 (246, 109) |
| 2 | 150 / 150 | 23 (294, 135) | 5 (294, 134) |
| 3 | 19 / 19 | 36 (229, 70) | 19 (229, 69) |
| 4 | 63 / 63 | 29 (274, 105) | 10 (273, 104) |
| 5 | 12 / 14 | 45 (210, 25) | 35 (212, 28) |
| 6 | 45 / 46 | 37 (246, 65) | 20 (246, 66) |

This is when a mutually acceptable contract first exists. The agents' offers
are not optimized for the opponent, so actual agreements will come later and
below the frontier.

## 4. Offer and acceptance strategies

### 4.1 Acceptance (AC_next)

Agent `i` accepts the opponent's offer `c` if

- `c` satisfies `i`'s hard constraints, and
- `U_i(c) ≥ u_i(t+1)`, i.e. `c` is worth at least as much as `i`'s own next offer.

At the second-to-last round, `u_i(t+1) = u_min`, so an agent accepts any
feasible offer at least as good as walking away.

### 4.2 Counter-offers (bare-bones baseline)

Each agent counter-proposes a contract that meets its own target for the turn,
evaluated only with its own utility. Neither agent models the opponent or
learns from rejections. The target band is `[u_i(t), u_i(t) + ε]`.

- **Customer:** a uniformly random contract in its target band (§5.5).
- **Supplier:** a 50-evaluation local search, returning the feasible contract
  closest above its target (§6.5).

The risk is that offers that ignore the opponent may rarely be acceptable to it
before the deadline. The no-deal rate is itself a result.

### 4.3 Later extension: trade-off offers

Not part of the first experiment. This is the similarity-based trade-off tactic
of Faratin, Sierra & Jennings (2002): the agent amends the opponent's latest
offer as little as possible to meet its own target.

```
c* = argmin_c  dist(c, opp_last)
     s.t.      U_i(c) ∈ [u_i(t), u_i(t) + ε],  hard constraints of i
```

`dist` is the L1 distance with each dimension normalized by its feasible range.
With a linear utility, the customer's smallest-distance moves concentrate on
the dish with the most utility per normalized unit (usually A), so the
normalization ranges effectively become part of its strategy.

### 4.4 Opening offer

Each agent's first offer is a fixed, deliberately extreme contract, whether it
opens at `t = 0` or responds at `t = 1`. Its `u_max` is defined as its own
utility for that contract (§5.4, §6.4).

| Agent | Opening contract | Opener's utility |
|---|---|---|
| Customer | Pays $0; 1 unit of each dish per round (its minimum) | `U_c` = 305 (envs 1, 3, 5), 375 (envs 2, 4, 6) |
| Supplier | $250 in total, split evenly ($42 × 4, $41 × 2); no dishes | `U_s` = 250 (all envs) |

Neither opening is acceptable to the other side. The supplier's is over the
customer's budget and below its dish minimums. The customer's has P_sat = 0 in
envs 1, 3, 4, 5 and 6; in env 2 it has P_sat = 1 but costs the supplier its
whole capital (`U_s` = −40). With no dishes, the supplier's payment timing
does not affect its utility.

## 5. Customer agent

The customer never calls the solver. Its utility has a closed form, and it knows
nothing about the supplier's side.

### 5.1 Information

| Known (own private info) | Unknown |
|---|---|
| Budget `B` (`starting_budget`, 200 in all envs) | Ingredient prices, delivery loss, spoilage |
| Private dish values `v` (12/6/3 in envs 1, 3, 5; 20/10/5 in envs 2, 4, 6) | The supplier's starting capital |
| Minimum dishes per round (1 of each) | P_sat of any contract |
| Outside option (200) | The supplier's utility |

### 5.2 Naive utility

The customer assumes the supplier delivers every promised dish:

```
U_c(c) = B − Σ_w payments_w + R · Σ_i v_i · q_i        (R = 5)
```

This is the solver's `E_customer_utility` formula (`B + value − paid`) with
guaranteed delivery. It is linear: cutting $1 of payment is worth +1, and adding
one unit of dish `i` per round is worth `+R·v_i` (+60/+30/+15 in envs 1, 3, 5).
It does not depend on payment timing.

### 5.3 Hard constraints

- `Σ payments ≤ B`
- `q_i ≥ minimum_dishes_per_production_round[i]` for every dish
- Payments and quantities are nonnegative integers within the sampling bounds
  (§5.5)

Any contract that breaks these is never offered or accepted.

### 5.4 Concession endpoints

- `u_min` = outside option (200). A deal is never worth less than walking away.
- `u_max` = `U_c` of the customer's opening contract (§4.4): 305 in envs 1, 3,
  5 and 375 in envs 2, 4, 6. `U_c` has no upper bound over unbounded
  quantities, so `u_max` is fixed by the opening rather than by maximizing.

### 5.5 Counter-offer: random within the target band

Because `U_c` is linear, the customer constructs a contract in its band
directly, with no rejection sampling:

1. Draw quantities `q` uniformly within the sampling bounds, with every
   `q_i ≥ 1`.
2. Draw a target utility uniformly from `[u_c(t), u_c(t) + ε]` and set the
   total payment so that `U_c` equals it (rounded to whole dollars).
3. If the total is negative or above `B`, redraw.

**Payment timing.** `U_c` does not depend on timing, so the customer copies the
shape of the supplier's last payment schedule, scaled to its own total
(largest-remainder rounding so the weeks sum exactly to the total). It uses
only what is on the table, no supplier-side knowledge. Random or flat timing
would sink P_sat in the cap20 envs: the LLMs' flat starting contract (2/2/2,
$30 × 6) has P_sat = 0.36 in env 5, and every env 5 frontier contract puts
$85–195 in week 1. In effect, the supplier sets the schedule and the customer
negotiates the total and the quantities.

### 5.6 Known blind spot

The customer cannot see P_sat, so it will propose contracts the supplier cannot
reliably fulfil. This is by design, since it is what makes this the
no-private-information baseline. The supplier rejects those offers. The customer
does not learn from rejections.

## 6. Supplier agent

The supplier uses the same skeleton as the customer: the time-dependent target
(§3), AC_next acceptance (§4.1) and a fixed opening offer (§4.4). What differs
is how it evaluates contracts, what it knows, and how it finds counter-offers.

### 6.1 Information

The supplier gets exactly what the LLM caterer's prompt contains
(`negotiation/domain-prompts/catering/caterer/info.txt`).

| Known (own private info) | Unknown |
|---|---|
| Starting capital (20 in envs 1, 3, 5; 40 in envs 2, 4, 6) | Customer budget `B` |
| Operating conditions: ingredient prices, delivery loss, spoilage, inventory cap | Customer private dish values |
| Recipes | Customer minimum dishes per round |
| Walk-away value (keep its starting capital) | Customer utility |

### 6.2 Utility

`U_s(c)` is the solver's `E_supplier_utility` for contract `c`, assuming the
customer pays the full schedule. It is profit: payments received minus
ingredient costs, excluding starting capital.

It has no closed form. For each contract the solver runs backward induction
over the production rounds to find the supplier's best ingredient-ordering
policy under random prices, delivery loss (it receives half to all of what it
orders, and pays for all of it), spoilage, the inventory cap, cash flow (it can
only buy with capital plus payments received so far) and the grim trigger (a
missed delivery ends the deal). Early payments fund ingredient orders, so `U_s`
and P_sat depend on payment timing, especially at cap20.

A naive closed-form supplier utility (average prices, full delivery, no
spoilage) was considered and rejected: the supplier uses the exact solver.

### 6.3 Hard constraints

- `P_sat(c) ≥ 0.95`, the same threshold the frontier search uses
  (`F1SearchConfig.minimum_satisfaction`).
- Payments and quantities are nonnegative integers within the supplier's search
  bounds.

**Payment cap.** The F1 frontier search caps total payment at 200, which equals
`B`. Giving the supplier that cap would leak the customer's budget. The
supplier's search space therefore uses a looser cap (300, above its $250
opening), and the customer rejects over-budget offers. Estimating `B` from the
customer's offers is not used: those offers only give a lower bound on `B`.

**Customer minimums** are not a constraint for the supplier, because the LLM
caterer is not told them. The supplier may propose contracts with zero units of
a dish, and the customer rejects them (§5.3).

### 6.4 Concession endpoints

- `u_min` = 0. `E_supplier_utility` excludes starting capital, so walking away
  with the starting capital, as the LLM prompt describes, is worth 0 in these
  units.
- `u_max` = `U_s` of the supplier's opening contract (§4.4): 250 in all envs.

### 6.5 Counter-offer: local search around a candidate

Each turn the supplier runs a local search of at most **50 solver
evaluations** around a starting contract and offers the best one found:

- **Start:** the customer's latest offer, so the supplier edits the contract in
  front of it.
- **Feasible:** `P_sat ≥ 0.95` and `U_s ≥ u_s(t)`.
- **Best:** among feasible contracts, the one with the smallest `U_s − u_s(t)`,
  i.e. the least over-demand. "Best" must not mean the highest `U_s`: the
  search would then climb to the most profitable nearby contract every turn and
  the supplier would never concede.
- **Moves:** ±1–3 dishes of one product, ±dollars on one payment week, and
  shifting dollars between weeks with the total unchanged. The step sizes are
  the F1 search's (`FINE_QUANTITY_DELTAS`, `FINE_PAYMENT_SWAPS` in
  `f1_common.py`).
- **Search order:** best-first, expanding the evaluated contract whose `U_s` is
  closest to the target band, evaluating 8 neighbours per step (one per GPU,
  §6.8). Ties are broken randomly with the run's seed.
- **Fallback:** if no evaluation this turn is feasible, offer the cached
  feasible contract (§6.6) closest above the target. If there is none, repeat
  the supplier's own last offer, the standard "no concession this round" move.

Incoming customer offers need one solver call each for the acceptance check
(§4.1).

### 6.6 Evaluation cache

One `(U_s, P_sat)` cache per environment, keyed by contract, persisted to disk
and shared across turns, runs and seeds. The supplier's target only falls, so a
contract that was above the band in an earlier turn can fall into it later at
no extra solver cost.

### 6.7 Tuning discipline

The search budget (50), move sizes and ε are search hyperparameters, not the
negotiation strategy, but they can still shift outcomes. Tune them once, on one
environment and for search reliability (how often a turn finds a feasible
contract within budget), not for the utility the supplier achieves. Then freeze
them for all environments and report them.

### 6.8 Solver and cost

The baseline uses this repo's solver (`algorithms/rc-solver/`). The F1 frontier
and the LLM contract scores come from it, so the negotiating agents, the final
scoring and the frontier all use the same numbers.

| Env | Solver time per contract (A100, after warm-up) |
|---|---|
| 1 | 0.25 s |
| 5 | 1.50 s |
| 6 | 1.65 s |

Almost all of that is the Bellman backward pass on the GPU; transition kernels
are cached (~1 ms). Output is bit-identical across runs and between A100 and
A40 GPUs, so contracts can be solved in parallel on all 8 GPUs with no change
in results. At 50 evaluations per turn, env 5 takes about 75 s per supplier
turn on one GPU and about 10 s on eight, or about 5 minutes per negotiation.

The solver in the ContractSim-Negotiation repo is 2–4.5× faster, but only
matches to float32 noise: usually within 0.1, occasionally about 2.5 on one
contract when rounding flips a near-tie between ingredient orders. It is not
used, to keep scores consistent with the frontier and the LLM results.

## 7. Outputs

- For every run, record the env, seed, opener, the offer trace, the agreed
  contract (or no deal), and the round of agreement.
- Score every agreed contract with the exact solver (`E_customer_utility`,
  `E_supplier_utility`, `P_sat`), the same way LLM contracts are scored in
  `analysis/`. Also record the customer's naive `U_c`: the gap between what it
  thought it agreed to and its exact score is a result in itself.
- Plot the agreed contracts on the existing frontier figure, with the LLM
  points.

## 8. Open decisions

- ε for both agents.
- Quantity bounds for the customer's sampling (§5.5) and the supplier's search.
- Number of seeds per environment.
- Whether to port the parts of the faster solver that only index values
  (computing customer value and total paid at the chosen order only; padded
  capital slicing). These should be bit-exact but have not been verified.

## References

- Faratin, Sierra & Jennings (1998). Negotiation decision functions for
  autonomous agents. *Robotics and Autonomous Systems*.
- Faratin, Sierra & Jennings (2002). Using similarity criteria to make issue
  trade-offs in automated negotiations. *Artificial Intelligence*.
- Genius negotiation platform, `TimeDependentAgent` (Boulware e = 0.2,
  Conceder e = 2).
