# Substitution + rollover

Implementation: [`../scripts/substitution_rollover.py`](../scripts/substitution_rollover.py)

## RC solver

- In the substitution attempt, current weekly minimums and incoming deficits must be produced as named dishes.
- Substitution points may cover only current demand above the ordinary minimums.
- Substitution credit applies only when it completely fulfills the obligation.
- Otherwise, rollover ranks plans by physical unweighted dish shortfall and fewest ingredients, without partial substitution credit.
- A bounded first residual becomes next round's exact deficit; a repeated, oversized, or final residual invokes zero-output grim.
- Deficit state and terminal behavior are inherited from base rollover.

## Compared with `solver_rcc_bound`

The legacy `solver_rcc_bound.py` applies partial substitution before rollover, weights partial service by private values, and can retain metrics on terminating plans. The modular version uses all-or-nothing substitution over base rollover's unweighted, zero-output semantics.
