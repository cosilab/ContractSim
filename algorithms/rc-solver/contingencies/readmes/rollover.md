# Rollover

Implementation: [`../scripts/rollover.py`](../scripts/rollover.py)

## RC solver

The current obligation is scheduled demand plus the previous per-dish deficit. Production is ranked by:

1. survival, with full service preferred;
2. least total unweighted dish shortfall;
3. fewest ingredients.

Weekly minimums are not enforced by this standalone clause. A first shortfall survives only when every dish deficit is within `max_deficit`; that deficit becomes next round's exact obligation. A repeated, oversized, or final-round shortfall invokes zero-output grim. The default cap is two units per dish.

## Compared with `solver_rcc_bound`

The legacy `solver_rcc_bound.py` uses the same carry-forward idea, but ranks partial service with customer-private-value weights and can record partial value on a terminating branch. The modular solver uses the base rollover rule's unweighted shortfall and consistent zero-output termination.
