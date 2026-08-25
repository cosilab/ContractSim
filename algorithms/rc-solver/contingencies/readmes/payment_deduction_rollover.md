# Payment deduction + rollover

Implementation: [`../scripts/payment_deduction_rollover.py`](../scripts/payment_deduction_rollover.py)

## RC solver

The current obligation is scheduled demand plus any incoming deficit. Production is ranked by:

1. avoiding immediate termination;
2. full service;
3. least deduction-price-weighted shortfall;
4. fewest ingredients.

A bounded first shortfall both reduces the following payment and becomes next round's per-dish deficit. Any shortfall with an incoming deficit, any oversized shortfall, or any final-round shortfall invokes zero-output grim. The deficit already records hardship, so no extra previous-shortfall axis is needed.

## Compared with `solver_rcc_bound`

The legacy `solver_rcc_bound.py` uses a similar clause state and survival priority. The modular version follows the base contingencies: it applies the actual reduced payment directly and clears all production/value metrics when termination is certain.
