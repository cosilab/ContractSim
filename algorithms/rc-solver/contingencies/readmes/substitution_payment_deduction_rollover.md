# Substitution + payment deduction + rollover

Implementation: [`../scripts/substitution_payment_deduction_rollover.py`](../scripts/substitution_payment_deduction_rollover.py)

## RC solver

- In the substitution attempt, weekly minimums and incoming deficits must be produced as named dishes.
- Substitution credit applies only when it completely fulfills the obligation.
- Otherwise, the selector uses physical shortfall with the payment-deduction + rollover priorities: survival, full service, least deduction-price-weighted shortfall, and fewest ingredients.
- A bounded first residual both reduces the following payment and becomes the next exact deficit.
- A repeated, oversized, or final residual invokes zero-output grim.

## Compared with `solver_rcc_bound`

The legacy `solver_rcc_bound.py` treats substitution as a continuous overlay on partial plans. The modular version uses all-or-nothing substitution, is environment-driven, and inherits exact payment timing, deficit state, and zero-output termination from its base implementations.
