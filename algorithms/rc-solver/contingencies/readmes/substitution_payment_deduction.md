# Substitution + payment deduction

Implementation: [`../scripts/substitution_payment_deduction.py`](../scripts/substitution_payment_deduction.py)

## RC solver

- In the substitution attempt, minimums must be produced exactly; above-minimum production creates substitution points.
- Substitution credit applies only when it completely fulfills the obligation.
- Otherwise, payment deduction ranks plans by physical deduction-price-weighted shortfall and fewest ingredients, without partial substitution credit.
- A first residual shortfall reduces the following payment; a consecutive one invokes zero-output grim.
- Payment timing and hardship state are inherited from the base payment-deduction implementation.

## Compared with `solver_rcc_bound`

The legacy `solver_rcc_bound.py` credits partial substitution before pricing fallback shortfall. The modular solver instead uses all-or-nothing substitution, reads clause data from `Environment`, applies the reduced payment at the correct transition, and uses zero output on termination.
