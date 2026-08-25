# Payment deduction

Implementation: [`../scripts/payment_deduction.py`](../scripts/payment_deduction.py)

## RC solver

Production is ranked by:

1. full service;
2. least deduction-price-weighted shortfall;
3. fewest ingredients.

Weekly minimums are not enforced by this standalone clause. A first shortfall is delivered and credited, and its value reduces the following payment to no less than zero. A consecutive shortfall invokes zero-output grim. A two-state previous-shortfall axis records that history.

The reduced payment is applied directly in the transition that receives it, so supplier utility and total payment use the actual amount paid.

## Compared with `solver_rcc_bound`

The legacy `solver_rcc_bound.py` uses similar production and hardship priorities, but models a deduction by shifting the continuation capital index and can still count the full scheduled payment. It can also retain partial production metrics on a terminating repeated shortfall. The modular solver uses exact payment timing and zero-output termination.
