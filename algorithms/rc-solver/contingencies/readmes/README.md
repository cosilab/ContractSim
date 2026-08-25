# Contingencies

Implementations are in [`../scripts/`](../scripts/). Each script has a short clause note:

| Clause | Note |
| --- | --- |
| Substitution | [substitution](substitution.md) |
| Payment deduction | [payment deduction](payment_deduction.md) |
| Rollover | [rollover](rollover.md) |
| Substitution + payment deduction | [substitution + payment deduction](substitution_payment_deduction.md) |
| Substitution + rollover | [substitution + rollover](substitution_rollover.md) |
| Payment deduction + rollover | [payment deduction + rollover](payment_deduction_rollover.md) |
| All three | [substitution + payment deduction + rollover](substitution_payment_deduction_rollover.md) |

## Shared approach

- Recipes, minimums, substitution points, private values, inventory limits, and probability kernels come from `Environment`; clause prices and caps are explicit inputs.
- Each module selects production locally, then supplies outcome tables to its Bellman solver.
- A branch that must terminate uses zero production and zero current customer value.
- The modules do not import or copy `solver_rcc_bound` tables or constants.

## Legacy comparison

The legacy `solver_rcc_bound.py` combines clauses inside one solver. It uses fixed A/B/C and X/Y/Z bound structures, applies substitution to partial plans, and has different payment-state and terminating-branch accounting. The modular code keeps each clause independent and environment-driven.

The three modular combinations use substitution as an all-or-nothing overlay: failed substitution contributes no credit to payment-deduction or rollover fallback shortfall.
