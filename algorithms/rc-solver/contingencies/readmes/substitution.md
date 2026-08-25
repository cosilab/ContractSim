# Substitution

Implementation: [`../scripts/substitution.py`](../scripts/substitution.py)

## RC solver

- Weekly minimums must be produced as the named dishes.
- Production above those minimums creates the public substitution points supplied by `Environment`.
- Points may cover only above-minimum demand.
- Point allocation maximizes points served.
- Production selection prioritizes exact minimums, total point-weighted service, then fewer ingredients.
- Any incomplete result invokes zero-output grim in the Bellman solver.

Private customer values are used for welfare reporting, not production selection.

## Compared with `solver_rcc_bound`

The legacy `solver_rcc_bound.py` uses the same minimum-versus-substitutable split, but ranks service using private-value weights and legacy bound structures. The modular implementation uses public substitution points and values read from `Environment`.
