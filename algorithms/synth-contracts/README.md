# Synthetic contracts

Performance-ready contracts live under `data/policy-suite/<contract-id>/` with
this layout:

```text
contract.txt
terms.json
config.json
baselines/rcc/policy.npz
```

## Folder naming

Contract directories use:

```text
env-<evaluation>_from-env-<source>_p<band>_<contingency-variant>
```

For example, `env-01_from-env-03_p080_substitution_rollover` contains terms
generated for environment 3, evaluated and solved in environment 1, from the
source satisfaction-probability band `p080`, with substitution and rollover.

- `env-XX` is the evaluation environment used to export the policy.
- `from-env-XX` is the environment used to generate the source contract terms.
- `p050`, `p060`, `p070`, `p080`, and `p090` denote source `P_sat` bands
  `[0.45, 0.55)`, `[0.55, 0.65)`, `[0.65, 0.75)`, `[0.75, 0.85)`, and
  `[0.85, 0.95]`, respectively.
- The suffix names the contingency variant. Grim trigger is the baseline and is
  included in every variant; other suffix components are `substitution`,
  `payment_deduction`, and `rollover`.

## Generation provenance

Generation was performed on an unmodified Lambda Stack 22.04 image on a
Lambda Cloud 1× NVIDIA GH200 instance.

The base contract-band generation runs completed on August 17, 2026. The
policy-suite expansion and policy-export run completed on August 19, 2026.

`config.json` must contain the contract's integer `environment_id` from 1 to 6.
Performance batches select only contracts matching `--env` or each ID in
`--envs`, and direct runs reject mismatches.

Contract-generation scripts are in `generation/`.

Generate `contract.txt` for an existing contract directory with:

```bash
python generation/generate_contract_txt.py data/policy-suite/<contract-id>
```

Performance batch runs default to `data/`; pass
`--contracts-root algorithms/synth-contracts/data/policy-suite` to run the
generated policy suite.
