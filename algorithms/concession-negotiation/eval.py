"""Score every run in a folder with the exact solver: python eval.py [runs|sweep]"""
import csv
import json
import sys
from functools import cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rc-solver"))

from contract import Contract  # noqa: E402
from rc_solver import RCCSolver  # noqa: E402

from customer import DISHES, ENVS, Customer  # noqa: E402
from supplier import Supplier  # noqa: E402

KEYS = ("P_sat", "E_customer_utility", "E_supplier_utility", "naive_U_c", "naive_U_s")


@cache
def solver(env_id):
    return RCCSolver.from_json(ENVS / str(env_id) / "environment.json")


@cache
def score(env_id, pay, q):
    sched = json.loads((ENVS / str(env_id) / "environment.json").read_text())["base_game"]["schedule"]
    contract = Contract.from_mapping({
        "product_prices": {d: 0 for d in DISHES},  # unused by the base solver
        "production_schedule": [{"week": w, "product_quantities": dict(zip(DISHES, q))} for w in sched["production_weeks"]],
        "payment_schedule": [{"week": w, "amount": a} for w, a in zip(sched["payment_weeks"], pay)],
    })
    return solver(env_id).solve(contract) | {
        "naive_U_c": Customer(env_id).utility(pay, q), "naive_U_s": Supplier(env_id).utility(pay, q)}


root = HERE / (sys.argv[1] if len(sys.argv) > 1 else "runs")
envs = sys.argv[2:]  # optional env ids, to shard across GPUs; CSV is written only without them
rows = []
for path in sorted(p for e in envs or ["*"] for p in root.glob(f"env{e}/*.json")):
    run = json.loads(path.read_text())
    if run["deal"] is not None and "eval" not in run:
        run["eval"] = score(run["env"], *map(tuple, run["deal"]))
        path.write_text(json.dumps(run))
    ev = run.get("eval", {})
    rows.append({k: v for k, v in run.items() if k not in ("deal", "trace", "eval")}
                | {"deal": run["deal"]} | {k: ev.get(k) for k in KEYS})
    print(path.relative_to(HERE), *(rows[-1][k] for k in KEYS))

if not envs:
    with open(root / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
