"""BATNA sweep: raise one agent's walk-away value by a gain g, keep the other's true one.

g runs 0, STEP, ... up to that side's largest gain on the env's F1 frontier.
Usage: python sweep.py [step] [env ...] [--customer-below N] [--supplier-below M]   (defaults: 10, all envs, 0, 0)
--customer-below N also sweeps the customer's walk-away down to its true value - N.
--supplier-below M also sweeps the supplier's walk-away down to its true value - M, in steps of 1.
"""
import argparse
import json
from pathlib import Path

from customer import Customer
from orchestrate import negotiate
from supplier import Supplier

HERE = Path(__file__).parent
OUT = HERE / "sweep"
SEEDS = (42, 43, 44)
ap = argparse.ArgumentParser()
ap.add_argument("step", type=int, nargs="?", default=10)
ap.add_argument("envs", type=int, nargs="*")
ap.add_argument("--customer-below", type=int, default=0)
ap.add_argument("--supplier-below", type=int, default=0)
args = ap.parse_args()
STEP, ENV_IDS = args.step, args.envs or range(1, 7)


def f1_points(env_id):
    d = HERE.parent / "pareto" / "data" / f"env-0{env_id}"
    p = d / "f1_frontier.json" if (d / "f1_frontier.json").exists() else d / "f1-merged" / "f1_frontier.json"
    return [(x["E_customer_utility"], x["E_supplier_utility"]) for x in json.loads(p.read_text())["points"]]


for env_id in ENV_IDS:
    c, s = Customer(env_id), Supplier(env_id)
    pts = f1_points(env_id)
    grid = [("supplier", g) for g in range(0, int(max(u for _, u in pts) - s.u_min) + 1, STEP)]  # g=0: true point
    grid += [("customer", g) for g in range(STEP, int(max(u for u, _ in pts) - c.u_min) + 1, STEP)]
    grid += [("customer", -g) for g in range(STEP, args.customer_below + 1, STEP)]
    grid += [("supplier", -g) for g in range(1, args.supplier_below + 1)]
    out = OUT / f"env{env_id}"
    out.mkdir(parents=True, exist_ok=True)
    for side, g in grid:
        batna = (None, s.u_min + g) if side == "supplier" else (c.u_min + g, None)
        for seed in SEEDS:
            _, t, deal, trace = negotiate(env_id, seed, batna)
            (out / f"{side}_gain{g}_seed{seed}.json").write_text(json.dumps(
                {"env": env_id, "seed": seed, "side": side, "gain": g, "round": t, "deal": deal, "trace": trace}))
