import json
import random
from pathlib import Path

ENVS = Path(__file__).resolve().parents[2] / "envs"
DISHES = ("A", "B", "C")


class Customer:
    def __init__(self, env_id, eps=10, q_max=5, step=4, pay_step=30):
        env = json.loads((ENVS / str(env_id) / "environment.json").read_text())
        game = env["base_game"]
        self.B = game["customer"]["starting_budget"]
        self.u_min = game["customer"]["outside_option_utility"]
        self.mins = tuple(game["customer"]["minimum_dishes_per_production_round"][d] for d in DISHES)
        self.v = tuple(env["customer_preference"]["private_values"][d] for d in DISHES)
        self.R = len(game["schedule"]["production_weeks"])
        self.n_pay = len(game["schedule"]["payment_weeks"])
        self.T = 2 * game["schedule"]["negotiation_max_rounds"]  # T counts turns; LLMs get 2 per round
        self.beta = 5
        self.eps = eps
        self.q_max = q_max
        self.step = step
        self.pay_step = pay_step
        self.u_max = self.utility(*self.opening())

    def target(self, t):
        return self.u_min + (self.u_max - self.u_min) * (1 - (t / self.T) ** (1 / self.beta))

    def utility(self, payments, q):
        return self.B - sum(payments) + self.R * sum(v * x for v, x in zip(self.v, q))

    def opening(self):
        return (0,) * self.n_pay, self.mins

    def respond(self, opp, t):
        """Return `opp` to accept, else a counter-offer (None if none found)."""
        pay, q = opp
        ok = sum(pay) <= self.B and all(x >= m for x, m in zip(q, self.mins))
        if ok and self.utility(pay, q) >= self.target(t):
            return opp
        return self.counter(opp, t)

    def counter(self, opp, t, n=10000):
        pay, q = opp
        lo, hi = self.target(t), self.target(t) + self.eps
        cands = []
        for _ in range(n):
            c = (tuple(max(0, p + random.randint(-self.pay_step, self.pay_step)) for p in pay),
                 tuple(random.randint(max(m, x - self.step), min(self.q_max, x + self.step)) for m, x in zip(self.mins, q)))
            if sum(c[0]) <= self.B and lo <= self.utility(*c) <= hi:
                cands.append(c)
        return random.choice(cands) if cands else None


def split(total, n):
    """Random split of `total` dollars into `n` weekly payments."""
    cuts = sorted(random.randint(0, total) for _ in range(n - 1))
    return tuple(b - a for a, b in zip((0, *cuts), (*cuts, total)))
