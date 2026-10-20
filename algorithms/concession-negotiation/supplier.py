import json
import math
import random
from itertools import product
from pathlib import Path

from customer import DISHES, split

ENVS = Path(__file__).resolve().parents[2] / "envs"


class Supplier:
    def __init__(self, env_id, open_total=250, cap=300, eps=10, min_sat=0.7, q_max=5, step=4, pay_step=30):
        env = json.loads((ENVS / str(env_id) / "environment.json").read_text())
        game = env["base_game"]
        self.prices = env["operating_conditions"]["prices"]["ingredients"]
        self.mean_price = {k: sum(o["value"] * o["probability"] for o in v["outcomes"]) for k, v in self.prices.items()}
        self.recipes = game["recipes"]
        self.inv_cap = game["inventory"]["max_units_per_ingredient"]
        self.max_order = game["inventory"]["max_order_per_ingredient_per_round"]
        lo = lambda x: eval(env["operating_conditions"]["delivery"]["minimum_received"], {"floor": math.floor, "ordered": x})
        # exp_order[n]: smallest order whose expected delivery (uniform on [lo(x), x]) covers a need of n
        self.exp_order = [next((x for x in range(n, self.max_order + 1) if (lo(x) + x) / 2 >= n), self.max_order)
                          for n in range(200)]
        self.capital = env["caterer_capital"]["starting_balance"]
        sched = game["schedule"]
        self.R = len(sched["production_weeks"])
        self.n_pay = len(sched["payment_weeks"])
        self.T = 2 * sched["negotiation_max_rounds"]
        self.open_total = open_total
        self.cap = cap
        self.eps = eps
        self.min_sat = min_sat
        self.q_max = q_max
        self.step = step
        self.pay_step = pay_step
        self.beta = 5
        self.u_min = 0
        self.u_max = self.utility(*self.opening())

    def target(self, t):
        return self.u_min + (self.u_max - self.u_min) * (1 - (t / self.T) ** (1 / self.beta))

    def need(self, q):
        return {k: sum(self.recipes[d].get(k, 0) * x for d, x in zip(DISHES, q)) for k in self.prices}

    def order(self, need):
        """Order enough that the expected delivery covers the need."""
        return {k: self.exp_order[n] for k, n in need.items()}

    def p_sat(self, payments, q):
        """Each round must afford its order (no carryover). Budget =
        capital + payments so far - expected spend of earlier rounds (as in `utility`)."""
        order = self.order(self.need(q))
        cost = {}
        for combo in product(*(self.prices[k]["outcomes"] for k in order)):
            c = sum(order[k] * o["value"] for k, o in zip(order, combo))
            cost[c] = cost.get(c, 0) + math.prod(o["probability"] for o in combo)
        spent = (sum(payments) - self.utility(payments, q)) / self.R
        budget, p = self.capital, 1.0
        for pay in payments[:self.R]:  # each payment week precedes a production week
            budget += pay
            p *= sum(pr for c, pr in cost.items() if c <= budget)
            budget -= spent
        return p

    def utility(self, payments, q):
        """Total payment - expected ingredient cost of each round's order at mean prices."""
        cost = sum(x * self.mean_price[k] for k, x in self.order(self.need(q)).items())
        return sum(payments) - self.R * cost

    def opening(self):
        return split(self.open_total, self.n_pay), (0, 0, 0)

    def respond(self, opp, t):
        """Return `opp` to accept, else a counter-offer (None if none found)."""
        if self.utility(*opp) >= self.target(t) and self.p_sat(*opp) >= self.min_sat:
            return opp
        return self.counter(opp, t)

    def counter(self, opp, t, n=10000):
        pay, q = opp
        lo, hi = self.target(t), self.target(t) + self.eps
        cands = []
        for _ in range(n):
            c = (tuple(max(0, p + random.randint(-self.pay_step, self.pay_step)) for p in pay),
                 tuple(random.randint(max(0, x - self.step), min(self.q_max, x + self.step)) for x in q))
            if sum(c[0]) <= self.cap and lo <= self.utility(*c) <= hi and self.p_sat(*c) >= self.min_sat:
                cands.append(c)
        return random.choice(cands) if cands else None
