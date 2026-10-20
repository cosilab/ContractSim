import json
import random
from pathlib import Path

from customer import Customer
from supplier import Supplier


def negotiate(env_id, seed, batna=(None, None)):
    random.seed(seed)
    agents = [Customer(env_id), Supplier(env_id)]
    for a, b in zip(agents, batna):
        if b is not None:
            a.u_min, a.u_max = b, max(a.u_max, b)  # above the opening: demand b every round
    offer = agents[0].opening()
    last = [offer, None]
    trace = [offer]
    for t in range(1, agents[0].T + 1):  # customer (agents[0]) moves last, at T
        i = t % 2
        reply = agents[i].respond(offer, t)
        if reply == offer and i == 0:  # only the customer accepts; the supplier re-proposes the offer
            return agents[0], t, offer, trace
        if t == 1:
            reply = agents[i].opening()
        offer = last[i] = reply or last[i]
        trace.append(offer)
    return agents[0], None, None, trace


if __name__ == "__main__":
    for env_id in range(1, 7):
        for seed in (42, 43, 44):
            opener, t, deal, trace = negotiate(env_id, seed)
            out = Path(__file__).parent / "runs" / f"env{env_id}"
            out.mkdir(parents=True, exist_ok=True)
            (out / f"seed{seed}.json").write_text(json.dumps(
                {"env": env_id, "seed": seed, "opener": type(opener).__name__, "round": t, "deal": deal, "trace": trace}))
            print(env_id, seed, type(opener).__name__, t, deal)
