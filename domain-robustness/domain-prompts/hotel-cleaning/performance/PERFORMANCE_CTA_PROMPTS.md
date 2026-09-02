# Hotel-Cleaning Performance CTAs

This document records the model-facing prompts emitted by the fixed Env 9
hotel-cleaning performance runner. The game state remains in canonical Env 9
units internally, while every model-facing dollar amount is multiplied by 100
and formatted as hotel-domain USD.

The Hotel Owner acts once in each payment week (weeks 1, 3, 5, 7, 9, and 11).
The Cleaning Provider acts twice in each cleaning week (weeks 2, 4, 6, 8, and
10): first to buy cleaning supplies, then to perform cleaning services. A run
has only one LLM-controlled role; the counterparty is an `rc`, `re`, or `rcc`
rational baseline.

There is no communication, acceptance, termination, or renegotiation action
during performance.

## Monetary boundary

- Canonical Env 9 money is multiplied by 100 before it is shown to an LLM.
- A Hotel Owner payment must be a finite, non-negative whole number, a multiple
  of $100, and no greater than the displayed remaining budget.
- A valid displayed payment is divided by 100 before it enters the canonical
  game. Thus, a model response of `{"payment": 1200}` becomes a canonical
  payment of 12.
- Quantities, inventory, weeks, delivery outcomes, and expiration outcomes are
  not scaled.

## Phase-transition observation

At the start of a match, the LLM side observes:

> The negotiation phase is over. The performance phase now begins.
>
> The agreed contract is included in your private information.
>
> Each week you'll be told the game state and asked for a JSON decision.

The adapted hotel contract is already embedded in the LLM's fixed private
information. This transition does not request an action.

## Outcome observations

New observations are buffered until the LLM's next action and are also stored
in its associative memory. They are separate from the CTA itself.

### Hotel Owner

After a payment:

> Week {week}: You paid {displayed payment}. Budget remaining: {displayed remaining budget}.

After a cleaning week:

> Week {week} cleaning complete. Cleaning services performed this round: {named service quantities}. Cumulative cleaning services: {named cumulative service quantities}.

### Cleaning Provider

After a Hotel Owner payment:

> Week {week}: Hotel Owner paid {displayed payment}. Your balance is now {displayed balance}.

When at least one carried supply expires:

> Week {week} supply expiration: lost {positive named supply quantities}. Inventory is now {named supply inventory}.

When nothing expires:

> Week {week}: no supply expiration. Inventory: {named supply inventory}.

After an order and stochastic supply delivery:

> Week {week} supply delivery: ordered {named supply quantities}, received {named supply quantities}. Balance: {displayed balance}. Inventory: {named supply inventory}.

After performing cleaning services:

> Week {week} cleaning: performed {named service quantities}. Supply inventory after: {named supply inventory}. Cumulative cleaning services: {named cumulative service quantities}.

Supply names are `Detergent`, `Disinfectant`, and `Wet-wipes`. Service names
are `Conference Room Cleaning`, `King-Sized Room Cleaning`, and
`Single Room Cleaning`.

## 1. Hotel Owner payment CTA

Used in every payment week when the Hotel Owner is the LLM role:

> ── Week {week} of 11 (PAYMENT WEEK) ──
>
> Total paid so far: {displayed total paid}
>
> Remaining budget: {displayed remaining budget}
>
> In light of the above, make a payment to the Cleaning Provider. The payment must be a non-negative multiple of $100, including $0, and no greater than your remaining budget. Respond ONLY with JSON: `{"payment": <int>}`

If the amount is not numeric, finite, integral, non-negative, divisible by 100,
or within the remaining budget, the next action request is prefixed with:

> Your payment was REJECTED — payments must be non-negative multiples of $100, including $0, and no greater than your remaining budget of {displayed remaining budget}. Try again.

There are at most three amount-validation attempts. If all fail, the payment
defaults to $0.

## 2. Cleaning Provider supply-purchase CTA

Used after supply expiration in every cleaning week when the Cleaning Provider
is the LLM role:

> ── Week {week} of 11 (CLEANING WEEK — buying) ──
>
> Balance: {displayed provider balance}
>
> Inventory: {named supply inventory}
>
> Market prices: Detergent {displayed price}/unit, Disinfectant {displayed price}/unit, Wet-wipes {displayed price}/unit
>
> Buy cleaning supplies to perform cleaning services.
>
> Respond ONLY with JSON: `{"buy_Detergent": <int>, "buy_Disinfectant": <int>, "buy_Wet_Wipes": <int>}`

Values are converted to integers and clamped to the canonical order interval
from 0 through 12. The resulting order must be affordable using the Cleaning
Provider's canonical balance. If it is not affordable, the next request is
prefixed with:

> Your order was REJECTED — it costs {displayed order cost} but you only have {displayed provider balance}. Order fewer cleaning supplies.

There are at most three affordability attempts. If all fail, all purchase
quantities default to zero.

## 3. Cleaning Provider service CTA

Used after the supply order and stochastic delivery are resolved:

> ── Cleaning services (Week {week} of 11) ──
>
> Balance: {displayed provider balance}
>
> Inventory: {named supply inventory}
>
> Perform cleaning services using your current cleaning supplies.
>
> Respond ONLY with JSON: `{"perform_Conference_Room_Cleaning": <int>, "perform_King_Sized_Room_Cleaning": <int>, "perform_Single_Room_Cleaning": <int>}`

Values are converted to integers and clamped to a minimum of zero. The request
must be feasible under the fixed Env 9 recipes and current supply inventory. If
it is infeasible, the next request is prefixed with:

> Your cleaning-service decision was REJECTED — not enough cleaning supplies.
>
> {Supply}: need {required quantity} but only have {available quantity}
>
> Try again.

The shortage line is included once for each deficient supply. There are at
most three feasibility attempts. If all fail, all cleaning-service quantities
default to zero.

## Invalid-JSON retry

Each action request requires all of its listed JSON keys. When JSON parsing
fails or a key is missing, the retry CTA is:

> Your previous response wasn't valid JSON with keys {required key list}. Respond ONLY with the requested JSON. Original request: {original CTA}

JSON parsing is attempted up to three times per action request. If all three
attempts fail, the action-specific all-zero fallback is used. Subsequent
amount, affordability, or feasibility validation proceeds on the parsed or
fallback action as applicable.
