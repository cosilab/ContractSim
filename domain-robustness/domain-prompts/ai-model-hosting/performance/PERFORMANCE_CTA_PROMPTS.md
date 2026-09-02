# AI-Model-Hosting Performance CTAs

This document records the model-facing prompts emitted by the fixed Env 9
AI-model-hosting performance runner. The game state remains in canonical Env 9
units internally, while every model-facing dollar amount is multiplied by
1,000 and formatted as AI-domain USD.

The AI Startup Founder acts once in each payment week (weeks 1, 3, 5, 7, 9,
and 11). The AI Service Provider acts twice in each service week (weeks 2, 4,
6, 8, and 10): first to reserve resources, then to provide services. A run has
only one LLM-controlled role; the counterparty is an `rc`, `re`, or `rcc`
rational baseline.

There is no communication, acceptance, termination, or renegotiation action
during performance.

## Monetary boundary

- Canonical Env 9 money is multiplied by 1,000 before it is shown to an LLM.
- An AI Startup Founder payment must be finite, non-negative, integral, a
  multiple of $1,000, and no greater than the displayed remaining budget.
- A valid displayed payment is divided by 1,000 before it enters the canonical
  game. Thus, `{"payment": 12000}` becomes a canonical payment of 12.
- Resource and service quantities, available capacity, weeks,
  capacity-allocation outcomes, and expiration outcomes are not scaled.

## Phase-transition observation

At the start of a match, the LLM side observes:

> The negotiation phase is over. The performance phase now begins.
>
> The agreed contract is included in your private information.
>
> Each week you'll be told the game state and asked for a JSON decision.

The adapted AI contract is already embedded in the LLM's fixed private
information. This transition does not request an action.

## Outcome observations

New observations are buffered until the LLM's next action and are also stored
in its associative memory. They are separate from the CTA itself.

### AI Startup Founder

After a payment:

> Week {week}: You paid {displayed payment}. Budget remaining: {displayed remaining budget}.

After a service week:

> Week {week} service round complete. Services provided this round: {named service quantities}. Cumulative services: {named cumulative service quantities}.

### AI Service Provider

After an AI Startup Founder payment:

> Week {week}: AI Startup Founder paid {displayed payment}. Your balance is now {displayed balance}.

When at least one carried resource expires:

> Week {week} resource expiration: lost {positive named resource quantities}. Available resources are now {named resource quantities}.

When nothing expires:

> Week {week}: no resource expiration. Available resources: {named resource quantities}.

After a reservation and stochastic capacity allocation:

> Week {week} capacity allocation: reserved {named resource quantities}, received {named resource quantities}. Balance: {displayed balance}. Available resources: {named resource quantities}.

After providing services:

> Week {week} service provision: provided {named service quantities}. Available resources after: {named resource quantities}. Cumulative services: {named cumulative service quantities}.

Resource names are `CPU`, `Memory`, and `GPU`. Service names are
`Fine-tuning`, `Model-hosting`, and `Inference`.

## 1. AI Startup Founder payment CTA

Used in every payment week when the AI Startup Founder is the LLM role:

> ── Week {week} of 11 (PAYMENT WEEK) ──
>
> Total paid so far: {displayed total paid}
>
> Remaining budget: {displayed remaining budget}
>
> In light of the above, make a payment to the AI Service Provider. The payment must be a non-negative multiple of $1,000, including $0, and no greater than your remaining budget. Respond ONLY with JSON: `{"payment": <int>}`

If the amount is not numeric, finite, integral, non-negative, divisible by
1,000, or within the remaining budget, the next request is prefixed with:

> Your payment was REJECTED — payments must be non-negative multiples of $1,000, including $0, and no greater than your remaining budget of {displayed remaining budget}. Try again.

There are at most three amount-validation attempts. If all fail, the payment
defaults to $0.

## 2. AI Service Provider resource-reservation CTA

Used after resource expiration in every service week when the AI Service
Provider is the LLM role:

> ── Week {week} of 11 (SERVICE WEEK — reserving) ──
>
> Balance: {displayed provider balance}
>
> Available resources: {named resource quantities}
>
> Market prices: CPU {displayed price}/unit, Memory {displayed price}/unit, GPU {displayed price}/unit
>
> Reserve resources to provide AI services.
>
> Respond ONLY with JSON: `{"reserve_CPU": <int>, "reserve_Memory": <int>, "reserve_GPU": <int>}`

Values are converted to integers and clamped to the fixed reservation interval
from 0 through 12. The resulting reservation must be affordable using the AI
Service Provider's current balance. If it is not affordable, the next request
is prefixed with:

> Your reservation was REJECTED — it costs {displayed reservation cost} but you only have {displayed provider balance}. Reserve fewer resources.

There are at most three affordability attempts. If all fail, all reservation
quantities default to zero.

## 3. AI Service Provider service-provision CTA

Used after the resource reservation and stochastic capacity allocation are
resolved:

> ── Service provision (Week {week} of 11) ──
>
> Balance: {displayed provider balance}
>
> Available resources: {named resource quantities}
>
> Provide AI services using your current resources.
>
> Respond ONLY with JSON: `{"provide_Fine_tuning": <int>, "provide_Model_hosting": <int>, "provide_Inference": <int>}`

Values are converted to integers and clamped to a minimum of zero. The request
must be feasible under the fixed Env 9 resource recipes and currently available
resources.
If it is infeasible, the next request is prefixed with:

> Your service-provision decision was REJECTED — not enough resources.
>
> {Resource}: need {required quantity} but only have {available quantity}
>
> Try again.

The shortage line is included once for each deficient resource. There are at
most three feasibility attempts. If all fail, all service quantities default
to zero.

## Invalid-JSON retry

Each action request requires all of its listed JSON keys. When JSON parsing
fails or a key is missing, the retry CTA is:

> Your previous response wasn't valid JSON with keys {required key list}. Respond ONLY with the requested JSON. Original request: {original CTA}

JSON parsing is attempted up to three times per action request. If all three
attempts fail, the action-specific all-zero fallback is used. Subsequent
amount, affordability, or feasibility validation proceeds on the parsed or
fallback action as applicable.
