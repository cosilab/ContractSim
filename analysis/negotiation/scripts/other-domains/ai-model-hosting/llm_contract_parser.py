"""LLM-backed extraction of structured AI model-hosting contract terms.

This is the current-analysis counterpart of
``research-Qs/llm-prereqs/contract_terms.py``.  Raw formal offers are still
identified deterministically from the negotiation trace; this module only
turns each offer's natural-language contract into the structured terms consumed
by validation and the RCC solver.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

_SCRIPT_ROOT = Path(__file__).resolve().parents[2]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from contract_parser import (
    DISH_PATTERNS,
    PAYMENT_WEEKS,
    PRODUCTION_WEEKS,
    ParsedContract,
    parse_contract as parse_contract_deterministically,
)


DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_BATCH_SIZE = 30
MAX_JSON_ATTEMPTS = 3
DEFAULT_ROLLOVER_MAX_DEFICIT = 2
MONEY_SCALE = 1_000
KEY_FILES = {
    "anthropic": "ANTHROPIC_API.key",
    "openai": "OPENAI_API.key",
    "gemini": "GEMINI_API.key",
}
DEFAULT_MODELS = {
    "anthropic": "claude-opus-5",
    "openai": "gpt-5.6-sol",
    "gemini": DEFAULT_MODEL,
}
JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

CONTINGENCY_GRIM = "grim_trigger"
CONTINGENCY_PAYDED = "payment_deduction"
CONTINGENCY_ROLLOVER = "rollover"
CONTINGENCY_PAYDED_ROLLOVER = "payment_deduction_rollover"
CONTINGENCY_SUBSTITUTION = "substitution"
CONTINGENCY_ORDER = (
    CONTINGENCY_SUBSTITUTION,
    CONTINGENCY_PAYDED,
    CONTINGENCY_ROLLOVER,
    CONTINGENCY_GRIM,
)

# The RCC solver represents substitution as an overlay on four primary modes.
# Their 4 x 2 cross-product is the eight effective contingency combinations
# used by the benchmark (grim is present in every effective combination).
COMBINATION_GRIM = "grim_trigger"
COMBINATION_SUBSTITUTION_GRIM = "substitution_grim_trigger"
COMBINATION_PAYDED_GRIM = "payment_deduction_grim_trigger"
COMBINATION_SUBSTITUTION_PAYDED_GRIM = (
    "substitution_payment_deduction_grim_trigger"
)
COMBINATION_ROLLOVER_GRIM = "rollover_grim_trigger"
COMBINATION_SUBSTITUTION_ROLLOVER_GRIM = "substitution_rollover_grim_trigger"
COMBINATION_PAYDED_ROLLOVER_GRIM = "payment_deduction_rollover_grim_trigger"
COMBINATION_SUBSTITUTION_PAYDED_ROLLOVER_GRIM = (
    "substitution_payment_deduction_rollover_grim_trigger"
)
CONTINGENCY_COMBINATIONS = (
    COMBINATION_GRIM,
    COMBINATION_SUBSTITUTION_GRIM,
    COMBINATION_PAYDED_GRIM,
    COMBINATION_SUBSTITUTION_PAYDED_GRIM,
    COMBINATION_ROLLOVER_GRIM,
    COMBINATION_SUBSTITUTION_ROLLOVER_GRIM,
    COMBINATION_PAYDED_ROLLOVER_GRIM,
    COMBINATION_SUBSTITUTION_PAYDED_ROLLOVER_GRIM,
)


PROMPT = """Extract structured numerical contract terms from these service contracts.

Return ONLY valid JSON in this exact shape:
{{
  "contracts": [
    {{
      "contract_id": "<same id from input>",
      "service_prices": {{"A": <number|null>, "B": <number|null>, "C": <number|null>}},
      "production_schedule": [
        {{"week": 2, "A": <int|null>, "B": <int|null>, "C": <int|null>}},
        {{"week": 4, "A": <int|null>, "B": <int|null>, "C": <int|null>}},
        {{"week": 6, "A": <int|null>, "B": <int|null>, "C": <int|null>}},
        {{"week": 8, "A": <int|null>, "B": <int|null>, "C": <int|null>}},
        {{"week": 10, "A": <int|null>, "B": <int|null>, "C": <int|null>}}
      ],
      "payment_schedule": [
        {{"week": 1, "amount": <number|null>}},
        {{"week": 3, "amount": <number|null>}},
        {{"week": 5, "amount": <number|null>}},
        {{"week": 7, "amount": <number|null>}},
        {{"week": 9, "amount": <number|null>}},
        {{"week": 11, "amount": <number|null>}}
      ],
      "contingency_set": ["<implemented_clause_name>"],
      "contingency_params": {{
        "substitution": {{"min_qty": {{"A": int, "B": int, "C": int}}}},
        "payment_deduction": {{"min_qty": {{"A": int, "B": int, "C": int}}}},
        "rollover": {{"max_deficit": int|null}}
      }}
    }}
  ]
}}

Mapping:
- A = Fine-tuning 
- B = Model-hosting
- C = Inference

Rules:
- Parse each input contract_id exactly once and copy its identifier exactly.
- `production_schedule` must contain exactly one row for each of weeks 2, 4, 6,
  8, and 10. Each row has `week`, `A`, `B`, and `C`; quantities are
  non-negative integers or null.
- `payment_schedule` must contain exactly one row for each of weeks 1, 3, 5, 7,
  9, and 11. Each row has `week` and `amount`; amounts are non-negative numbers
  or null.
- If the same quantities apply in weeks 2, 4, 6, 8, and 10, expand them.
- If a fixed payment applies to multiple payment weeks, expand it.
- Be especially careful that Week 11 is payment week 11, not week 1.
- Do not invent missing values. Use null when absent or ambiguous.
- Amounts and prices must be plain numbers, with no dollar signs.
- Only return the fields shown above.
- Extract only contingencies explicitly expressed by the contract. Do not add a
  default or implicit fallback contingency.
- Include every implemented contingency explicitly present in the contract.
  Every array entry must be exactly one of: "substitution", "grim_trigger",
  "payment_deduction", or "rollover". Do not return any other clause name.
- If the contract contains no explicit contingency, return `contingency_set: []`
  and `contingency_params: {{}}`.
- Use "grim_trigger" only when the contract explicitly states termination,
  suspension of future performance, or an equivalent trigger after breach.
- Use "substitution" when the contract explicitly allows some promised services
  above the required minimum quantities to be replaced with other services. Use
  expressly stated minimum quantities; if none are stated, use
  {{"A": 1, "B": 1, "C": 1}}.
- Use "payment_deduction" when a provision shortfall results in a deduction from
  a future scheduled payment.
- Use "rollover" when missed services carry forward or must be cured later. If
  the contract states a maximum deficit, extract that number; otherwise use
  null.
- Only include parameter blocks for active contingencies. The permitted blocks
  are `substitution: {{"min_qty": {{"A": int, "B": int, "C": int}}}}`,
  `payment_deduction: {{"min_qty": {{"A": int, "B": int, "C": int}}}}`, and
  `rollover: {{"max_deficit": int|null}}`.

INPUT CONTRACTS JSON:
{contracts_json}
"""


@dataclass(frozen=True)
class LLMBatchResult:
    terms_by_id: dict[str, dict[str, Any]]
    raw_response: str


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _load_api_key(provider: str) -> str:
    env_name = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
    }[provider]
    environment_value = os.environ.get(env_name)
    if environment_value:
        return environment_value.strip()
    keys_dir = Path(
        os.environ.get("KEYS_DIR", str(_repo_root() / "keys" / "cosi"))
    )
    return (keys_dir / KEY_FILES[provider]).read_text().strip()


def _call_llm(
    provider: str,
    model: str,
    prompt: str,
) -> str:
    if provider == "gemini":
        import google.genai as genai

        client = genai.Client(api_key=_load_api_key(provider))
        response = client.models.generate_content(
            model=model,
            contents=prompt,
        )
        return str(response.text or "").strip()
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=_load_api_key(provider))
        response = client.messages.create(
            model=model,
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )
        return "\n".join(
            getattr(block, "text", "")
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
    if provider == "openai":
        import openai

        client = openai.OpenAI(api_key=_load_api_key(provider))
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": prompt}],
        )
        return response.output_text.strip()
    raise ValueError(f"Unsupported provider: {provider}")


def _json_from_text(raw: str) -> dict[str, Any]:
    cleaned = JSON_FENCE_RE.sub("", raw).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match is None:
            raise
        return json.loads(match.group(0))


def _response_payload_from_text(raw: str) -> dict[str, Any]:
    payload = _json_from_text(raw)
    if not isinstance(payload, dict):
        raise ValueError("LLM response must be a JSON object")
    if not isinstance(payload.get("contracts"), list):
        raise ValueError("LLM response must contain a contracts array")
    return payload


def numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except ValueError:
        return None


def _clean_number(value: float | None) -> int | float | None:
    if value is None:
        return None
    return int(value) if abs(value - round(value)) < 1e-9 else value


def _canonical_money(value: Any) -> int | float | None:
    selected = numeric(value)
    if selected is None:
        return None
    return _clean_number(selected / MONEY_SCALE)


def _canonical_decimal(value: Decimal) -> Decimal:
    return value / Decimal(MONEY_SCALE)


def _int_map(
    raw: Any,
    *,
    fallback: dict[str, Any] | None = None,
) -> dict[str, int | None]:
    source = raw if isinstance(raw, dict) else {}
    base = fallback or {}
    output: dict[str, int | None] = {}
    for dish in DISH_PATTERNS:
        selected = numeric(source.get(dish))
        if selected is None:
            selected = numeric(base.get(dish))
        output[dish] = int(round(selected)) if selected is not None else None
    return output


def normalize_contingencies(raw_terms: dict[str, Any]) -> list[str]:
    candidates = raw_terms.get("contingency_set")
    if candidates is None:
        # Compatibility with any pre-v11 term files supplied manually.
        candidates = raw_terms.get("contingencies")
    candidates = candidates or []
    items = [str(item).strip().lower() for item in candidates]
    params = raw_terms.get("contingency_params")
    if isinstance(params, dict):
        items.extend(str(key).strip().lower() for key in params)
    active = {item.replace("-", "_") for item in items}
    return [name for name in CONTINGENCY_ORDER if name in active]


def resolve_contingency_mode(contingencies: list[str]) -> str | None:
    active = set(contingencies)
    if CONTINGENCY_PAYDED in active and CONTINGENCY_ROLLOVER in active:
        return CONTINGENCY_PAYDED_ROLLOVER
    if CONTINGENCY_PAYDED in active:
        return CONTINGENCY_PAYDED
    if CONTINGENCY_ROLLOVER in active:
        return CONTINGENCY_ROLLOVER
    if CONTINGENCY_GRIM in active:
        return CONTINGENCY_GRIM
    return None


def resolve_effective_contingency_combination(contingencies: list[str]) -> str:
    """Map detected mechanisms to one of the benchmark's eight RCC cases."""
    active = set(contingencies)
    substitution = CONTINGENCY_SUBSTITUTION in active
    payment_deduction = (
        CONTINGENCY_PAYDED in active
        or CONTINGENCY_PAYDED_ROLLOVER in active
    )
    rollover = (
        CONTINGENCY_ROLLOVER in active
        or CONTINGENCY_PAYDED_ROLLOVER in active
    )
    mapping = {
        (False, False, False): COMBINATION_GRIM,
        (True, False, False): COMBINATION_SUBSTITUTION_GRIM,
        (False, True, False): COMBINATION_PAYDED_GRIM,
        (True, True, False): COMBINATION_SUBSTITUTION_PAYDED_GRIM,
        (False, False, True): COMBINATION_ROLLOVER_GRIM,
        (True, False, True): COMBINATION_SUBSTITUTION_ROLLOVER_GRIM,
        (False, True, True): COMBINATION_PAYDED_ROLLOVER_GRIM,
        (True, True, True): COMBINATION_SUBSTITUTION_PAYDED_ROLLOVER_GRIM,
    }
    return mapping[(substitution, payment_deduction, rollover)]


def normalize_terms(raw_terms: dict[str, Any]) -> dict[str, Any]:
    raw_prices = raw_terms.get("service_prices")
    raw_prices = raw_prices if isinstance(raw_prices, dict) else {}
    dish_prices = {
        dish: _canonical_money(raw_prices.get(dish)) for dish in DISH_PATTERNS
    }

    production_by_week: dict[int, dict[str, int | None]] = {}
    for row in raw_terms.get("production_schedule") or []:
        if not isinstance(row, dict) or numeric(row.get("week")) is None:
            continue
        production_by_week[int(numeric(row["week"]) or 0)] = _int_map(row)
    production_schedule = [
        {
            "week": week,
            **production_by_week.get(week, {dish: None for dish in DISH_PATTERNS}),
        }
        for week in PRODUCTION_WEEKS
    ]

    payment_by_week: dict[int, int | float | None] = {}
    for row in raw_terms.get("payment_schedule") or []:
        if not isinstance(row, dict) or numeric(row.get("week")) is None:
            continue
        payment_by_week[int(numeric(row["week"]) or 0)] = _canonical_money(
            row.get("amount")
        )
    payment_schedule = [
        {"week": week, "amount": payment_by_week.get(week)}
        for week in PAYMENT_WEEKS
    ]

    contingencies = normalize_contingencies(raw_terms)
    raw_params = raw_terms.get("contingency_params")
    raw_params = raw_params if isinstance(raw_params, dict) else {}
    contingency_params: dict[str, Any] = {}
    # The current 12-environment benchmark uses one of every dish per
    # production round.  The archived research-Qs parser used B=2 for its
    # earlier environment set, so that old constant must not be copied here.
    default_minimums = {"A": 1, "B": 1, "C": 1}
    if CONTINGENCY_SUBSTITUTION in contingencies:
        block = raw_params.get(CONTINGENCY_SUBSTITUTION)
        block = block if isinstance(block, dict) else {}
        contingency_params[CONTINGENCY_SUBSTITUTION] = {
            "min_qty": _int_map(block.get("min_qty"), fallback=default_minimums)
        }
    if CONTINGENCY_PAYDED in contingencies:
        block = raw_params.get(CONTINGENCY_PAYDED)
        block = block if isinstance(block, dict) else {}
        contingency_params[CONTINGENCY_PAYDED] = {
            "min_qty": _int_map(block.get("min_qty"), fallback=default_minimums),
            # RCC deductions are valued at the agreed per-dish prices. This is
            # derived locally rather than treated as an independent LLM field.
            "deduction_prices": _int_map(dish_prices),
            "deduction_prices_source": "agreed_dish_prices",
        }
    if CONTINGENCY_ROLLOVER in contingencies:
        block = raw_params.get(CONTINGENCY_ROLLOVER)
        block = block if isinstance(block, dict) else {}
        extracted_deficit = numeric(block.get("max_deficit"))
        extracted_deficit_int = (
            int(round(extracted_deficit))
            if extracted_deficit is not None
            else None
        )
        contingency_params[CONTINGENCY_ROLLOVER] = {
            "extracted_max_deficit": extracted_deficit_int,
            "max_deficit": min(
                max(extracted_deficit_int, 0),
                DEFAULT_ROLLOVER_MAX_DEFICIT,
            )
            if extracted_deficit_int is not None
            else DEFAULT_ROLLOVER_MAX_DEFICIT,
            "max_deficit_scope": "per_dish",
        }

    return {
        "dish_prices": dish_prices,
        "production_schedule": production_schedule,
        "payment_schedule": payment_schedule,
        "contingency_set": contingencies,
        "derived_contingency_mode": resolve_contingency_mode(contingencies),
        "contingency_mode_source": "offline_rule",
        "contingency_params": contingency_params,
    }


def parse_contracts_with_llm(
    contracts: list[dict[str, str]],
    *,
    provider: str = DEFAULT_PROVIDER,
    model: str | None = None,
) -> LLMBatchResult:
    selected_model = model or DEFAULT_MODELS[provider]
    prompt = PROMPT.format(
        contracts_json=json.dumps({"contracts": contracts}, indent=2)
    )
    raw = ""
    payload: dict[str, Any] | None = None
    last_error: ValueError | None = None
    for attempt in range(1, MAX_JSON_ATTEMPTS + 1):
        attempt_prompt = prompt
        if attempt > 1:
            attempt_prompt += (
                "\n\nYour previous response was not valid, complete JSON in the "
                "required shape. Return one complete JSON object only."
            )
        raw = _call_llm(provider, selected_model, attempt_prompt)
        try:
            payload = _response_payload_from_text(raw)
            break
        except (json.JSONDecodeError, ValueError) as error:
            last_error = error

    if payload is None:
        raise ValueError(
            f"LLM failed to return valid JSON after {MAX_JSON_ATTEMPTS} attempts"
        ) from last_error

    output: dict[str, dict[str, Any]] = {}
    for item in payload.get("contracts", []):
        if not isinstance(item, dict):
            continue
        contract_id = str(item.get("contract_id") or "").strip()
        if contract_id:
            output[contract_id] = normalize_terms(item)
    return LLMBatchResult(terms_by_id=output, raw_response=raw)


def parsed_contract_from_terms(text: str, terms: dict[str, Any]) -> ParsedContract:
    """Adapt LLM terms to the existing offline validation representation."""
    prices: dict[str, Decimal | None] = {}
    issues: list[str] = []
    for dish in DISH_PATTERNS:
        value = numeric((terms.get("dish_prices") or {}).get(dish))
        prices[dish] = Decimal(str(value)) if value is not None else None
        if value is None:
            issues.append(f"missing_price:{dish}")

    deliveries: dict[int, dict[str, int]] = {}
    for row in terms.get("production_schedule") or []:
        week_value = numeric(row.get("week")) if isinstance(row, dict) else None
        if week_value is None or int(week_value) not in PRODUCTION_WEEKS:
            continue
        values = {dish: numeric(row.get(dish)) for dish in DISH_PATTERNS}
        if all(value is not None for value in values.values()):
            deliveries[int(week_value)] = {
                dish: int(round(value or 0)) for dish, value in values.items()
            }
    if set(deliveries) != set(PRODUCTION_WEEKS):
        issues.append("delivery_weeks:" + ",".join(map(str, sorted(deliveries))))

    payments: dict[int, Decimal] = {}
    for row in terms.get("payment_schedule") or []:
        week_value = numeric(row.get("week")) if isinstance(row, dict) else None
        amount = numeric(row.get("amount")) if isinstance(row, dict) else None
        if (
            week_value is not None
            and int(week_value) in PAYMENT_WEEKS
            and amount is not None
        ):
            payments[int(week_value)] = Decimal(str(amount))
    if set(payments) != set(PAYMENT_WEEKS):
        issues.append("payment_weeks:" + ",".join(map(str, sorted(payments))))

    # Keep the simple, explicit stated-total checks from the offline parser.
    # Core prices, quantities, payments, and contingencies all come from the LLM.
    stated = parse_contract_deterministically(text)
    return ParsedContract(
        text=text,
        prices=prices,
        deliveries=deliveries,
        payments=payments,
        reported_contract_values=[
            _canonical_decimal(value) for value in stated.reported_contract_values
        ],
        reported_payment_totals=[
            _canonical_decimal(value) for value in stated.reported_payment_totals
        ],
        contingencies=list(
            terms.get("contingency_set")
            or terms.get("contingencies")
            or []
        ),
        parse_issues=issues,
    )


def solver_contract_from_terms(
    terms: dict[str, Any],
    *,
    environment_id: int,
) -> dict[str, Any]:
    """Translate normalized LLM terms without discarding contingency metadata."""
    parsed = parsed_contract_from_terms("", terms)
    if not parsed.complete:
        raise ValueError("Cannot solve a contract with missing LLM-parsed terms")
    production_rows = sorted(
        terms["production_schedule"], key=lambda row: int(row["week"])
    )
    payment_rows = sorted(
        terms["payment_schedule"], key=lambda row: int(row["week"])
    )

    def exact_number(value: Any) -> int | float:
        selected = numeric(value)
        if selected is None:
            raise ValueError("Cannot solve a contract containing a null number")
        return int(selected) if abs(selected - round(selected)) < 1e-9 else selected

    explicit_contingencies = list(
        terms.get("contingency_set")
        or terms.get("contingencies")
        or []
    )
    effective_contingencies = list(explicit_contingencies)
    if CONTINGENCY_GRIM not in effective_contingencies:
        effective_contingencies.append(CONTINGENCY_GRIM)
    effective_mode = resolve_contingency_mode(effective_contingencies)
    if effective_mode is None:
        effective_mode = CONTINGENCY_GRIM
    effective_params = {
        str(name): dict(block) if isinstance(block, dict) else block
        for name, block in (terms.get("contingency_params") or {}).items()
    }
    if effective_mode in {CONTINGENCY_PAYDED, CONTINGENCY_PAYDED_ROLLOVER}:
        payment_params = dict(effective_params.get(CONTINGENCY_PAYDED) or {})
        payment_params["deduction_prices"] = {
            dish: exact_number(terms["dish_prices"][dish]) for dish in DISH_PATTERNS
        }
        payment_params["deduction_prices_source"] = "agreed_dish_prices"
        effective_params[CONTINGENCY_PAYDED] = payment_params
    if effective_mode in {CONTINGENCY_ROLLOVER, CONTINGENCY_PAYDED_ROLLOVER}:
        rollover_params = dict(effective_params.get(CONTINGENCY_ROLLOVER) or {})
        extracted_deficit = numeric(rollover_params.get("max_deficit"))
        rollover_params["max_deficit"] = (
            min(
                max(int(round(extracted_deficit)), 0),
                DEFAULT_ROLLOVER_MAX_DEFICIT,
            )
            if extracted_deficit is not None
            else DEFAULT_ROLLOVER_MAX_DEFICIT
        )
        rollover_params["max_deficit_scope"] = "per_dish"
        effective_params[CONTINGENCY_ROLLOVER] = rollover_params

    return {
        "environment_id": environment_id,
        "contract_qty": {
            dish: int(production_rows[0][dish]) for dish in DISH_PATTERNS
        },
        "contract_qty_schedule": [
            {dish: int(row[dish]) for dish in DISH_PATTERNS}
            for row in production_rows
        ],
        "dish_prices": {
            dish: exact_number(terms["dish_prices"][dish]) for dish in DISH_PATTERNS
        },
        "payments": [exact_number(row["amount"]) for row in payment_rows[:-1]],
        "post_game_payment": exact_number(payment_rows[-1]["amount"]),
        "sat_thresh": 0.95,
        "contingencies": effective_contingencies,
        "contingency_mode": effective_mode,
        "contingency_combination": resolve_effective_contingency_combination(
            effective_contingencies
        ),
        "contingency_params": effective_params,
    }

