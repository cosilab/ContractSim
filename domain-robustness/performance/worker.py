"""Execute and persist one domain-robustness performance matchup."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from contracts import load as load_contract
from domain import load_contract_views, metadata as domain_metadata
from domains import DOMAINS
from llm_runtime import save_domain_agent_states
from sides import CUSTOMER_BASELINES, SUPPLIER_BASELINES, LLMSide
from llm_utils import load_environment_metadata
from llm_utils.artifacts import claim_output_directory, write_json
from llm_utils.llm import EmptyModelResponseError
from performance.engine import run_performance
from performance.prompting import treatment_metadata


def run_one(
    *,
    domain: str,
    role: str,
    contracts_root: Path,
    contract_id: str,
    provider: str,
    baseline_name: str,
    seed: int,
    out_dir: Path,
    log_reasoning: bool,
    goal_treatment: str,
) -> dict[str, Any]:
    contracts_root = contracts_root.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    record = load_contract(contracts_root, contract_id)
    contract_dir: Path = record["contract_dir"]
    _canonical_contract, model_contract, _materialized = load_contract_views(
        contract_dir, domain
    )
    focal_customer = role == DOMAINS[domain]["performance_roles"][0]
    prompt_role = "customer" if focal_customer else "supplier"
    customer = provider if focal_customer else baseline_name
    supplier = baseline_name if focal_customer else provider
    treated_role = (
        None
        if goal_treatment == "control"
        else ("customer" if focal_customer else "supplier")
    )
    run_config = {
        **load_environment_metadata(5),
        **domain_metadata(domain),
        "implementation": "Performance",
        "contract_id": contract_id,
        "contract_dir": str(contract_dir),
        "customer": customer,
        "supplier": supplier,
        "seed": int(seed),
        **treatment_metadata(goal_treatment, treated_role),
    }

    llm_side = LLMSide(
        domain=domain,
        canonical_role=prompt_role,
        provider=provider,
        out_dir=out_dir,
        log_reasoning=log_reasoning,
        contract_text=model_contract,
        goal_treatment=goal_treatment,
    )
    if focal_customer:
        customer_side = llm_side
        supplier_side = SUPPLIER_BASELINES[baseline_name](
            contract_id, contract_dir, record["config"], record["terms"],
            environment_id=5,
        )
    else:
        customer_side = CUSTOMER_BASELINES[baseline_name](
            record["config"], record["terms"], environment_id=5
        )
        supplier_side = llm_side

    claim = claim_output_directory(
        out_dir,
        metadata={
            "implementation": run_config["implementation"],
            "environment_id": 5,
            "contract_id": contract_id,
        },
    )
    with claim:
        llm_side.prepare_artifacts()
        write_json(out_dir / "config" / "run_config.json", run_config)
        try:
            result = run_performance(
                customer=customer_side,
                supplier=supplier_side,
                terms=record["terms"],
                environment_id=5,
                seed=seed,
            )
            summary = {**run_config, "valid": True, **asdict(result)}
            terminal_status = "complete"
        except EmptyModelResponseError as exc:
            summary = {
                **run_config,
                "valid": False,
                "invalid_reason": "empty_model_response",
                "provider": exc.provider,
                "model": exc.model_name,
                "diagnostics": exc.diagnostics,
            }
            terminal_status = "invalid"
        write_json(out_dir / "data" / "summary.json", summary)
        role_index = 0 if focal_customer else 1
        save_domain_agent_states(
            out_dir,
            llm_side.runtime.agents,
            llm_side.runtime.configs,
            roles=(("customer" if focal_customer else "supplier"),),
            slugs=(DOMAINS[domain]["performance_slugs"][role_index],),
        )
        claim.finalize(
            terminal_status,
            {"valid": summary["valid"]},
        )
    return summary
