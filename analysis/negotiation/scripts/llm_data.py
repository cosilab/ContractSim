"""Shared collection and caching helpers for negotiation LLM analysis."""

from __future__ import annotations

import hashlib
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from analyze import analyze_all, read_json
from llm_contract_parser import parse_contracts_with_llm


SCHEMA_VERSION = 12


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def sha256_text(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def collect_contract_inputs(
    runs_root: Path,
    env_root: Path,
    campaign_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Collect raw proposal records without interpreting contract semantics."""
    negotiations, offer_rows = analyze_all(
        runs_root,
        env_root,
        campaign_root=campaign_root,
    )
    offers_by_run: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for offer in offer_rows:
        offers_by_run[str(offer["run_key"])].append(offer)

    inputs: list[dict[str, Any]] = []
    for negotiation in negotiations:
        run_key = str(negotiation["run_key"])
        source_run_key = str(negotiation["source_run_key"])
        environment_id = int(negotiation["environment_id"])
        source_environment_id = int(negotiation["source_environment_id"])
        matchup = str(negotiation["matchup"])
        matchup_dir = Path(str(negotiation["summary_path"])).parents[1]
        source_matchup_relpath = str(matchup_dir.relative_to(runs_root))
        offers = sorted(
            offers_by_run.get(run_key, []),
            key=lambda row: int(row["offer_index"]),
        )
        matching_final_indices = [
            int(offer["offer_index"])
            for offer in offers
            if bool(offer.get("is_final"))
        ]
        accepted_offer_index = (
            max(matching_final_indices)
            if negotiation["accepted"] and matching_final_indices
            else None
        )

        for offer in offers:
            offer_index = int(offer["offer_index"])
            inputs.append(
                {
                    "record_id": f"{source_run_key}/offer-{offer_index:04d}",
                    "record_type": "proposal",
                    "run_key": run_key,
                    "environment_id": environment_id,
                    "source_environment_id": source_environment_id,
                    "matchup": matchup,
                    "offer_index": offer_index,
                    "round_number": int(offer["round"]),
                    "proposer_role": str(offer["proposer_role"]),
                    "proposer_provider": str(offer["proposer_provider"]),
                    "proposer_model": str(offer["proposer_model"]),
                    "is_accepted_final": offer_index == accepted_offer_index,
                    "contract_text": str(offer["contract_text"]),
                    "source_matchup_relpath": source_matchup_relpath,
                }
            )

        if negotiation["accepted"] and accepted_offer_index is None:
            summary = read_json(Path(str(negotiation["summary_path"])))
            final_text = str(summary.get("contract_text") or "").strip()
            if final_text:
                inputs.append(
                    {
                        "record_id": f"{source_run_key}/final",
                        "record_type": "supplemental_final",
                        "run_key": run_key,
                        "environment_id": environment_id,
                        "source_environment_id": source_environment_id,
                        "matchup": matchup,
                        "offer_index": None,
                        "round_number": int(negotiation["rounds_started"]),
                        "proposer_role": "unknown",
                        "proposer_provider": "unknown",
                        "proposer_model": "unknown",
                        "is_accepted_final": True,
                        "contract_text": final_text,
                        "source_matchup_relpath": source_matchup_relpath,
                    }
                )
    return inputs


def _batches(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def _term_cache_path(output: Path, record_id: str) -> Path:
    return output / "llm-terms" / f"{sha256_text(record_id)}.json"


def mirrored_term_path(data_root: Path, item: Mapping[str, Any]) -> Path:
    """Location of one parsed proposal in the run-mirrored analysis tree."""
    relative = item.get("source_matchup_relpath")
    if not relative:
        raise ValueError(f"Missing source_matchup_relpath for {item.get('record_id')}")
    record_id = str(item["record_id"])
    leaf = record_id.rsplit("/", 1)[-1]
    return data_root / str(relative) / "data" / "contract_terms" / f"{leaf}.json"


def prepare_llm_terms(
    *,
    contract_inputs: list[dict[str, Any]],
    output: Path,
    provider: str,
    model: str,
    batch_size: int,
    force_llm_parse: bool,
    cache_path_for: Callable[[Mapping[str, Any]], Path] | None = None,
    batch_audit_dir: Path | None = None,
    max_parallel: int = 1,
    allow_api: bool = False,
) -> dict[str, dict[str, Any]]:
    """Parse raw offers in cached, auditable batches using the selected LLM."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_parallel < 1:
        raise ValueError("max_parallel must be positive")
    parsed: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, Any]] = []
    for item in contract_inputs:
        record_id = str(item["record_id"])
        raw_hash = sha256_text(str(item["contract_text"]))
        cache_path = (
            cache_path_for(item)
            if cache_path_for is not None
            else _term_cache_path(output, record_id)
        )
        if cache_path.exists() and not force_llm_parse:
            cache = read_json(cache_path)
            if (
                cache.get("schema_version") == SCHEMA_VERSION
                and cache.get("record_id") == record_id
                and cache.get("raw_contract_sha256") == raw_hash
                and cache.get("provider") == provider
                and cache.get("model") == model
                and isinstance(cache.get("structured_terms"), dict)
            ):
                parsed[record_id] = cache["structured_terms"]
                continue
        missing.append(item)

    if missing and not allow_api:
        raise RuntimeError(
            f"{len(missing)} parsed proposal records are missing or stale; "
            "rerun with --allow-api to authorize LLM calls"
        )

    batches = _batches(missing, batch_size)
    print(
        f"LLM term parsing: {len(parsed)} cached, {len(missing)} pending "
        f"in {len(batches)} batches via {provider}/{model}; "
        + (
            "sequential API calls"
            if max_parallel == 1
            else f"up to {max_parallel} parallel API calls"
        ),
        flush=True,
    )

    def parse_batch(
        batch_index: int,
        batch: list[dict[str, Any]],
    ) -> tuple[int, list[dict[str, Any]], list[dict[str, str]], Any, str]:
        contracts = [
            {
                "contract_id": str(item["record_id"]),
                "contract_text": str(item["contract_text"]),
            }
            for item in batch
        ]
        expected_ids = [item["contract_id"] for item in contracts]
        print(
            f"[LLM parse {batch_index}/{len(batches)}] starting "
            f"{len(batch)} contract{'s' if len(batch) != 1 else ''}",
            flush=True,
        )
        result = parse_contracts_with_llm(
            contracts,
            provider=provider,
            model=model,
        )
        returned_ids = set(result.terms_by_id)
        if returned_ids != set(expected_ids):
            missing_ids = sorted(set(expected_ids) - returned_ids)
            unexpected_ids = sorted(returned_ids - set(expected_ids))
            raise RuntimeError(
                "LLM parser did not preserve the batch contract IDs: "
                f"missing={missing_ids}, unexpected={unexpected_ids}. "
                "No partial batch was cached; rerun to retry."
            )
        batch_key = sha256_text(
            canonical_json(
                {
                    "provider": provider,
                    "model": model,
                    "records": [
                        {
                            "record_id": item["contract_id"],
                            "raw_contract_sha256": sha256_text(item["contract_text"]),
                        }
                        for item in contracts
                    ],
                }
            )
        )
        return batch_index, batch, contracts, result, batch_key

    def cache_batch(
        parsed_batch: tuple[
            int,
            list[dict[str, Any]],
            list[dict[str, str]],
            Any,
            str,
        ],
    ) -> None:
        batch_index, batch, contracts, result, batch_key = parsed_batch
        expected_ids = [item["contract_id"] for item in contracts]
        atomic_write_json(
            (batch_audit_dir or (output / "llm-batches")) / f"{batch_key}.json",
            {
                "schema_version": SCHEMA_VERSION,
                "parsed_at": now_iso(),
                "provider": provider,
                "model": model,
                "contract_ids": expected_ids,
                "raw_response": result.raw_response,
            },
        )
        for item in batch:
            record_id = str(item["record_id"])
            terms = result.terms_by_id[record_id]
            atomic_write_json(
                (
                    cache_path_for(item)
                    if cache_path_for is not None
                    else _term_cache_path(output, record_id)
                ),
                {
                    "schema_version": SCHEMA_VERSION,
                    "parsed_at": now_iso(),
                    "record_id": record_id,
                    "raw_contract_sha256": sha256_text(str(item["contract_text"])),
                    "provider": provider,
                    "model": model,
                    "batch_sha256": batch_key,
                    "structured_terms": terms,
                },
            )
            parsed[record_id] = terms
        print(
            f"[LLM parse {batch_index}/{len(batches)}] cached "
            f"{len(batch)} contract{'s' if len(batch) != 1 else ''}",
            flush=True,
        )

    if max_parallel == 1:
        for batch_index, batch in enumerate(batches, start=1):
            cache_batch(parse_batch(batch_index, batch))
        return parsed

    first_error: Exception | None = None
    with ThreadPoolExecutor(
        max_workers=min(max_parallel, len(batches)) if batches else 1,
        thread_name_prefix="contract-parser",
    ) as executor:
        futures = [
            executor.submit(parse_batch, batch_index, batch)
            for batch_index, batch in enumerate(batches, start=1)
        ]
        for future in as_completed(futures):
            try:
                cache_batch(future.result())
            except Exception as exc:
                if first_error is None:
                    first_error = exc
    if first_error is not None:
        raise first_error
    return parsed
