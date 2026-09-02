#!/usr/bin/env python3
"""Populate run-mirrored proposal terms with a batched parser LLM."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from analyze import (
    DEFAULT_ENV_ROOT,
    campaign_inferences_root,
    campaign_runs_root,
)
from llm_data import (
    SCHEMA_VERSION,
    atomic_write_json,
    collect_contract_inputs,
    mirrored_term_path,
    now_iso,
    prepare_llm_terms,
)
from llm_contract_parser import DEFAULT_BATCH_SIZE, DEFAULT_MODEL, DEFAULT_PROVIDER


def _selected_inputs(
    runs_root: Path,
    env_root: Path,
    environments: set[int] | None,
    campaign_root: Path,
) -> list[dict[str, Any]]:
    records = collect_contract_inputs(
        runs_root,
        env_root,
        campaign_root=campaign_root,
    )
    if environments is not None:
        records = [
            record
            for record in records
            if int(record["environment_id"]) in environments
        ]
    return records


def ensure_mirrored_directories(
    records: list[dict[str, Any]],
    data_root: Path,
) -> None:
    for record in records:
        mirrored_term_path(data_root, record).parent.mkdir(parents=True, exist_ok=True)


def write_matchup_indexes(
    records: list[dict[str, Any]],
    data_root: Path,
) -> None:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["source_matchup_relpath"])].append(record)
    for relative, matchup_records in grouped.items():
        rows = []
        for record in sorted(
            matchup_records,
            key=lambda item: (
                item["offer_index"] is None,
                int(item["offer_index"] or 0),
            ),
        ):
            path = mirrored_term_path(data_root, record)
            rows.append(
                {
                    "record_id": record["record_id"],
                    "offer_index": record["offer_index"],
                    "round": record["round_number"],
                    "proposer_role": record["proposer_role"],
                    "proposer_provider": record["proposer_provider"],
                    "proposer_model": record["proposer_model"],
                    "is_accepted_final": record["is_accepted_final"],
                    "terms_file": path.name,
                }
            )
        atomic_write_json(
            data_root / relative / "data" / "contract_terms" / "index.json",
            {
                "schema_version": SCHEMA_VERSION,
                "source_matchup_relpath": relative,
                "records": rows,
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--env-root", type=Path, default=DEFAULT_ENV_ROOT)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--env", type=int, action="append", choices=range(1, 7))
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai", "gemini"],
        default=DEFAULT_PROVIDER,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parsing_mode = parser.add_mutually_exclusive_group()
    parsing_mode.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="contracts per sequential API call (default: %(default)s)",
    )
    parsing_mode.add_argument(
        "--single-contract-parallel",
        type=int,
        metavar="N",
        help="send one contract per API call with up to N calls in parallel",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reparse compatible cached records instead of resuming from them.",
    )
    parser.add_argument(
        "--allow-api",
        action="store_true",
        help="authorize LLM calls for missing, stale, or forced records",
    )
    parser.add_argument(
        "--structure-only",
        action="store_true",
        help="Create the mirrored directories without making LLM calls.",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if (
        args.single_contract_parallel is not None
        and args.single_contract_parallel < 1
    ):
        parser.error("--single-contract-parallel must be positive")

    runs_root = (args.runs_root or campaign_runs_root(args.campaign)).resolve()
    env_root = args.env_root.resolve()
    campaign_root = campaign_inferences_root(args.campaign)
    data_root = (args.data_root or campaign_root / "parsed").resolve()
    environments = set(args.env) if args.env else None
    records = _selected_inputs(
        runs_root,
        env_root,
        environments,
        campaign_root,
    )
    ensure_mirrored_directories(records, data_root)
    if args.structure_only:
        print(
            f"Created the mirrored data layout for {len(records)} proposals "
            f"under {data_root}",
            flush=True,
        )
        return

    parsed = prepare_llm_terms(
        contract_inputs=records,
        output=data_root,
        provider=args.provider,
        model=args.model,
        batch_size=1 if args.single_contract_parallel is not None else args.batch_size,
        force_llm_parse=args.force,
        cache_path_for=lambda item: mirrored_term_path(data_root, item),
        batch_audit_dir=data_root / "_llm-batches",
        max_parallel=args.single_contract_parallel or 1,
        allow_api=args.allow_api,
    )
    write_matchup_indexes(records, data_root)
    atomic_write_json(
        data_root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": now_iso(),
            "runs_root": str(runs_root),
            "env_root": str(env_root),
            "environment_filter": sorted(environments) if environments else None,
            "provider": args.provider,
            "model": args.model,
            "batch_size": (
                1 if args.single_contract_parallel is not None else args.batch_size
            ),
            "max_parallel_api_calls": args.single_contract_parallel or 1,
            "single_contract_parallel": args.single_contract_parallel is not None,
            "n_records": len(records),
            "n_parsed": len(parsed),
            "layout": "<environment>/run <n>/<matchup>/data/contract_terms/<offer>.json",
            "batch_audit_path": "_llm-batches",
        },
    )
    print(f"Prepared {len(parsed)} proposal terms under {data_root}", flush=True)


if __name__ == "__main__":
    main()
