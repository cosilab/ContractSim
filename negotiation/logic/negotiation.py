"""Negotiation phase — multi-round LLM negotiation via Concordia."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from concordia.typing import entity as entity_lib

from llm_utils import delimit_contract
from llm_utils.artifacts import write_text
from llm_utils.llm.providers import EmptyModelResponseError

from .responses import parse_negotiation_response, strip_name_prefix

_NEG_JSON_INSTR_CATERER = (
    ' Respond ONLY with a JSON object (no markdown fences).'
    '\nTo negotiate: {"response": "<dialogue>", "contract": "<contract>"}'
    '\nTo terminate and walk away: {"response": "<dialogue>", "terminate": true}'
    '\n"response" must sound like a real face-to-face conversation — casual, concise,'
    ' with natural filler, interruptions, and short sentences. No formal greetings,'
    ' no "Thank you for your proposal", no email-speak.'
    '\nFormat "contract" with each term on its own line using \\n.'
    '\nInclude "contract" when proposing, revising, or agreeing to the customer\'s terms'
    ' (echo the agreed contract back so the customer can formally accept it).'
    ' Omit it only when no contract terms are under discussion.'
    '\nSet "terminate": true only when you have made a final decision to walk away.'
    ' Do not include "contract" when terminating.'
    '\nIMPORTANT: "Week" is the smallest unit of time in this environment.'
    ' You CANNOT break a week into individual days or negotiate'
    ' day-level scheduling. All delivery and payment terms must be specified per week.'
)
_NEG_JSON_INSTR_CUSTOMER = (
    ' Respond ONLY with a JSON object (no markdown fences).'
    '\nTo negotiate: {"response": "<dialogue>", "contract": "<contract>"}'
    '\nTo accept: {"response": "<dialogue>", "accept": true}'
    '\nTo terminate and walk away: {"response": "<dialogue>", "terminate": true}'
    '\n"response" must sound like a real face-to-face conversation — casual, concise,'
    ' with natural filler, interruptions, and short sentences. No formal greetings,'
    ' no "Thank you for your proposal", no email-speak.'
    '\nFormat "contract" with each term on its own line using \\n.'
    '\nInclude "contract" when proposing or revising terms. Omit it otherwise.'
    '\nSet "terminate": true only when you have made a final decision to walk away.'
    ' Do not include "contract" or "accept" when terminating.'
    '\nIMPORTANT: "Week" is the smallest unit of time in this environment.'
    ' You CANNOT break a week into individual days or negotiate'
    ' day-level scheduling. All delivery and payment terms must be specified per week.'
)


def run_negotiation(
    agents: Sequence[Any],
    models: Sequence[Any],
    max_rounds: int = 50,
    log_path: str | Path = "negotiation_log.txt",
    contract_path: str | Path = "contract.txt",
    record_llm_call: Callable[..., None] | None = None,
) -> tuple[str | None, bool, dict[str, Any]]:
    """Run the protocol for two already-constructed agents and their models."""
    if len(agents) != 2 or len(models) != 2:
        raise ValueError("negotiation requires exactly two agents and two models")
    record_call = record_llm_call or (lambda *_args, **_kwargs: None)
    print(f"\n{'='*60}\n  NEGOTIATION PHASE\n{'='*60}")

    log_lines = []
    last_msg = None
    pending_contract = None
    pending_proposer = None   # "Customer" or "Caterer"
    pending_round = None      # round number when proposed
    turns_taken = 0

    def log(text):
        print(text)
        log_lines.append(text)

    def broadcast(speaker, msg, contract=None):
        observation = f"{speaker}: {msg}"
        if contract:
            observation += (
                f"\n\n[CONTRACT PROPOSAL by {speaker}]\n"
                f"{delimit_contract(contract)}"
            )
        for a in agents:
            a.observe(observation)

    def save(final_msg, contract=None):
        log(final_msg)
        write_text(log_path, "\n".join(log_lines))
        print(f">>> Chat log saved to {log_path}")
        contract_file = Path(contract_path)
        if contract:
            border = "=" * 50
            body = "\n".join(l.strip() for l in contract.split("\n") if l.strip())
            write_text(
                contract_file,
                f"{border}\n  AGREED CONTRACT\n{border}\n{body}\n{border}\n",
            )
            print(f">>> Contract saved to {contract_path}")
        elif contract_file.exists():
            contract_file.unlink()

    def _contract_context():
        """One-liner describing pending contract for CTAs."""
        if not pending_contract:
            return ""
        return (
            f'\nThe most recent contract was proposed by {pending_proposer} in round {pending_round}:'
            f'\n{delimit_contract(pending_contract)}'
        )

    for rnd in range(1, max_rounds + 1):
        log(f"\n--- Round {rnd} of {max_rounds} ---")

        for idx, agent in enumerate(agents):
            is_caterer = (idx == 1)

            # --- Build call-to-action ---
            if rnd == 1 and idx == 0:
                cta = (
                    f"Start the negotiation. You have {max_rounds} rounds total to reach a deal. "
                    "Only propose or revise a contract when the conversation "
                    "has enough information "
                    "to draft one. You may terminate the negotiation and walk away at any time; "
                    'to do so, set "terminate": true in your JSON response.'
                )
            elif not is_caterer and pending_proposer == agents[1].name and pending_round == rnd - 1:
                # Fast path: caterer proposed a contract in the immediately preceding round
                cta = (
                    f'The caterer proposed this contract in round {pending_round}:'
                    f'\n{delimit_contract(pending_contract)}\n'
                    f'They said: "{last_msg}". '
                    f'This is round {rnd} of {max_rounds}. '
                    'If you agree to this contract, set "accept": true in your JSON response to finalize the deal. '
                    'Do NOT include a "contract" field when accepting — if you do, your proposal will be ignored. '
                    "Otherwise counter-offer/revise, argue, or terminate the negotiation and walk away "
                    'by setting "terminate": true in your JSON response.'
                )
            elif not is_caterer:
                # Customer's turn but no recent caterer contract to accept
                cta = f'The caterer said: "{last_msg}". '
                if pending_contract:
                    cta += f'\n(For reference:{_contract_context()})\n'
                cta += (
                    f"This is round {rnd} of {max_rounds}. "
                    "Counter-offer, argue, propose/revise a contract, or terminate the negotiation and walk away "
                    'by setting "terminate": true in your JSON response. '
                    "Only propose or revise a contract when the conversation "
                    "has enough information "
                    "to draft one. "
                    "If you are happy with the agreed terms, ask the Caterer to write up "
                    "a formal contract so you can accept it."
                )
            elif is_caterer and pending_proposer == agents[0].name:
                # Caterer sees customer's proposed contract
                cta = (
                    f'The customer proposed this contract in round {pending_round}:'
                    f'\n{delimit_contract(pending_contract)}\n'
                    f'They said: "{last_msg}". '
                    f'This is round {rnd} of {max_rounds}. '
                    "If the terms work for you, write up a contract keeping in mind the agreed terms. "
                    "Otherwise counter-offer/revise, argue, or terminate the negotiation and walk away "
                    'by setting "terminate": true in your JSON response.'
                )
            else:
                # Caterer's turn, no customer contract pending
                cta = f'The customer said: "{last_msg}". '
                if pending_contract:
                    cta += f'\n(For reference:{_contract_context()})\n'
                cta += (
                    f"This is round {rnd} of {max_rounds}. "
                    "Answer their questions, counter-offer/revise, argue, or terminate the negotiation and walk away "
                    'by setting "terminate": true in your JSON response. '
                    "Only propose or revise a contract when the conversation "
                    "has enough information "
                    "to draft one."
                )

            json_instr = _NEG_JSON_INSTR_CATERER if is_caterer else _NEG_JSON_INSTR_CUSTOMER
            model = models[idx]
            try:
                raw = agent.act(entity_lib.ActionSpec(
                    call_to_action=cta + json_instr,
                    output_type=entity_lib.OutputType.FREE,
                    tag='negotiation_message',
                ))
            except EmptyModelResponseError as exc:
                invalid_status = {
                    "valid": False,
                    "invalid_reason": "empty_model_response",
                    "turns_taken": turns_taken,
                    "invalid_agent": agent.name,
                    "provider": exc.provider,
                    "model": exc.model_name,
                    "round": rnd,
                    "diagnostics": exc.diagnostics,
                }
                record_call(
                    agent.name,
                    model,
                    raw_output="",
                    parser_input="",
                    parse_meta={"parsed": False, "error": "empty_model_response"},
                    context={
                        "phase": "negotiation",
                        "action": "negotiate",
                        "round": rnd,
                        "invalid_run": True,
                        "call_to_action": cta + json_instr,
                    },
                )
                save(
                    f"\n>>> INVALID RUN — {agent.name} returned no visible text "
                    "after one retry. No blank message was broadcast."
                )
                return None, False, invalid_status
            turns_taken += 1
            parser_input = strip_name_prefix(agent.name, raw)
            response, contract, accept, terminate, parse_meta = parse_negotiation_response(
                parser_input
            )
            record_call(
                agent.name,
                model,
                raw_output=raw,
                parser_input=parser_input,
                parsed_json=parse_meta.get("parsed_json") or parse_meta.get("parsed_fields"),
                parse_meta=parse_meta,
                context={
                    "phase": "negotiation",
                    "action": "negotiate",
                    "round": rnd,
                    "accepted": accept,
                    "terminated": terminate,
                    "call_to_action": cta + json_instr,
                },
            )

            if response is None:
                response = parser_input

            log(f"\n{agent.name}: {response}")

            # --- Handle terminate / accept / contract ---
            if terminate:
                broadcast(agent.name, response)
                save(f"\n>>> {agent.name.capitalize()} terminated the negotiation. No deal.")
                return None, False, {"valid": True, "turns_taken": turns_taken}
            elif not is_caterer and accept:
                if contract:
                    log(f"\n[CONTRACT by {agent.name} IGNORED — cannot propose and accept simultaneously]")
                    agent.observe(
                        "Your contract proposal was ignored because you set 'accept' at the same time. "
                        "You can only accept the Caterer's most recent contract, not propose your own."
                    )
                elif (pending_contract and pending_proposer == agents[1].name
                      and pending_round == rnd - 1):
                    # Fast path: caterer proposed last round — close the deal
                    broadcast(agent.name, response)
                    log(f"\n[FINAL CONTRACT]:\n{pending_contract}")
                    save("\n>>> Customer accepted. Deal closed.", pending_contract)
                    return (
                        pending_contract,
                        True,
                        {"valid": True, "turns_taken": turns_taken},
                    )
                else:
                    # Slow path: customer wants to close but no recent caterer contract
                    log(f"\n[ACCEPT DEFERRED — no recent caterer contract to accept]")
                    agent.observe(
                        "Your acceptance could not be processed because the Caterer has not formally "
                        "submitted a contract recently. If you are happy with the terms, ask the Caterer "
                        "to write up the agreed terms as a formal contract so you can accept it."
                    )
            elif contract:
                log(f"\n[CONTRACT by {agent.name}]:\n{contract}")
                pending_contract = contract
                pending_proposer = agent.name
                pending_round = rnd

            accepted_with_ignored_contract = bool(
                not is_caterer and accept and contract
            )
            broadcast(
                agent.name,
                response,
                contract=None if accepted_with_ignored_contract else contract,
            )

            last_msg = response

    save("\n>>> Max rounds reached — no explicit acceptance.")
    return None, False, {"valid": True, "turns_taken": turns_taken}
