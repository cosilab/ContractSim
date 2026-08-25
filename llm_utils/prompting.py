"""Low-level prompt rendering and readability formatting."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path


CONTRACT_BEGIN = "<<< BEGIN CONTRACT >>>"
CONTRACT_END = "<<< END CONTRACT >>>"
TOKEN_RE = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")

# Keep this list byte-compatible with the established v1 negotiation prompts.
# In particular, adding ``ENVIRONMENT CONDITIONS:`` would change existing
# prompt hashes and belongs in an explicit prompt-version migration.
_SYSTEM_HEADERS = (
    "BEST ALTERNATIVE TO NO AGREEMENT:",
    "CONTRACT EXECUTION ACTIONS:",
    "PRODUCTION WEEK PROCESS (in order):",
    "ENVIRONMENT STOCHASTICITY:",
    "DISH SUBSTITUTABILITY:",
    "MINIMUM REQUIREMENTS:",
    "NEGOTIATION PROCESS:",
    "PRIVATE VALUATION:",
    "OBJECTIVES:",
    "IMPORTANT:",
    "RECIPES:",
    "SCHEDULE:",
)
_HEADER_RE = re.compile(
    r"(^| )(" + "|".join(re.escape(header) for header in _SYSTEM_HEADERS) + r") ?",
    flags=re.MULTILINE,
)


class PromptRenderError(ValueError):
    """Raised when a prompt template cannot be rendered exactly."""


def _render_text(
    template: str,
    variables: Mapping[str, object],
    *,
    source: Path,
) -> str:
    literal_template = TOKEN_RE.sub("", template)
    if "{{" in literal_template or "}}" in literal_template:
        raise PromptRenderError(
            f"Unresolved or malformed template delimiter remains in {source}"
        )

    required = set(TOKEN_RE.findall(template))
    missing = sorted(required - variables.keys())
    if missing:
        raise PromptRenderError(f"Missing template variables for {source}: {missing}")

    rendered = TOKEN_RE.sub(
        lambda match: str(variables[match.group(1)]),
        template,
    ).strip()
    return rendered


def render_template(
    path: str | Path,
    variables: Mapping[str, object],
) -> str:
    """Substitute all uppercase double-braced variables in one template file."""
    if not isinstance(variables, Mapping):
        raise PromptRenderError("template variables must be a mapping")
    template_path = Path(path)
    try:
        template = template_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise PromptRenderError(f"Missing prompt template: {template_path}") from exc
    except OSError as exc:
        raise PromptRenderError(f"Cannot read prompt template {template_path}: {exc}") from exc
    return _render_text(template, variables, source=template_path)


def delimit_contract(contract: str) -> str:
    """Wrap contract text in visually unambiguous boundary markers."""
    return f"{CONTRACT_BEGIN}\n{str(contract).strip()}\n{CONTRACT_END}"


def _format_headers_outside_contracts(text: str) -> str:
    def format_chunk(chunk: str) -> str:
        def replace(match: re.Match[str]) -> str:
            prefix = "" if match.group(1) == "" else "\n\n"
            return f"{prefix}{match.group(2)}\n"

        return _HEADER_RE.sub(replace, chunk)

    pieces = str(text).split(CONTRACT_BEGIN)
    formatted = [format_chunk(pieces[0])]
    for piece in pieces[1:]:
        if CONTRACT_END not in piece:
            formatted.extend((CONTRACT_BEGIN, piece))
            continue
        contract, remainder = piece.split(CONTRACT_END, 1)
        formatted.extend(
            (
                CONTRACT_BEGIN,
                contract,
                CONTRACT_END,
                format_chunk(remainder),
            )
        )
    return "".join(formatted).strip()


def format_system_prompt(goal: str, info: str) -> str:
    """Combine a rendered goal and private information for an agent."""
    formatted_goal = _format_headers_outside_contracts(goal)
    formatted_info = _format_headers_outside_contracts(info)
    return f"Goal:\n\n{formatted_goal}\n\nPrivate Info:\n\n{formatted_info}"
