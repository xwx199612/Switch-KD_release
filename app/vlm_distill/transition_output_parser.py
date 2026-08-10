"""Strict parser for the transition endpoint's before/after observations."""

from __future__ import annotations

import json
import re
from typing import Any


def _strip_fence(raw_output: str) -> str:
    text = raw_output.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else text


def _validate_list_of_strings(state: dict[str, Any], state_name: str, field_name: str) -> list[str]:
    path = f"{state_name}.{field_name}"
    if field_name not in state:
        raise ValueError(f"missing transition field: {path}")
    value = state[field_name]
    if not isinstance(value, list):
        raise ValueError(f"transition field {path} must be a list")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"transition field {path} must contain only strings")
    return list(value)


def _validate_state(state: Any, state_name: str) -> dict[str, list[str]]:
    if not isinstance(state, dict):
        raise ValueError(f"transition field {state_name} must be an object")
    unexpected = sorted(set(state) - {"elements", "focus_path"})
    if unexpected:
        raise ValueError(f"unexpected transition fields in {state_name}: {unexpected}")
    return {
        "elements": _validate_list_of_strings(state, state_name, "elements"),
        "focus_path": _validate_list_of_strings(state, state_name, "focus_path"),
    }


def parse_transition_output(raw_output: str) -> dict[str, Any]:
    """Parse and validate the model's before/after state-observation JSON."""
    try:
        parsed = json.loads(_strip_fence(raw_output))
        if not isinstance(parsed, dict):
            raise ValueError("top-level JSON must be an object")

        for state_name in ("before", "after"):
            if state_name not in parsed:
                raise ValueError(f"missing transition field: {state_name}")
        unexpected = sorted(set(parsed) - {"before", "after"})
        if unexpected:
            raise ValueError(f"unexpected transition fields: {unexpected}")

        transition = {
            "before": _validate_state(parsed["before"], "before"),
            "after": _validate_state(parsed["after"], "after"),
        }
        return {"usable": True, "parse_error": None, "transition": transition}
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"usable": False, "parse_error": str(exc), "transition": None}
