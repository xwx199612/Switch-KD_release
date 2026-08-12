"""Dedicated single-image navigation-focus resolution."""

from __future__ import annotations

import json
import time
from typing import Any

from .state_tracker import StateObservationError


FOCUS_RESOLVER_PROMPT_TEMPLATE = """Inspect the provided UI screenshot and identify which candidate is currently navigation-focused.

Compare candidates visually against their peer elements. Use only visible navigation-focus cues such as scale enlargement, border or outline, focus ring, elevation, raised appearance, or container emphasis. A card may be focused from border and/or scale alone; an app may be focused from a ring or enlarged container.

Do not infer focus from semantic importance, center position, item order, recommendation prominence, brightness alone, or prior knowledge. Return null only when no candidate is visually distinguishable.

Candidates:
{candidates}

Return only valid JSON with exactly this schema:
{"focused_index": integer or null}
"""


class FocusResolver:
    """Resolve visual navigation focus with an already-loaded VLM engine."""

    MAX_NEW_TOKENS = 64

    def __init__(self, engine: Any) -> None:
        self.engine = engine

    def resolve(self, image: Any, candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(candidates, list):
            raise StateObservationError("focus resolver candidates must be a list")
        candidate_lines = "\n".join(
            f'{index}. {candidate.get("text", "")} bbox={candidate.get("bbox_norm")}'
            for index, candidate in enumerate(candidates)
        )
        prompt = FOCUS_RESOLVER_PROMPT_TEMPLATE.replace("{candidates}", candidate_lines)
        started = time.perf_counter()
        raw_output = self.engine.generate_raw(image, prompt, self.MAX_NEW_TOKENS)
        elapsed = round(time.perf_counter() - started, 3)
        focused_index = self._parse_output(raw_output, len(candidates))
        debug = dict(getattr(self.engine, "last_debug", {}) or {})
        debug.update({
            "mode": "focus_resolution",
            "elapsed_seconds": elapsed,
            "generation_kwargs": {"do_sample": False, "max_new_tokens": self.MAX_NEW_TOKENS},
        })
        return {
            "focused_index": focused_index,
            "raw_output": raw_output,
            "inference_debug": debug,
        }

    @staticmethod
    def _parse_output(raw_output: str, candidate_count: int) -> int | None:
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise StateObservationError("Focus resolver output is empty")
        try:
            payload = json.loads(raw_output.strip())
        except (TypeError, json.JSONDecodeError) as exc:
            raise StateObservationError(f"Focus resolver output is malformed JSON: {exc}") from exc
        if not isinstance(payload, dict) or set(payload) != {"focused_index"}:
            raise StateObservationError(
                "Focus resolver output must be exactly {\"focused_index\": integer or null}"
            )
        focused_index = payload["focused_index"]
        if focused_index is None:
            return None
        if isinstance(focused_index, bool) or not isinstance(focused_index, int):
            raise StateObservationError("Focus resolver focused_index must be an integer or null")
        if not 0 <= focused_index < candidate_count:
            raise StateObservationError(
                f"Focus resolver focused_index {focused_index} is outside candidate range"
            )
        return focused_index


__all__ = ["FOCUS_RESOLVER_PROMPT_TEMPLATE", "FocusResolver"]
