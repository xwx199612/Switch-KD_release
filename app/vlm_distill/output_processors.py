from __future__ import annotations

import unicodedata
from typing import Any, Protocol

from .data_manifest import VlmSample
from .parsing_output_parser import parse_parsing_answer
from .transition_output_parser import parse_transition_output


class OutputProcessor(Protocol):
    mode: str

    def process(
        self,
        *,
        sample: VlmSample,
        raw_output: str,
        backend_result: dict[str, Any],
    ) -> dict[str, Any]:
        ...


class GenericTextOutputProcessor:
    mode = "text"

    def process(
        self,
        *,
        sample: VlmSample,
        raw_output: str,
        backend_result: dict[str, Any],
    ) -> dict[str, Any]:
        del sample
        result: dict[str, Any] = {
            "usable": True,
            "raw_model_output": raw_output.strip(),
            "answer": raw_output.strip(),
            "student_answer": raw_output.strip(),
        }
        if backend_result.get("inference_debug"):
            result["inference_debug"] = backend_result["inference_debug"]
        return result


class ParsingOutputProcessor:
    mode = "parsing"

    def process(
        self,
        *,
        sample: VlmSample,
        raw_output: str,
        backend_result: dict[str, Any],
    ) -> dict[str, Any]:
        del sample
        result = {
            "raw_model_output": raw_output,
            **parse_parsing_answer(raw_output),
        }
        result["elements"] = _deduplicate_parsing_elements(result.get("elements", []))
        result["element_count"] = len(result["elements"])
        result["usable"] = bool(result["elements"])
        if backend_result.get("inference_debug"):
            result["inference_debug"] = backend_result["inference_debug"]
        return result


def _deduplicate_parsing_elements(elements: Any) -> list[dict[str, Any]]:
    """Keep the first parsing element for each normalized text/bbox pair."""
    if not isinstance(elements, list):
        return []

    seen: set[tuple[str, tuple[Any, ...]]] = set()
    deduplicated: list[dict[str, Any]] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        text = element.get("text")
        bbox = element.get("bbox_norm")
        if not isinstance(text, str) or not isinstance(bbox, list):
            deduplicated.append(element)
            continue
        normalized_text = " ".join(
            unicodedata.normalize("NFKC", text).strip().casefold().split()
        )
        duplicate_key = (normalized_text, tuple(bbox))
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)
        deduplicated.append(element)
    return deduplicated


class TransitionOutputProcessor:
    mode = "transition"

    def process(self, *, sample: VlmSample, raw_output: str, backend_result: dict[str, Any]) -> dict[str, Any]:
        del sample
        result = {"raw_output": raw_output, **parse_transition_output(raw_output)}
        if backend_result.get("inference_debug"):
            result["inference_debug"] = backend_result["inference_debug"]
        return result


def build_output_processor(mode: str) -> OutputProcessor:
    normalized = str(mode).strip().lower()
    if normalized == "text":
        return GenericTextOutputProcessor()
    if normalized == "parsing":
        return ParsingOutputProcessor()
    if normalized == "transition":
        return TransitionOutputProcessor()
    raise ValueError(f"Unsupported output mode: {mode!r}; expected 'text', 'parsing', or 'transition'.")
