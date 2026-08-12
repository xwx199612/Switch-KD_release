"""Single-image parsing-based state tracking."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from .state_registry import StateRegistry, StateResolution


class StateObservationError(ValueError):
    """Raised when parsing output cannot be projected into a state observation."""


class StateObservationBuilder:
    """Deterministically project parsing elements into a state observation."""

    def build(
        self,
        elements: list[dict[str, Any]],
        focused_index: int | None = None,
    ) -> dict[str, list[str]]:
        """Project parsed elements into the StateFingerprint observation shape."""
        if not isinstance(elements, list):
            raise StateObservationError("parsing elements must be a list")

        visible_texts: list[str] = []
        if focused_index is not None and (
            isinstance(focused_index, bool)
            or not isinstance(focused_index, int)
            or not 0 <= focused_index < len(elements)
        ):
            raise StateObservationError("focused_index is outside the parsed element range")
        for index, element in enumerate(elements, start=1):
            if not isinstance(element, Mapping):
                raise StateObservationError(f"parsing element {index} must be an object")
            text = element.get("text")
            if not isinstance(text, str) or not text.strip():
                raise StateObservationError(f"parsing element {index} must have non-empty text")
            text = text.strip()
            visible_texts.append(text)

        return {
            "elements": visible_texts,
            "focus_path": [] if focused_index is None else [visible_texts[focused_index]],
        }

    __call__ = build


def build_state_observation(
    elements: list[dict[str, Any]], focused_index: int | None = None
) -> dict[str, list[str]]:
    """Build a state observation using the default deterministic adapter."""
    return StateObservationBuilder().build(elements, focused_index)


class StateTracker:
    """Resolve a sequence of single-image observations in one session registry."""

    def __init__(
        self,
        registry: StateRegistry,
        observer: Callable[[Any], dict[str, Any]],
    ) -> None:
        self.registry = registry
        self.observer = observer
        self.current_observation: dict[str, Any] | None = None
        self.current_resolution: StateResolution | None = None
        self.current_debug: dict[str, Any] = {}
        self._started = False

    def start(self, image: Any) -> StateResolution:
        """Observe and resolve the first image; may only be called once."""
        if self._started:
            raise RuntimeError("StateTracker.start() may only be called once")
        self._started = True
        return self._observe_and_resolve(image)

    def step(self, image: Any) -> StateResolution:
        """Observe and resolve the next image after ``start``."""
        if not self._started:
            raise RuntimeError("StateTracker.step() requires start() first")
        return self._observe_and_resolve(image)

    def _observe_and_resolve(self, image: Any) -> StateResolution:
        observation = self.observer(image)
        self.current_debug = dict(getattr(self.observer, "last_debug", {}) or {})
        resolution = self.registry.resolve(observation)
        self.current_observation = observation
        self.current_resolution = resolution
        return resolution


__all__ = [
    "StateObservationError",
    "StateObservationBuilder",
    "StateTracker",
    "build_state_observation",
]
