"""In-memory V1 registry and resolver for observed UI state fingerprints."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from .state_fingerprint import (
    SAME_STATE_THRESHOLD,
    StateFingerprint,
    build_state_fingerprint,
    match_state_fingerprints,
)


@dataclass(frozen=True)
class RegisteredState:
    """One runtime state and its immutable V1 canonical fingerprint."""

    state_id: str
    fingerprint: StateFingerprint


@dataclass(frozen=True)
class StateResolution:
    """The state selected or created for one observation."""

    state_id: str
    is_new: bool
    score: float


@dataclass(frozen=True)
class TransitionStateResolution:
    """State resolutions for a transition, resolved in before/after order."""

    before: StateResolution
    after: StateResolution


class TransitionInferencer(Protocol):
    """Callable that runs the existing parsed transition inference path."""

    def __call__(self, before_image: Any, after_image: Any) -> Any:
        ...


class TransitionInferenceError(RuntimeError):
    """Raised when transition inference cannot provide a parsed transition."""


class StateRegistry:
    """Deterministic in-memory registry of one canonical fingerprint per state."""

    def __init__(
        self,
        threshold: float = SAME_STATE_THRESHOLD,
        transition_inferencer: TransitionInferencer | None = None,
    ) -> None:
        self.threshold = threshold
        self.transition_inferencer = transition_inferencer
        self._states: dict[str, RegisteredState] = {}
        self._next_state_number = 1

    @property
    def states(self) -> Mapping[str, RegisteredState]:
        """Read-only view of the registered states in registration order."""
        return MappingProxyType(self._states)

    def __len__(self) -> int:
        return len(self._states)

    def resolve(self, observation: dict[str, Any]) -> StateResolution:
        """Build a fingerprint and resolve it against the registered states."""
        return self.resolve_fingerprint(build_state_fingerprint(observation))

    def resolve_fingerprint(self, fingerprint: StateFingerprint) -> StateResolution:
        """Resolve a pre-built fingerprint without rebuilding it."""
        best_state: RegisteredState | None = None
        best_score = -1.0

        for registered_state in self._states.values():
            match = match_state_fingerprints(
                fingerprint,
                registered_state.fingerprint,
                threshold=self.threshold,
            )
            # Strictly greater preserves the first registered state on ties.
            if match.state_score > best_score:
                best_state = registered_state
                best_score = match.state_score

        if best_state is not None and best_score >= self.threshold:
            return StateResolution(
                state_id=best_state.state_id,
                is_new=False,
                score=best_score,
            )

        state_id = f"S{self._next_state_number:03d}"
        self._next_state_number += 1
        self._states[state_id] = RegisteredState(state_id, fingerprint)
        return StateResolution(state_id=state_id, is_new=True, score=1.0)

    def resolve_transition(self, transition: dict[str, Any]) -> TransitionStateResolution:
        """Resolve before first and after second using this same registry."""
        before = self.resolve(transition["before"])
        after = self.resolve(transition["after"])
        return TransitionStateResolution(before=before, after=after)

    def resolve_images(self, before_image: Any, after_image: Any) -> TransitionStateResolution:
        """Infer and resolve a transition from the two images."""
        if self.transition_inferencer is None:
            raise TransitionInferenceError(
                "StateRegistry.resolve_images() requires a transition inferencer"
            )

        inference_result = self.transition_inferencer(before_image, after_image)
        if not isinstance(inference_result, Mapping):
            raise TransitionInferenceError("Transition inference unusable: invalid result")

        transition = inference_result.get("transition")
        parse_error = inference_result.get("parse_error")
        if not inference_result.get("usable") or transition is None:
            detail = f": {parse_error}" if parse_error else ""
            raise TransitionInferenceError(f"Transition inference unusable{detail}")

        return self.resolve_transition(transition)

    def reset(self) -> None:
        """Clear all states and restart runtime IDs at S001."""
        self._states.clear()
        self._next_state_number = 1


__all__ = [
    "RegisteredState",
    "TransitionInferenceError",
    "TransitionInferencer",
    "StateRegistry",
    "StateResolution",
    "TransitionStateResolution",
]
