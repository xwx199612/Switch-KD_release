"""Stopping criteria for pathological repeated parsing output."""

from __future__ import annotations

from typing import Any


class RepeatedTokenBlockStoppingCriteria:
    """Stop after three consecutive copies of a meaningful token block."""

    MIN_BLOCK_LENGTH = 16
    MAX_BLOCK_LENGTH = 64
    REPEAT_COUNT = 3
    RECENT_TOKEN_WINDOW = 256

    def __init__(self) -> None:
        self.prompt_length: int | None = None
        self.triggered = False
        self.block_length: int | None = None

    def __call__(self, input_ids: Any, scores: Any = None, **kwargs: Any) -> bool:
        del scores, kwargs
        if self.prompt_length is None:
            raise RuntimeError("prompt_length must be set before repetition stopping")

        for sequence in input_ids:
            generated = sequence[self.prompt_length :].tolist()
            recent = generated[-self.RECENT_TOKEN_WINDOW :]
            max_block_length = min(
                self.MAX_BLOCK_LENGTH,
                len(recent) // self.REPEAT_COUNT,
            )
            for block_length in range(max_block_length, self.MIN_BLOCK_LENGTH - 1, -1):
                end = len(recent)
                first_start = end - self.REPEAT_COUNT * block_length
                if first_start < 0:
                    continue
                block = recent[first_start : first_start + block_length]
                if all(
                    recent[first_start + offset * block_length :
                          first_start + (offset + 1) * block_length] == block
                    for offset in range(1, self.REPEAT_COUNT)
                ):
                    self.triggered = True
                    self.block_length = block_length
                    return True
        return False


__all__ = ["RepeatedTokenBlockStoppingCriteria"]
