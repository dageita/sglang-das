"""Model-independent bucket selection for local, row-independent segments.

This module deliberately has no torch or model imports. All sizes refer to
local hidden-state rows, not request prompt lengths or global TP token counts.
"""

from bisect import bisect_left
from dataclasses import dataclass


@dataclass(frozen=True)
class BucketPolicy:
    buckets: tuple
    max_padding_rows: int = 64
    max_padding_ratio: float = 0.03
    min_rows: int = 8

    def __post_init__(self):
        if not self.buckets or tuple(sorted(set(self.buckets))) != self.buckets:
            raise ValueError("Graph buckets must be positive, sorted and unique")
        if self.buckets[0] <= 0 or self.min_rows <= 0:
            raise ValueError("Graph row sizes must be positive")
        if self.max_padding_rows < 0 or not 0 <= self.max_padding_ratio <= 1:
            raise ValueError("Invalid graph padding bounds")

    def select(self, rows):
        if rows < self.min_rows or rows > self.buckets[-1]:
            return None, "outside_range"
        index = bisect_left(self.buckets, rows)
        bucket = self.buckets[index]
        padding = bucket - rows
        if padding > self.max_padding_rows or padding > rows * self.max_padding_ratio:
            return None, "padding_limit"
        return bucket, None
