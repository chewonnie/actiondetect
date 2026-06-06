"""
Sliding-window majority-vote stabilizer for a stream of integer class predictions.

Before the buffer is full (fewer than `window` items seen), the mode is taken
over whatever items are currently in the buffer (i.e., partial-window mode).
Ties in the mode are broken by recency: among tied labels, the one that
appears most recently in the deque wins.
"""

from collections import Counter, deque


class MajorityVoteSmoother:
    """Stabilize a stream of integer class labels with a sliding majority vote.

    Args:
        window: Number of recent predictions to keep. Must be >= 1.
    """

    def __init__(self, window: int) -> None:
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        self._buf: deque = deque(maxlen=window)

    def update(self, label: int | None) -> int | None:
        """Add a new label and return the current majority-vote result.

        Args:
            label: Integer class prediction, or None to skip this frame.

        Returns:
            The mode of the current buffer, with ties broken by recency
            (the candidate that appears latest in the buffer wins).
            Returns None if label is None or the buffer is empty.
        """
        if label is None:
            return None

        self._buf.append(label)

        # Count occurrences of each label in the buffer.
        counts = Counter(self._buf)
        max_count = max(counts.values())

        # Find all labels tied for the maximum count.
        tied = {lbl for lbl, cnt in counts.items() if cnt == max_count}

        # Break ties by recency: scan buffer right-to-left, first hit wins.
        for lbl in reversed(self._buf):
            if lbl in tied:
                return lbl

        # Unreachable, but satisfies type checker.
        return next(iter(tied))
