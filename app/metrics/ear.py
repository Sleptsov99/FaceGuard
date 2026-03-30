"""Eye Aspect Ratio calculation and eye state classification."""

from enum import Enum
import numpy as np


# Default EAR thresholds (Soukupova & Cech, tuned for MediaPipe landmarks)
EAR_OPEN_THRESHOLD   = 0.25
EAR_CLOSED_THRESHOLD = 0.15


class EyeState(Enum):
    OPEN             = "open"
    PARTIALLY_CLOSED = "partially_closed"
    CLOSED           = "closed"

    @classmethod
    def from_ear(
        cls,
        ear: float,
        open_thresh: float   = EAR_OPEN_THRESHOLD,
        closed_thresh: float = EAR_CLOSED_THRESHOLD,
    ) -> "EyeState":
        if ear >= open_thresh:
            return cls.OPEN
        if ear >= closed_thresh:
            return cls.PARTIALLY_CLOSED
        return cls.CLOSED


def compute_ear(points: np.ndarray) -> float:
    """
    Eye Aspect Ratio from 6 landmark points (shape (6, 2)).

    Point order: [outer_corner, upper1, upper2, inner_corner, lower1, lower2]

    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    """
    p1, p2, p3, p4, p5, p6 = points
    vertical_a = np.linalg.norm(p2 - p6)
    vertical_b = np.linalg.norm(p3 - p5)
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return 0.0
    return float((vertical_a + vertical_b) / (2.0 * horizontal))
