"""Stable observation provenance shared by scorers and native KV adapters."""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Region:
    frame_id: str
    sensor: int
    box: tuple[float, float, float, float] = (0., 0., 1., 1.)
    weight: float = 1.

    def __post_init__(self):
        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("Region frame_id must be a nonempty string")
        if type(self.sensor) is not int or self.sensor < 0:
            raise ValueError("Region sensor must be a nonnegative integer")
        box = tuple(float(x) for x in self.box)
        if len(box) != 4 or not all(math.isfinite(x) for x in box):
            raise ValueError("Region needs a finite y0,x0,y1,x1 box")
        y0, x0, y1, x1 = box
        if not (0 <= y0 < y1 <= 1 and 0 <= x0 < x1 <= 1):
            raise ValueError("Region coordinates must describe a nonempty normalized box")
        if not math.isfinite(self.weight) or self.weight <= 0:
            raise ValueError("Region weight must be positive and finite")
        object.__setattr__(self, "box", box)

@dataclass(frozen=True)
class Candidate:
    uid: str
    time: float
    regions: tuple[Region, ...]
    provenance: str = ""

    def __post_init__(self):
        if not isinstance(self.uid, str) or not self.uid:
            raise ValueError("Candidate uid must be a nonempty string")
        if not math.isfinite(self.time) or self.time < 0:
            raise ValueError("Candidate time must be finite and nonnegative")
        regions = tuple(self.regions)
        if not regions or any(not isinstance(r, Region) for r in regions):
            raise ValueError("Every native KV needs explicit observation regions")
        if not isinstance(self.provenance, str):
            raise ValueError("Candidate provenance must be a string")
        object.__setattr__(self, "regions", regions)
