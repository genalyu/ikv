"""Exact 25 Hz simulator <-> 30 Hz model / 10 Hz image clocks.

Training repeats each source row causally onto a 30 Hz grid. For predicted
right-edge action targets, invert that grid with CEIL, so round-tripping source
commands does not repeat the first command and skip another. RGB uses FLOOR,
matching the causal conversion. Keep these clocks across chunks.
"""
from fractions import Fraction
import numpy as np


def action_indices(native_start, native_count, model_hz=30, control_hz=25):
    if min(native_count,model_hz,control_hz)<1 or native_start<0:
        raise ValueError("Positive rates/count and nonnegative offset required")
    start=Fraction(native_start*control_hz,model_hz)
    end=Fraction((native_start+native_count)*control_hz,model_hz)
    if start.denominator!=1 or end.denominator!=1:
        raise ValueError("Chunk boundaries must align with the control clock")
    k=np.arange(int(start),int(end),dtype=np.int64)
    index=(k*model_hz+control_hz-1)//control_hz-native_start
    if (index<0).any() or (index>=native_count).any():
        raise ValueError("Chunk cannot cover the requested action times")
    return index


def observation_steps(control_start,control_count,control_hz=25,video_hz=10):
    """Observation steps in (start,start+count], excluding initial frame zero."""
    if min(control_count,control_hz,video_hz)<1 or control_start<0:
        raise ValueError("Invalid observation clock")
    stop=control_start+control_count
    result=[]
    for v in range(1,((stop+1)*video_hz+control_hz-1)//control_hz+1):
        step=v*control_hz//video_hz
        if control_start<step<=stop:result.append(step)
    return result
