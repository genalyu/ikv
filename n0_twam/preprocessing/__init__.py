"""Preprocessing helpers for sparse RGB-motion inputs."""

from .dinov2 import DinoPatchOutput, FrozenDinoV2PatchEncoder
from .rgb_motion_sequence import RGBDCameraSequence, RGBMotionSequencePreprocessor

__all__ = [
    "DinoPatchOutput",
    "FrozenDinoV2PatchEncoder",
    "RGBDCameraSequence",
    "RGBMotionSequencePreprocessor",
]
