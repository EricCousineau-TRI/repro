"""
Provides simple containers for images that come from cameras.
"""
import dataclasses as dc

import numpy as np

from .array import (
    DepthArray,
    DoubleArray,
    ImageArray,
    IntrinsicsArray,
    LabelArray,
    RgbArray,
)
from .generic import Dict


class RigidTransform:
    # Fake version of `pydrake.math.RigidTransform_[float]`.
    def __init__(self, p: DoubleArray[3]):
        self._p = p

    def GetAsMatrix4(self) -> DoubleArray[(4, 4), :]:
        X = np.ones((4, 4))
        X[:3, 3] = self._p
        return X


@dc.dataclass
class CameraImage:
    array: ImageArray
    K: IntrinsicsArray
    X_TC: RigidTransform


@dc.dataclass
class CameraRgbImage(CameraImage):
    array: RgbArray


@dc.dataclass
class CameraDepthImage(CameraImage):
    array: DepthArray


@dc.dataclass
class CameraLabelImage(CameraImage):
    array: LabelArray


@dc.dataclass
class CameraImageSet(CameraImage):
    rgb: CameraRgbImage
    depth: CameraDepthImage
    label: CameraLabelImage


# Camera Id -> Camera images
# TODO(eric.cousineau): Make this functional, using `typish`.
CameraImageSetMap: Dict[str, CameraImageSet] = dict
