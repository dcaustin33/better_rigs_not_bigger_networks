"""Camera representation for Gaussian splatting rendering.

This module provides the Camera dataclass used for gsplat rendering,
storing camera intrinsics and extrinsics in a format compatible with
the gsplat rasterization API.
"""

from dataclasses import dataclass

from torch import Tensor


@dataclass
class Camera:
    """Camera representation for Gaussian splatting rendering.

    Stores camera parameters and provides convenient property accessors
    for gsplat rendering operations.

    Args:
        intrinsic: (3, 3) intrinsic matrix [fx, 0, cx; 0, fy, cy; 0, 0, 1]
        extrinsic: (4, 4) world-to-camera transform
        width: Image width in pixels
        height: Image height in pixels
    """

    intrinsic: Tensor
    extrinsic: Tensor
    width: int
    height: int

    @property
    def fx(self) -> float:
        """Focal length x."""
        return self.intrinsic[0, 0].item()

    @property
    def fy(self) -> float:
        """Focal length y."""
        return self.intrinsic[1, 1].item()

    @property
    def cx(self) -> float:
        """Principal point x."""
        return self.intrinsic[0, 2].item()

    @property
    def cy(self) -> float:
        """Principal point y."""
        return self.intrinsic[1, 2].item()

    @property
    def viewmat(self) -> Tensor:
        """World-to-camera matrix (4, 4)."""
        return self.extrinsic

    @classmethod
    def from_dataset_sample(
        cls,
        intrinsic: Tensor,
        extrinsic: Tensor,
        height: int,
        width: int,
    ) -> "Camera":
        """Create Camera from dataset sample tensors.

        Factory method for creating a Camera from the tensors returned
        by PeopleSnapshotDataset.__getitem__().

        Args:
            intrinsic: (3, 3) camera intrinsic matrix
            extrinsic: (4, 4) world-to-camera transform
            height: Image height in pixels
            width: Image width in pixels

        Returns:
            Camera instance with the provided parameters
        """
        return cls(
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            width=width,
            height=height,
        )
