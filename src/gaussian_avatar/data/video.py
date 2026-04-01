"""Video frame extraction utilities.

This module provides functions for extracting frames from video files
in the PeopleSnapshot dataset, with support for lazy loading and disk caching.
"""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np

def extract_frame(video_path: Union[str, Path], frame_idx: int) -> np.ndarray:
    """Extract a single frame from a video file.

    Uses OpenCV VideoCapture to seek to and read a specific frame.
    The frame is converted from BGR (OpenCV default) to RGB.

    Args:
        video_path: Path to the .mp4 video file
        frame_idx: 0-indexed frame number to extract

    Returns:
        RGB image as (H, W, 3) uint8 array

    Raises:
        FileNotFoundError: If video_path does not exist
        IndexError: If frame_idx is negative or beyond video length
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if frame_idx < 0:
        raise IndexError(f"Frame index must be non-negative, got {frame_idx}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_idx >= total_frames:
        cap.release()
        raise IndexError(
            f"Frame index {frame_idx} out of range for video with {total_frames} frames"
        )

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")

    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def get_video_frame_count(video_path: Union[str, Path]) -> int:
    """Get the total number of frames in a video file.

    Args:
        video_path: Path to the .mp4 video file

    Returns:
        Total number of frames in the video

    Raises:
        FileNotFoundError: If video_path does not exist
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    return count

class FrameCache:
    """Manages caching of extracted video frames to disk.

    Frames are stored as PNG files in a directory structure:
        <cache_dir>/<subject>/frame_NNNNNN.png

    The cache supports storing and retrieving frames by subject name
    and frame index, preserving RGB channel order.
    """

    def __init__(self, cache_dir: Union[str, Path]) -> None:
        """Initialize frame cache.

        Args:
            cache_dir: Root directory for storing cached frames
        """
        self.cache_dir = Path(cache_dir)

    def _get_frame_path(self, subject: str, frame_idx: int) -> Path:
        """Get the file path for a cached frame.

        Args:
            subject: Subject identifier (e.g., "male-3-casual")
            frame_idx: Frame index number

        Returns:
            Path to the cached PNG file
        """
        return self.cache_dir / subject / f"frame_{frame_idx:06d}.png"

    def exists(self, subject: str, frame_idx: int) -> bool:
        """Check if a frame is cached.

        Args:
            subject: Subject identifier
            frame_idx: Frame index number

        Returns:
            True if frame is cached, False otherwise
        """
        return self._get_frame_path(subject, frame_idx).exists()

    def get(self, subject: str, frame_idx: int) -> Optional[np.ndarray]:
        """Get a cached frame.

        Args:
            subject: Subject identifier
            frame_idx: Frame index number

        Returns:
            RGB image as (H, W, 3) uint8 array, or None if not cached
        """
        path = self._get_frame_path(subject, frame_idx)
        if not path.exists():
            return None

        # Read as BGR, convert to RGB
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            return None

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def put(
        self,
        subject: str,
        frame_idx: int,
        image: np.ndarray,
        skip_existing: bool = True,
    ) -> None:
        """Cache a frame to disk.

        Creates the subject subdirectory if it doesn't exist.
        When skip_existing is True (default), skips writing if the file
        already exists on disk. This prevents race conditions when multiple
        DataLoader workers attempt to cache the same frame concurrently.

        Args:
            subject: Subject identifier
            frame_idx: Frame index number
            image: RGB image as (H, W, 3) uint8 array
            skip_existing: If True, skip writing if the file already exists
        """
        path = self._get_frame_path(subject, frame_idx)

        if skip_existing and path.exists():
            return

        path.parent.mkdir(parents=True, exist_ok=True)

        # Convert RGB to BGR for OpenCV imwrite
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr_image)

def cache_all_frames(
    video_path: Union[str, Path],
    cache_dir: Union[str, Path],
    subject: str,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    image_size: Optional[Tuple[int, int]] = None,
    skip_existing: bool = True,
) -> None:
    """Extract frames from a video and cache them to disk.

    Extracts a range of frames from a video file and saves them as PNG files
    in the cache directory. Optionally resizes frames during extraction.

    Args:
        video_path: Path to the .mp4 video file
        cache_dir: Root directory for storing cached frames
        subject: Subject identifier for organizing cached files
        start_frame: First frame index to extract (inclusive)
        end_frame: Last frame index to extract (exclusive), or None for all frames
        image_size: Optional (height, width) to resize frames, or None for original size
        skip_existing: If True, skip frames that are already cached

    Raises:
        FileNotFoundError: If video_path does not exist
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cache = FrameCache(cache_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if end_frame is None:
        end_frame = total_frames

    start_frame = max(0, start_frame)
    end_frame = min(end_frame, total_frames)

    for frame_idx in range(start_frame, end_frame):
        if skip_existing and cache.exists(subject, frame_idx):
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if image_size is not None:
            h, w = image_size
            frame = cv2.resize(frame, (w, h))

        cache.put(subject, frame_idx, frame)

    cap.release()
