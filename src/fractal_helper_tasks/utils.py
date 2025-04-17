# Copyright 2025 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
"""Utils for helper tasks."""

from typing import Optional


def normalize_chunk_size_dict(chunk_sizes: dict[str, Optional[int]]):
    """Converts all chunk_size axes names to lower case and assert validity.

    Args:
        chunk_sizes: Dictionary of chunk sizes that should be adapted. Can
            contain new chunk sizes for t, c, z, y & x.

    Returns:
        chunk_sizes_norm: Normalized chunk_sizes dict.
    """
    chunk_sizes = chunk_sizes or {}
    chunk_sizes_norm = {}
    for key, value in chunk_sizes.items():
        chunk_sizes_norm[key.lower()] = value

    valid_axes = ["t", "c", "z", "y", "x"]
    for axis in chunk_sizes_norm:
        if axis not in valid_axes:
            raise ValueError(
                f"Axis {axis} is not supported. Valid axes choices are "
                f"{valid_axes}."
            )
    return chunk_sizes_norm
