# Copyright 2025 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
"""Utils for helper tasks."""

from typing import Optional

import ngio
from ngio.core.utils import create_empty_ome_zarr_label


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


def rechunk_label(
    orig_ngff_image: ngio.NgffImage,
    new_ngff_image: ngio.NgffImage,
    label: str,
    chunk_sizes: list[int],
    overwrite: bool = False,
    rebuild_pyramids: bool = True,
):
    """Saves a rechunked label image into a new OME-Zarr

    The label image is based on an existing label image in another OME-Zarr.

    Args:
        orig_ngff_image: Original OME-Zarr that contains the label image
        new_ngff_image: OME-Zarr to which the rechunked label image should be
            added.
        label: Name of the label image.
        chunk_sizes: New chunk sizes that should be applied
        overwrite: Whether the label image in `new_ngff_image` should be
            overwritten if it already exists.
        rebuild_pyramids: Whether pyramids are built fresh in the rechunked
            label image. This has a small performance overhead, but ensures
            that this task is save against off-by-one issues when pyramid
            levels aren't easily downsampled by 2.
    """
    old_label = orig_ngff_image.labels.get_label(name=label)
    label_level_paths = orig_ngff_image.labels.levels_paths(name=label)
    # Compute the chunksize tuple
    chunks = old_label.on_disk_dask_array.chunks
    new_chunksize = [c[0] for c in chunks]
    # Overwrite chunk_size with user-set chunksize
    for i, axis in enumerate(old_label.dataset.on_disk_axes_names):
        if axis in chunk_sizes:
            if chunk_sizes[axis] is not None:
                new_chunksize[i] = chunk_sizes[axis]
    create_empty_ome_zarr_label(
        store=new_ngff_image.store
        + "/"
        + "labels"
        + "/"
        + label,  # FIXME: Set this better?
        on_disk_shape=old_label.on_disk_shape,
        chunks=new_chunksize,
        dtype=old_label.on_disk_dask_array.dtype,
        on_disk_axis=old_label.dataset.on_disk_axes_names,
        pixel_sizes=old_label.dataset.pixel_size,
        xy_scaling_factor=old_label.metadata.xy_scaling_factor,
        z_scaling_factor=old_label.metadata.z_scaling_factor,
        time_spacing=old_label.dataset.time_spacing,
        time_units=old_label.dataset.time_axis_unit,
        levels=label_level_paths,
        name=label,
        overwrite=overwrite,
        version=old_label.metadata.version,
    )

    # Fill in labels .attrs to contain the label name
    list_of_labels = new_ngff_image.labels.list()
    if label not in list_of_labels:
        new_ngff_image.labels._label_group.attrs["labels"] = [
            *list_of_labels,
            label,
        ]

    if rebuild_pyramids:
        # Set the highest resolution, then consolidate to build a new pyramid
        new_ngff_image.labels.get_label(name=label, highest_resolution=True).set_array(
            orig_ngff_image.labels.get_label(
                name=label, highest_resolution=True
            ).on_disk_dask_array
        )
        new_ngff_image.labels.get_label(
            name=label, highest_resolution=True
        ).consolidate()
    else:
        for label_path in label_level_paths:
            new_ngff_image.labels.get_label(name=label, path=label_path).set_array(
                orig_ngff_image.labels.get_label(
                    name=label, path=label_path
                ).on_disk_dask_array
            )
