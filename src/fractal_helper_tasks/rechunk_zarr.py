# Copyright 2024 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
"""Rechunk an existing Zarr."""

import logging
import os
import shutil
from typing import Any, Optional

import ngio
from ngio.core.utils import create_empty_ome_zarr_label
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def rechunk_zarr(
    *,
    zarr_url: str,
    chunk_sizes: Optional[dict[str, Optional[int]]] = None,
    suffix: str = "rechunked",
    rebuild_pyramids: bool = True,
    overwrite_input: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Drops singleton t dimension.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        chunk_sizes: Dictionary of chunk sizes to adapt. One can set any of
            the t, c, z, y, x axes that exist in the input image to be resized
            to a different chunk size. For example, {"y": 4000, "x": 4000}
            will set a new x & y chunking while maintaining the other chunk
            sizes. {"z": 10} will just change the Z chunking while keeping
            all other chunk sizes the same as the input.
        suffix: Suffix of the rechunked image.
        rebuild_pyramids: Whether pyramids are built fresh in the rechunked
            image. This has a small performance overhead, but ensures that
            this task is save against off-by-one issues when pyramid levels
            aren't easily downsampled by 2.
        overwrite_input: Whether the old image without rechunking should be
            overwritten (to avoid duplicating the data needed).
        overwrite: Whether to overwrite potential pre-existing output with the
            name zarr_url_suffix.
    """
    chunk_sizes = chunk_sizes or {}
    valid_axes = ["t", "c", "z", "y", "x"]
    for axis in valid_axes:
        if axis not in chunk_sizes:
            chunk_sizes[axis] = None

    rechunked_zarr_url = zarr_url + f"_{suffix}"
    ngff_image = ngio.NgffImage(zarr_url)
    pyramid_paths = ngff_image.levels_paths
    highest_res_img = ngff_image.get_image()
    axes_names = highest_res_img.dataset.on_disk_axes_names
    chunks = highest_res_img.on_disk_dask_array.chunks

    # Compute the chunksize tuple
    new_chunksize = [c[0] for c in chunks]
    # Overwrite chunk_size with user-set chunksize
    for i, axis in enumerate(axes_names):
        if axis in chunk_sizes:
            if chunk_sizes[axis] is not None:
                new_chunksize[i] = chunk_sizes[axis]

    # TODO: Check for extra axes specified

    new_ngff_image = ngff_image.derive_new_image(
        store=rechunked_zarr_url,
        name=ngff_image.image_meta.name,
        overwrite=overwrite,
        copy_labels=False,  # Copy if rechunk labels is not selected?
        copy_tables=True,
        chunks=new_chunksize,
    )

    ngff_image = ngio.NgffImage(zarr_url)

    if rebuild_pyramids:
        # Set the highest resolution, then consolidate to build a new pyramid
        new_ngff_image.get_image(highest_resolution=True).set_array(
            ngff_image.get_image(highest_resolution=True).on_disk_dask_array
        )
        new_ngff_image.get_image(highest_resolution=True).consolidate()
    else:
        for path in pyramid_paths:
            new_ngff_image.get_image(path=path).set_array(
                ngff_image.get_image(path=path).on_disk_dask_array
            )

    # Copy labels: Loop over them
    # Labels don't have a channel dimension
    chunk_sizes["c"] = None
    label_names = ngff_image.labels.list()
    for label in label_names:
        old_label = ngff_image.labels.get_label(name=label)
        label_level_paths = ngff_image.labels.levels_paths(name=label)
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
            new_ngff_image.labels.get_label(
                name=label, highest_resolution=True
            ).set_array(
                ngff_image.labels.get_label(
                    name=label, highest_resolution=True
                ).on_disk_dask_array
            )
            new_ngff_image.labels.get_label(
                name=label, highest_resolution=True
            ).consolidate()
        else:
            for label_path in label_level_paths:
                new_ngff_image.labels.get_label(name=label, path=label_path).set_array(
                    ngff_image.labels.get_label(
                        name=label, path=label_path
                    ).on_disk_dask_array
                )
    if overwrite_input:
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(rechunked_zarr_url, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
        return
    else:
        image_list_updates = dict(
            image_list_updates=[dict(zarr_url=rechunked_zarr_url, origin=zarr_url)]
        )
        return image_list_updates


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=rechunk_zarr,
        logger_name=logger.name,
    )
