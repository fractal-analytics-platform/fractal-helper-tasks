# Copyright 2025 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
"""Rechunk an existing Zarr."""

import logging
import os
import shutil
from typing import Any, Optional

import ngio
from pydantic import validate_call

from fractal_helper_tasks.utils import normalize_chunk_size_dict, rechunk_label

logger = logging.getLogger(__name__)


@validate_call
def rechunk_zarr(
    *,
    zarr_url: str,
    chunk_sizes: Optional[dict[str, Optional[int]]] = None,
    suffix: str = "rechunked",
    rechunk_labels: bool = True,
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
        rechunk_labels: Whether to apply the same rechunking to all label
            images of the OME-Zarr as well.
        rebuild_pyramids: Whether pyramids are built fresh in the rechunked
            image. This has a small performance overhead, but ensures that
            this task is save against off-by-one issues when pyramid levels
            aren't easily downsampled by 2.
        overwrite_input: Whether the old image without rechunking should be
            overwritten (to avoid duplicating the data needed).
        overwrite: Whether to overwrite potential pre-existing output with the
            name zarr_url_suffix.
    """
    logger.info(f"Running `rechunk_zarr` on {zarr_url=} with {chunk_sizes=}.")

    chunk_sizes = normalize_chunk_size_dict(chunk_sizes)

    rechunked_zarr_url = zarr_url + f"_{suffix}"
    ngff_image = ngio.NgffImage(zarr_url)
    pyramid_paths = ngff_image.levels_paths
    highest_res_img = ngff_image.get_image()
    axes_names = highest_res_img.dataset.on_disk_axes_names
    chunks = highest_res_img.on_disk_dask_array.chunks

    # Compute the chunksize tuple
    new_chunksize = [c[0] for c in chunks]
    logger.info(f"Initial chunk sizes were: {chunks}")
    # Overwrite chunk_size with user-set chunksize
    for i, axis in enumerate(axes_names):
        if axis in chunk_sizes:
            if chunk_sizes[axis] is not None:
                new_chunksize[i] = chunk_sizes[axis]

    for axis in chunk_sizes:
        if axis not in axes_names:
            raise NotImplementedError(
                f"Rechunking with {axis=} is specified, but the OME-Zarr only "
                f"has the following axes: {axes_names}"
            )

    logger.info(f"Chunk sizes after rechunking will be: {new_chunksize=}")

    new_ngff_image = ngff_image.derive_new_image(
        store=rechunked_zarr_url,
        name=ngff_image.image_meta.name,
        overwrite=overwrite,
        copy_labels=not rechunk_labels,
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

    # Copy labels
    if rechunk_labels:
        chunk_sizes["c"] = None
        label_names = ngff_image.labels.list()
        for label in label_names:
            rechunk_label(
                orig_ngff_image=ngff_image,
                new_ngff_image=new_ngff_image,
                label=label,
                chunk_sizes=chunk_sizes,
                overwrite=overwrite,
                rebuild_pyramids=rebuild_pyramids,
            )

    if overwrite_input:
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(rechunked_zarr_url, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
        return
    else:
        output = dict(
            image_list_updates=[
                dict(
                    zarr_url=rechunked_zarr_url,
                    origin=zarr_url,
                    types=dict(rechunked=True),
                )
            ],
            filters=dict(types=dict(rechunked=True)),
        )
        return output


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=rechunk_zarr,
        logger_name=logger.name,
    )
