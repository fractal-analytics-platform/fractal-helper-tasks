# Copyright 2024 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is derived from a Fractal task core task developed by
# Tommaso Comparin & Marco Franzon
"""Task to remove singleton T dimension from an OME-Zarr."""

import logging
import os
import shutil
from typing import Any

import dask.array as da
import ngio
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def drop_t_dimension(
    *,
    zarr_url: str,
    suffix: str = "no_T",
    overwrite_input: bool = False,
) -> dict[str, Any]:
    """Drops singleton t dimension.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        suffix: Suffix to be used for the new Zarr image. If overwrite_input
            is True, this file is only temporary.
        overwrite_input: Whether the existing iamge should be overwritten with
            the new OME-Zarr without the T dimension.
    """
    # Normalize zarr_url
    zarr_url_old = zarr_url.rstrip("/")
    zarr_url_new = f"{zarr_url_old}_{suffix}"

    logger.info(f"{zarr_url_old=}")
    logger.info(f"{zarr_url_new=}")

    old_ome_zarr = ngio.open_ome_zarr_container(zarr_url_old)
    old_ome_zarr_img = old_ome_zarr.get_image()
    if not old_ome_zarr_img.has_axis("t"):
        raise ValueError(
            f"The Zarr image {zarr_url_old} does not contain a T axis. "
            "Thus, the drop T dimension task can't be applied to it."
        )
    # TODO: Check if T dimension not singleton
    image = old_ome_zarr_img.get_array(mode="dask")
    t_index = old_ome_zarr_img.axes_handler.get_index("t")
    new_img = da.squeeze(image, axis=t_index)
    pixel_size = old_ome_zarr_img.pixel_size
    new_pixel_size = ngio.PixelSize(x=pixel_size.x, y=pixel_size.y, z=pixel_size.z)
    axes_names = list(old_ome_zarr_img.axes)
    del axes_names[t_index]
    chunk_sizes = old_ome_zarr_img.chunks
    new_chunk_sizes = chunk_sizes[:t_index] + chunk_sizes[t_index + 1 :]
    new_ome_zarr_container = old_ome_zarr.derive_image(
        store=zarr_url_new,
        shape=new_img.shape,
        chunks=new_chunk_sizes,
        dtype=old_ome_zarr_img.dtype,
        pixel_size=new_pixel_size,
        axes_names=axes_names,
        copy_tables=True,
    )
    new_image_container = new_ome_zarr_container.get_image()
    new_image_container.set_array(new_img)
    new_image_container.consolidate()

    if overwrite_input:
        image_list_update = dict(zarr_url=zarr_url_old, types=dict(has_t=False))
        os.rename(zarr_url_old, f"{zarr_url_old}_tmp")
        os.rename(zarr_url_new, zarr_url_old)
        shutil.rmtree(f"{zarr_url}_tmp")
    else:
        image_list_update = dict(
            zarr_url=zarr_url_new, origin=zarr_url_old, types=dict(has_t=False)
        )

    return {"image_list_updates": [image_list_update]}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=drop_t_dimension,
        logger_name=logger.name,
    )
