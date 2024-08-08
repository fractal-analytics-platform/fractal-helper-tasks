# Copyright 2024 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is derived from a Fractal task core task developed by
# Tommaso Comparin & Marco Franzon
"""Task to remove singelton T dimension from an OME-Zarr."""

import logging
from typing import Any

import dask.array as da
import zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from pydantic import validate_call

logger = logging.getLogger(__name__)


def get_attrs_without_t(zarr_url: str):
    """Generate zattrs without the t dimension.

    Args:
        zarr_url: Path to the zarr image
    """
    image_group = zarr.open_group(zarr_url)
    zattrs = image_group.attrs.asdict()
    # print(zattrs)
    for multiscale in zattrs["multiscales"]:
        # Update axes
        multiscale["axes"] = multiscale["axes"][1:]
        # Update coordinate Transforms
        for dataset in multiscale["datasets"]:
            for transform in dataset["coordinateTransformations"]:
                if transform["type"] == "scale":
                    transform["scale"] = transform["scale"][1:]
    return zattrs


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
        overwrite_input: Whether
    """
    # Normalize zarr_url
    zarr_url_old = zarr_url.rstrip("/")
    if overwrite_input:
        zarr_url_new = zarr_url_old
    else:
        zarr_url_new = f"{zarr_url_old}_{suffix}"

    logger.info(f"{zarr_url_old=}")
    logger.info(f"{zarr_url_new=}")

    # Read some parameters from metadata
    ngff_image = load_NgffImageMeta(zarr_url_old)

    # Check that T axis is the first axis:
    if not ngff_image.multiscale.axes[0].name == "t":
        logger.warning(
            f"The Zarr image {zarr_url_old} did not contain a T axis as its "
            f"first axis. The axes were: {ngff_image.multiscale.axes} \n"
            "The Drop T axis task is skipped"
        )
        return {}

    # Load 0-th level
    data_tczyx = da.from_zarr(zarr_url_old + "/0")
    # TODO: Check that T dimension is actually a singleton.
    new_data = data_tczyx[0, ...]

    if overwrite_input:
        image_list_update = dict(zarr_url=zarr_url_old, types=dict(has_t=False))
    else:
        # Generate attrs without the T dimension
        new_attrs = get_attrs_without_t(zarr_url_old)
        new_image_group = zarr.group(zarr_url_new)
        new_image_group.attrs.put(new_attrs)
        image_list_update = dict(
            zarr_url=zarr_url_new, origin=zarr_url_old, types=dict(has_t=False)
        )
        # TODO: Check if image contains labels & raise error (or even copy them)
        # FIXME: Check if image contains ROI tables & copy them

    # Write to disk (triggering execution)
    logger.debug(f"Writing Zarr without T dimension to {zarr_url_new}")
    new_data.to_zarr(
        f"{zarr_url_new}/0",
        overwrite=True,
        dimension_separator="/",
        write_empty_chunks=False,
    )
    logger.debug(f"Finished writing Zarr without T dimension to {zarr_url_new}")
    build_pyramid(
        zarrurl=zarr_url_new,
        overwrite=True,
        num_levels=ngff_image.num_levels,
        coarsening_xy=ngff_image.coarsening_xy,
        chunksize=new_data.chunksize,
    )

    return {"image_list_updates": [image_list_update]}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=drop_t_dimension,
        logger_name=logger.name,
    )
