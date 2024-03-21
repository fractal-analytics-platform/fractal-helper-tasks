# Copyright 2024 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
#
# This file is derived from a Fractal task core task developed by
# Tommaso Comparin & Marco Franzon
"""Task to remove singelton T dimension from an OME-Zarr."""
import logging
from pathlib import Path
from typing import Any, Sequence

import dask.array as da
import zarr
from fractal_tasks_core.ngff import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from pydantic.decorator import validate_arguments

logger = logging.getLogger(__name__)


def get_attrs_without_t(zarr_url: str):
    """
    Generate zattrs without the t dimension.

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


@validate_arguments
def drop_t_dimension(
    *,
    input_paths: Sequence[str],
    output_path: str,
    component: str,
    metadata: dict[str, Any],
    suffix: str = "no_T",
    # overwrite_input: bool = False,
) -> dict[str, Any]:
    """
    Drops singleton t dimension.

    Args:
        input_paths: This parameter is not used by this task.
            This task only supports a single input path.
            (standard argument for Fractal tasks, managed by Fractal server).
        output_path: Path were the output of this task is stored.
            Example: `"/some/path/"` => puts the new OME-Zarr file in that
            folder.
            (standard argument for Fractal tasks, managed by Fractal server).
        component: Path to the OME-Zarr image in the OME-Zarr plate that
            is processed. Component is typically changed by the `copy_ome_zarr`
            task before, to point to a new mip Zarr file.
            Example: `"some_plate_mip.zarr/B/03/0"`.
            (standard argument for Fractal tasks, managed by Fractal server).
        metadata: Dictionary containing metadata about the OME-Zarr.
            This task requires the key `copy_ome_zarr` to be present in the
            metadata (as defined in `copy_ome_zarr` task).
            (standard argument for Fractal tasks, managed by Fractal server).
        suffix: Suffix to be used for the new Zarr image. If overwrite_input
            is True, this file is only temporary.
    """
    # Preliminary checks
    if len(input_paths) > 1:
        raise NotImplementedError

    zarrurl_old = (Path(input_paths[0]).resolve() / component).as_posix()
    new_component = component + "_" + suffix
    zarrurl_new = (Path(input_paths[0]).resolve() / new_component).as_posix()
    logger.info(f"{zarrurl_old=}")
    logger.info(f"{zarrurl_new=}")

    # Read some parameters from metadata
    ngff_image = load_NgffImageMeta(zarrurl_old)

    # Check that T axis is the first axis:
    if not ngff_image.multiscale.axes[0].name == "t":
        logger.warning(
            f"The Zarr image {zarrurl_old} did not contain a T axis as its "
            f"first axis. The axes were: {ngff_image.multiscale.axes} \n"
            "The Drop T axis task is skipped"
        )
        return {}

    # TODO: Generate attrs without the T dimension
    new_attrs = get_attrs_without_t(zarrurl_old)
    new_image_group = zarr.group(zarrurl_new)
    new_image_group.attrs.put(new_attrs)

    # TODO: Check if image contains labels & raise error (or even copy them)
    # FIXME: Check if image contains ROI tables & copy them

    # Load 0-th level
    data_tczyx = da.from_zarr(zarrurl_old + "/0")
    new_data = data_tczyx[0, ...]
    # Write to disk (triggering execution)
    logger.debug(f"Writing Zarr without T dimension to {zarrurl_new}")
    new_data.to_zarr(
        f"{zarrurl_new}/0",
        overwrite=True,
        dimension_separator="/",
        write_empty_chunks=False,
    )
    logger.debug(f"Finished writing Zarr without T dimension to {zarrurl_new}")
    build_pyramid(
        zarrurl=zarrurl_new,
        overwrite=True,
        num_levels=ngff_image.num_levels,
        coarsening_xy=ngff_image.coarsening_xy,
        chunksize=new_data.chunksize,
    )
    return {}


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=drop_t_dimension,
        logger_name=logger.name,
    )
