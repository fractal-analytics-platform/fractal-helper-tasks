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
import ngio.images
import ngio.images.label
from ngio.ome_zarr_meta import AxesMapper
from pydantic import validate_call

from fractal_helper_tasks.utils import normalize_chunk_size_dict

logger = logging.getLogger(__name__)


def change_chunks(
    initial_chunks: list[int],
    axes_mapper: AxesMapper,
    chunk_sizes: dict[str, Optional[int]],
) -> list[int]:
    """Create a new chunk_size list with rechunking.

    Based on the initial chunks, the axes_mapper of the OME-Zarr & the
    chunk_sizes dictionary with new chunk sizes, create a new chunk_size list.

    """
    for axes_name, chunk_value in chunk_sizes.items():
        if chunk_value is not None:
            axes_index = axes_mapper.get_index(axes_name)
            if axes_index is None:
                raise ValueError(
                    f"Rechunking with {axes_name=} is specified, but the "
                    "OME-Zarr only has the following axes: "
                    f"{axes_mapper.on_disk_axes_names}"
                )
            initial_chunks[axes_index] = chunk_value
    return initial_chunks


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
    ome_zarr_container = ngio.open_ome_zarr_container(zarr_url)
    pyramid_paths = ome_zarr_container.levels_paths
    highest_res_img = ome_zarr_container.get_image()
    chunks = highest_res_img.chunks
    new_chunksize = change_chunks(
        initial_chunks=list(chunks),
        axes_mapper=highest_res_img.meta.axes_mapper,
        chunk_sizes=chunk_sizes,
    )

    logger.info(f"Chunk sizes after rechunking will be: {new_chunksize=}")

    new_ome_zarr_container = ome_zarr_container.derive_image(
        store=rechunked_zarr_url,
        name=ome_zarr_container.image_meta.name,
        overwrite=overwrite,
        copy_labels=not rechunk_labels,
        copy_tables=True,
        chunks=new_chunksize,
    )

    if rebuild_pyramids:
        # Set the highest resolution, then consolidate to build a new pyramid
        new_image = new_ome_zarr_container.get_image()
        new_image.set_array(ome_zarr_container.get_image().get_array(mode="dask"))
        new_image.consolidate()
    else:
        for path in pyramid_paths:
            new_ome_zarr_container.get_image(path=path).set_array(
                ome_zarr_container.get_image(path=path).get_array(mode="dask")
            )

    # Rechunk labels
    if rechunk_labels:
        chunk_sizes["c"] = None
        label_names = ome_zarr_container.list_labels()
        for label in label_names:
            old_label = ome_zarr_container.get_label(name=label)
            new_chunksize = change_chunks(
                initial_chunks=list(old_label.chunks),
                axes_mapper=old_label.meta.axes_mapper,
                chunk_sizes=chunk_sizes,
            )
            new_label = new_ome_zarr_container.derive_label(
                name=label,
                ref_image=old_label,
                chunks=new_chunksize,
                overwrite=overwrite,
            )
            if rebuild_pyramids:
                new_label.set_array(old_label.get_array(mode="dask"))
                new_label.consolidate()
            else:
                label_pyramid_paths = old_label.meta.paths
                for path in label_pyramid_paths:
                    old_label = ome_zarr_container.get_label(name=label, path=path)
                    new_label.set_array(
                        old_label.get_array(mode="dask")
                    )

    if overwrite_input:
        os.rename(zarr_url, f"{zarr_url}_tmp")
        os.rename(rechunked_zarr_url, zarr_url)
        shutil.rmtree(f"{zarr_url}_tmp")
        return
    else:
        # FIXME: Update well metadata to add the new image if the image is in
        # a well
        output = dict(
            image_list_updates=[
                dict(
                    zarr_url=rechunked_zarr_url,
                    origin=zarr_url,
                    types=dict(rechunked=True),
                )
            ],
        )
        return output


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=rechunk_zarr,
        logger_name=logger.name,
    )
