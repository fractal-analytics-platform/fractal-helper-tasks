# Copyright 2025 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
"""Compute task for copying labels across multiplexing acquisitions (HCS)."""

import logging
from typing import Any

import ngio
from pydantic import validate_call

logger = logging.getLogger("copy_labels_multiplexing")


@validate_call
def copy_labels_multiplexing(
    *,
    zarr_url: str,
    init_args: dict[str, Any],
    overwrite: bool = True,
) -> None:
    """Copy labels from a reference acquisition to this acquisition.

    For each requested label, derives an empty label image in the target
    OME-Zarr using the reference label as a template, then copies the data
    at every pyramid level using dask (memory-efficient for large labels).

    Args:
        zarr_url: Path or url to the target OME-Zarr image.
            (standard argument for Fractal compound task compute functions).
        init_args: Arguments produced by the init task, containing
            ``reference_zarr_url`` and ``label_names``.
        overwrite: Whether to overwrite an existing label in the target
            image. Defaults to True so that re-runs succeed without
            manual cleanup.
    """
    reference_zarr_url: str = init_args["reference_zarr_url"]
    label_names: list[str] = init_args["label_names"]

    logger.info(
        f"Running `copy_labels_to_multiplexing_acquisitions` on '{zarr_url}'. "
        f"Reference: '{reference_zarr_url}'. Labels: {label_names}."
    )

    ref_container = ngio.open_ome_zarr_container(reference_zarr_url)
    target_container = ngio.open_ome_zarr_container(zarr_url)

    for label_name in label_names:
        logger.info(f"Copying label '{label_name}'.")

        ref_label = ref_container.get_label(label_name)

        # Create empty label pyramid in target, using reference as template
        target_container.derive_label(
            name=label_name,
            ref_image=ref_label,
            overwrite=overwrite,
        )

        # Copy data level by level via dask
        for level_path in ref_label.meta.paths:
            src = ref_container.get_label(label_name, path=level_path)
            tgt = target_container.get_label(label_name, path=level_path)
            tgt.set_array(src.get_array(mode="dask"))
            logger.info(f"  Level {level_path}: copied shape {src.shape}.")

    logger.info(f"Finished copying {len(label_names)} label(s) to '{zarr_url}'.")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=copy_labels_multiplexing,
        logger_name=logger.name,
    )
