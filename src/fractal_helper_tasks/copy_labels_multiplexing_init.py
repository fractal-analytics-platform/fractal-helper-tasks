# Copyright 2025 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
"""Init task for copying labels across multiplexing acquisitions (HCS)."""

import logging
import os.path
import posixpath
from typing import Any

import ngio
from pydantic import validate_call

logger = logging.getLogger("copy_labels_multiplexing_init")


def _url_dirname(url: str) -> str:
    r"""Return the parent path of a URL or local filesystem path.

    Uses ``posixpath`` for cloud URLs (which always use ``/`` as separator)
    and ``os.path`` for local paths (which uses the OS-native separator,
    handling ``\\`` on Windows).
    """
    url = url.rstrip("/")
    return posixpath.dirname(url) if "://" in url else os.path.dirname(url)


def _url_basename(url: str) -> str:
    r"""Return the final component of a URL or local filesystem path.

    Uses ``posixpath`` for cloud URLs (which always use ``/`` as separator)
    and ``os.path`` for local paths (which uses the OS-native separator,
    handling ``\\`` on Windows).
    """
    url = url.rstrip("/")
    return posixpath.basename(url) if "://" in url else os.path.basename(url)


@validate_call
def copy_labels_multiplexing_init(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    reference_acquisition: int = 0,
    label_names: list[str],
) -> dict[str, list[dict[str, Any]]]:
    """Initialize label copy across multiplexing acquisitions.

    Groups images by well, validates that the reference acquisition is present
    exactly once per well and that all requested labels exist in it, then
    emits one parallelization item per (well x non-reference acquisition) pair.

    Args:
        zarr_urls: Paths to all OME-Zarr images to be processed.
            (standard argument for Fractal non-parallel tasks).
        zarr_dir: Path to the directory containing the OME-Zarr images.
            (standard argument for Fractal non-parallel tasks).
        reference_acquisition: Acquisition ID (integer) of the image that
            contains the source labels. Defaults to 0.
        label_names: Names of the labels to copy from the reference
            acquisition to all other acquisitions in the same well.

    Returns:
        Dictionary with a ``parallelization_list`` for the Fractal server.
    """
    logger.info(
        f"Running `copy_labels_multiplexing_init` "
        f"on {len(zarr_urls)} images, {reference_acquisition=}, "
        f"{label_names=}."
    )

    # Group zarr_urls by well path
    wells: dict[str, list[str]] = {}
    for url in zarr_urls:
        well_path = _url_dirname(url)
        wells.setdefault(well_path, []).append(url)

    parallelization_list: list[dict[str, Any]] = []

    for well_path, well_urls in wells.items():
        # Derive plate root (2 levels up from image: image → col → row → plate)
        plate_root = _url_dirname(_url_dirname(well_path))
        plate = ngio.open_ome_zarr_plate(plate_root)

        # Extract row and column from the well path
        row_col = well_path[len(plate_root) + 1 :]
        row, col = row_col.split("/")
        well = plate.get_well(row=row, column=col)

        # Map each url to its acquisition ID
        acq_map: dict[str, int | None] = {
            url: well.get_image_acquisition_id(_url_basename(url)) for url in well_urls
        }

        ref_urls = [url for url, acq in acq_map.items() if acq == reference_acquisition]

        if len(ref_urls) == 0:
            raise ValueError(
                f"Reference acquisition {reference_acquisition} not found in "
                f"well '{well_path}'. Available acquisition IDs: "
                f"{sorted({v for v in acq_map.values() if v is not None})}."
            )
        if len(ref_urls) > 1:
            raise ValueError(
                f"Multiple images with acquisition ID {reference_acquisition} "
                f"found in well '{well_path}': {ref_urls}. "
                "Exactly one reference image per well is required."
                "This task doesn't currently support multiple reference "
                "acquisitions per well, as the matching of acquisitions is "
                "would then be ambiguous."
            )

        ref_url = ref_urls[0]

        # Validate that all requested labels exist in the reference
        ref_container = ngio.open_ome_zarr_container(ref_url)
        existing_labels = ref_container.list_labels()
        missing = [n for n in label_names if n not in existing_labels]
        if missing:
            raise ValueError(
                f"The following labels are missing from the reference image "
                f"'{ref_url}': {missing}. "
                f"Available labels: {existing_labels}."
            )

        # Emit one job per non-reference acquisition in this well
        target_urls = [url for url in well_urls if url != ref_url]
        logger.info(
            f"Well '{well_path}': reference='{ref_url}', "
            f"{len(target_urls)} target acquisition(s)."
        )
        for target_url in target_urls:
            parallelization_list.append(
                {
                    "zarr_url": target_url,
                    "init_args": {
                        "reference_zarr_url": ref_url,
                        "label_names": label_names,
                    },
                }
            )

    logger.info(f"Finished init: {len(parallelization_list)} parallelization items.")
    return {"parallelization_list": parallelization_list}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=copy_labels_multiplexing_init)
