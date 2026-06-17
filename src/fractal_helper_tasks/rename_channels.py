# Copyright 2025 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
"""Task to rename channels in OME-Zarr images."""

import logging

import ngio
from pydantic import validate_call

logger = logging.getLogger("rename_channels")


@validate_call
def rename_channels(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    channel_name_map: dict[str, str],
    fail_on_missing_channels: bool = False,
) -> None:
    """Rename channels in OME-Zarr images.

    For each image, applies the supplied name mapping to the existing channel
    labels. Only Zarr metadata is updated — no image arrays are rewritten.
    Map entries whose keys are absent from an image are skipped with a warning
    (or raise a ``ValueError`` if ``fail_on_missing_channels`` is ``True``).

    Args:
        zarr_urls: Paths to all OME-Zarr images to be processed.
            (standard argument for Fractal non-parallel tasks).
        zarr_dir: Path to the directory containing the OME-Zarr images.
            (standard argument for Fractal non-parallel tasks).
        channel_name_map: Mapping of old channel name to new channel name.
            Enter the old channel name as the key and the new channel name as
            the value. May contain more entries than an image has channels;
            unmatched entries are skipped. Useful in multiplexing workflows
            where different images carry different channel subsets.
        fail_on_missing_channels: If ``True``, raise a ``ValueError`` when
            any key in ``channel_name_map`` is absent from an image's
            channels. Defaults to ``False`` (skip with a warning).
    """
    logger.info(
        f"Running `rename_channels` on {len(zarr_urls)} images. "
        f"Mapping: {channel_name_map}."
    )

    for url in zarr_urls:
        container = ngio.open_ome_zarr_container(url)
        current_labels = container.get_image().channel_labels

        missing = [k for k in channel_name_map if k not in current_labels]
        if missing:
            if fail_on_missing_channels:
                raise ValueError(
                    f"The following channels from the mapping are not present "
                    f"in '{url}': {missing}. "
                    f"Available channels: {current_labels}."
                )
            logger.warning(
                f"{url}: channels not found (skipped): {missing}. "
                f"Available channels: {current_labels}."
            )

        new_labels = [channel_name_map.get(label, label) for label in current_labels]

        if new_labels == current_labels:
            logger.info(f"{url}: no channels to rename, skipping.")
            continue

        renamed = {
            old: new
            for old, new in zip(current_labels, new_labels, strict=True)
            if old != new
        }
        logger.info(f"{url}: renaming channels {renamed}.")
        container.set_channel_labels(labels=new_labels)

    logger.info("Finished `rename_channels`.")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=rename_channels)
