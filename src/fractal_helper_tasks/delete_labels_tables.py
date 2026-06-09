# Copyright 2025 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
"""Task to delete labels and tables from OME-Zarr images."""

import logging

import ngio
from pydantic import validate_call

logger = logging.getLogger("delete_labels_tables")


@validate_call
def delete_labels_tables(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    labels_to_delete: list[str] | None = None,
    tables_to_delete: list[str] | None = None,
) -> None:
    """Delete labels and tables from OME-Zarr images.

    For each image, lists the existing labels and tables, logs which ones
    will be deleted, and removes them. Items in the deletion lists that do
    not exist in a given image are silently skipped.

    Args:
        zarr_urls: Paths to all OME-Zarr images to be processed.
            (standard argument for Fractal non-parallel tasks).
        zarr_dir: Path to the directory containing the OME-Zarr images.
            (standard argument for Fractal non-parallel tasks).
        labels_to_delete: Names of label images to delete from each
            OME-Zarr. Labels absent from a given image are skipped.
        tables_to_delete: Names of tables to delete from each OME-Zarr.
            Tables absent from a given image are skipped.
    """
    labels_to_delete = labels_to_delete or []
    tables_to_delete = tables_to_delete or []
    logger.info(
        f"Running `delete_labels_tables` on {len(zarr_urls)} images. "
        f"Labels to delete: {labels_to_delete}. "
        f"Tables to delete: {tables_to_delete}."
    )

    for url in zarr_urls:
        container = ngio.open_ome_zarr_container(url)

        existing_labels = container.list_labels()
        existing_tables = container.list_tables()
        logger.info(
            f"{url}: existing labels={existing_labels}, "
            f"existing tables={existing_tables}."
        )

        labels_to_remove = [n for n in labels_to_delete if n in existing_labels]
        tables_to_remove = [t for t in tables_to_delete if t in existing_tables]

        if not labels_to_remove and not tables_to_remove:
            logger.info(f"{url}: nothing to delete, skipping.")
            continue

        logger.info(
            f"{url}: deleting labels={labels_to_remove}, tables={tables_to_remove}."
        )

        for name in labels_to_delete:
            container.delete_label(name, missing_ok=True)

        for name in tables_to_delete:
            container.delete_table(name, missing_ok=True)

    logger.info("Finished `delete_labels_tables`.")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=delete_labels_tables,
        logger_name=logger.name,
    )
