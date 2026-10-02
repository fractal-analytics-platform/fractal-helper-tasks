# Copyright 2025 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
"""Task to delete images, labels and tables from OME-Zarr images."""

import logging

import fsspec
import ngio
from ngio.utils import NgioError
from pydantic import validate_call

logger = logging.getLogger("delete_images_labels_tables")


@validate_call
def delete_images_labels_tables(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    images_names_to_delete: list[str] | None = None,
    labels_to_delete: list[str] | None = None,
    tables_to_delete: list[str] | None = None,
) -> dict[str, list[str]] | None:
    """Delete images, labels and tables from OME-Zarr images.

    For each image, lists the existing labels and tables, logs which ones
    will be deleted, and removes them. Items in the deletion lists that do
    not exist in a given image are silently skipped.

    Args:
        zarr_urls: Paths to all OME-Zarr images to be processed.
            (standard argument for Fractal non-parallel tasks).
        zarr_dir: Path to the directory containing the OME-Zarr images.
            (standard argument for Fractal non-parallel tasks).
        images_names_to_delete: Names of images to delete from each OME-Zarr.
            Checks for each image if it matches any of the listed names. If 
            yes, it gets deleted (both on disk, removed from Fractal & from 
            a potential HCS plate metadata).
        labels_to_delete: Names of label images to delete from each
            OME-Zarr. Labels absent from a given image are skipped.
        tables_to_delete: Names of tables to delete from each OME-Zarr.
            Tables absent from a given image are skipped.
    """
    images_names_to_delete = images_names_to_delete or []
    labels_to_delete = labels_to_delete or []
    tables_to_delete = tables_to_delete or []
    logger.info(
        f"Running `delete_images_labels_tables` on {len(zarr_urls)} images. "
        f"Images to delete: {images_names_to_delete}. "
        f"Labels to delete: {labels_to_delete}. "
        f"Tables to delete: {tables_to_delete}."
    )
    image_list_removals = []
    # First delete all the images
    remaining_zarr_urls = []
    if len(images_names_to_delete) > 0:
        for url in zarr_urls:
            image_name = url.rstrip("/").split("/")[-1]
            if image_name in images_names_to_delete:
                # If the image is part of a HCS plate, remove it from the
                # well metadata. Done before deleting the data, so the well
                # never references a missing image if the deletion fails.
                well_url, _ = url.rstrip("/").rsplit("/", 1)
                try:
                    well = ngio.open_ome_zarr_well(well_url, mode="r")
                except NgioError:
                    well = None
                if well is not None and image_name in well.paths():
                    plate_url, row, column = well_url.rsplit("/", 2)
                    plate = ngio.open_ome_zarr_plate(plate_url)
                    plate.remove_image(
                        row=row, column=column, image_path=image_name
                    )
                    logger.debug(
                        f"Removed {image_name} from well {row}/{column} "
                        f"metadata of plate {plate_url}."
                    )

                # Delete the image data from disk
                fs, path = fsspec.url_to_fs(url)
                if fs.exists(path):
                    fs.rm(path, recursive=True)
                    logger.info(f"Deleted image {url} from disk.")

                # Add deleted image to the image_list_removals
                image_list_removals.append(url)
            else:
                remaining_zarr_urls.append(url)
    else:
        remaining_zarr_urls = zarr_urls

    # For the remaining zarr_urls after image deletion, clean up labels and 
    # tables
    for url in remaining_zarr_urls:
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

    logger.info("Finished `delete_images_labels_tables`.")
    if len(image_list_removals) > 0:
        return {"image_list_removals": image_list_removals}


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=delete_images_labels_tables)
