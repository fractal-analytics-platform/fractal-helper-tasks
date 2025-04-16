"""Fractal task to convert 2D segmentations into 3D segmentations."""

import logging
from typing import Optional

import dask.array as da
import ngio
from ngio.utils import NgioFileNotFoundError
from pydantic import validate_call

logger = logging.getLogger(__name__)


@validate_call
def convert_2D_segmentation_to_3D(
    zarr_url: str,
    label_name: str,
    level: str = "0",
    tables_to_copy: Optional[list[str]] = None,
    new_label_name: Optional[str] = None,
    new_table_names: Optional[list] = None,
    plate_suffix: str = "_mip",
    image_suffix_2D_to_remove: Optional[str] = None,
    image_suffix_3D_to_add: Optional[str] = None,
    z_chunks: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """Convert 2D segmentation to 3D segmentation.

    This task loads the 2D segmentation, replicates it along the Z slice and
    stores it back into the 3D OME-Zarr image.

    This is a temporary workaround task, as long as we store 2D data in
    a separate OME-Zarr file from the 3D data. If the 2D & 3D OME-Zarr images
    have different suffixes in their name, use `image_suffix_2D_to_remove` &
    `image_suffix_3D_to_add`. If their base names are different, this task
    does not support processing them at the moment.

    It makes the assumption that the 3D OME-Zarrs are stored in the same place
    as the 2D OME-Zarrs (same based folder).

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        label_name: Name of the label to copy from 2D OME-Zarr to
            3D OME-Zarr
        tables_to_copy: List of tables to copy from 2D OME-Zarr
            to 3D OME-Zarr
        new_label_name: Optionally overwriting the name of the label in
            the 3D OME-Zarr
        new_table_names: Optionally overwriting the names of the tables
            in the 3D OME-Zarr
        level: Level of the 2D OME-Zarr label to copy from. Valid choices are
            "0", "1", etc. (depending on which levels are available in the
            OME-Zarr label).
        plate_suffix: Suffix of the 2D OME-Zarr that needs to be removed to
            generate the path to the 3D OME-Zarr. If the 2D OME-Zarr is
            "/path/to/my_plate_mip.zarr/B/03/0" and the 3D OME-Zarr is located
            in "/path/to/my_plate.zarr/B/03/0", the correct suffix is "_mip".
        image_suffix_2D_to_remove: If the image name between 2D & 3D don't
            match, this is the suffix that should be removed from the 2D image.
            If the 2D image is in "/path/to/my_plate_mip.zarr/B/03/
            0_registered" and the 3D image is in "/path/to/my_plate.zarr/
            B/03/0", the value should be "_registered"
        image_suffix_3D_to_add: If the image name between 2D & 3D don't
            match, this is the suffix that should be added to the 3D image.
            If the 2D image is in "/path/to/my_plate_mip.zarr/B/03/0" and the
            3D image is in "/path/to/my_plate.zarr/B/03/0_illum_corr", the
            value should be "_illum_corr".
        z_chunks: Chunking for the Z dimension. Set this parameter if you want
            the label image to be chunked differently from the 3D image in
            the Z dimension.
        overwrite: If `True`, overwrite existing label and ROI tables in the
            3D OME-Zarr
    """
    logger.info("Starting 2D to 3D conversion")
    # Normalize zarr_url
    zarr_url = zarr_url.rstrip("/")
    # 0) Preparation
    if new_table_names:
        if not tables_to_copy:
            raise ValueError(
                "If new_table_names is set, tables_to_copy must also be set."
            )
        if len(new_table_names) != len(tables_to_copy):
            raise ValueError(
                "If new_table_names is set, it must have the same number of "
                f"entries as tables_to_copy. They were: {new_table_names=}"
                f"and {tables_to_copy=}"
            )

    zarr_3D_url = zarr_url.replace(f"{plate_suffix}.zarr", ".zarr")
    # Handle changes to image name
    if image_suffix_2D_to_remove:
        zarr_3D_url = zarr_3D_url.rstrip(image_suffix_2D_to_remove)
    if image_suffix_3D_to_add:
        zarr_3D_url += image_suffix_3D_to_add

    if new_label_name is None:
        new_label_name = label_name
    if new_table_names is None:
        new_table_names = tables_to_copy

    try:
        ome_zarr_container_3d = ngio.open_ome_zarr_container(zarr_3D_url)
    except NgioFileNotFoundError as e:
        raise ValueError(
            f"3D OME-Zarr {zarr_3D_url} not found. Please check the "
            f"suffix (set to {plate_suffix})."
        ) from e

    logger.info(
        f"Copying {label_name} from {zarr_url} to {zarr_3D_url} as "
        f"{new_label_name}."
    )

    # 1) Load a 2D label image
    ome_zarr_container_2d = ngio.open_ome_zarr_container(zarr_url)
    label_img = ome_zarr_container_2d.get_label(label_name, path=level)

    if not label_img.is_2d:
        raise ValueError(
            f"Label image {label_name} is not 2D. It has a shape of "
            f"{label_img.shape} and the axes "
            f"{label_img.axes_mapper.on_disk_axes_names}."
        )

    chunks = list(label_img.chunks)
    label_dask = label_img.get_array(mode="dask")

    # 2) Set up the 3D label image
    ref_image_3d = ome_zarr_container_3d.get_image(
        pixel_size=label_img.pixel_size,
    )

    z_index = label_img.axes_mapper.get_index("z")
    y_index = label_img.axes_mapper.get_index("y")
    x_index = label_img.axes_mapper.get_index("x")
    z_index_3d_reference = ref_image_3d.axes_mapper.get_index("z")
    if z_chunks:
        chunks[z_index] = z_chunks
    else:
        chunks[z_index] = ref_image_3d.chunks[z_index_3d_reference]
    chunks = tuple(chunks)

    nb_z_planes = ref_image_3d.shape[z_index_3d_reference]

    shape_3d = (nb_z_planes, label_img.shape[y_index], label_img.shape[x_index])

    pixel_size = label_img.pixel_size
    pixel_size.z = ref_image_3d.pixel_size.z
    axes_names = label_img.axes_mapper.on_disk_axes_names

    z_extent = nb_z_planes * pixel_size.z

    new_label_container = ome_zarr_container_3d.derive_label(
        name=new_label_name,
        ref_image=ref_image_3d,
        shape=shape_3d,
        pixel_size=pixel_size,
        axes_names=axes_names,
        chunks=chunks,
        dtype=label_img.dtype,
        overwrite=overwrite,
    )

    # 3) Create the 3D stack of the label image
    label_img_3D = da.stack([label_dask.squeeze()] * nb_z_planes)

    # 4) Save changed label image to OME-Zarr
    new_label_container.set_array(label_img_3D, axes_order="zyx")

    logger.info(f"Saved {new_label_name} to 3D Zarr at full resolution")
    # 5) Build pyramids for label image
    new_label_container.consolidate()
    logger.info(f"Built a pyramid for the {new_label_name} label image")

    # 6) Copy tables
    if tables_to_copy:
        for i, table_name in enumerate(tables_to_copy):
            if table_name not in ome_zarr_container_2d.list_tables():
                raise ValueError(
                    f"Table {table_name} not found in 2D OME-Zarr {zarr_url}."
                )
            table = ome_zarr_container_2d.get_table(table_name)
            if table.type() == "roi_table" or table.type() == "masking_ROI_table":
                for roi in table.rois():
                    roi.z_length = z_extent
            else:
                # For some reason, I need to load the table explicitly before
                # I can write it again
                # FIXME: Check with Lorenzo why this is
                table.dataframe  # noqa #B018
            ome_zarr_container_3d.add_table(
                name=new_table_names[i], table=table, overwrite=False
            )

    logger.info("Finished 2D to 3D conversion")

    # Give the 3D image as an output so that filters are applied correctly
    # (because manifest type filters get applied to the output image)
    image_list_updates = dict(
        image_list_updates=[
            dict(
                zarr_url=zarr_3D_url,
            )
        ]
    )
    return image_list_updates


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=convert_2D_segmentation_to_3D,
        logger_name=logger.name,
    )
