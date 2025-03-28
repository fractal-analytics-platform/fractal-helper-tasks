"""Fractal task to convert 2D segmentations into 3D segmentations."""

import logging
from typing import Optional

import anndata as ad
import dask.array as da
import numpy as np
import zarr
from fractal_tasks_core.labels import prepare_label_group
from fractal_tasks_core.ngff.zarr_utils import load_NgffImageMeta
from fractal_tasks_core.pyramids import build_pyramid
from fractal_tasks_core.tables import write_table
from pydantic import validate_call

logger = logging.getLogger(__name__)


def read_table_and_attrs(zarr_url: str, roi_table):
    """Read table & attrs from Zarr Anndata tables."""
    table_url = f"{zarr_url}/tables/{roi_table}"
    table = ad.read_zarr(table_url)
    table_attrs = get_zattrs(table_url)
    return table, table_attrs


def update_table_metadata(group_tables, table_name):
    """Update table metadata."""
    if "tables" not in group_tables.attrs:
        group_tables.attrs["tables"] = [table_name]
    elif table_name not in group_tables.attrs["tables"]:
        group_tables.attrs["tables"] = group_tables.attrs["tables"] + [table_name]


def get_zattrs(zarr_url):
    """Get zattrs of a Zarr as a dictionary."""
    with zarr.open(zarr_url, mode="r") as zarr_img:
        return zarr_img.attrs.asdict()


def make_zattrs_3D(attrs, z_pixel_size, new_label_name):
    """Creates 3D zattrs based on 2D attrs.

    Performs the following checks:
    1) If the label image has 2 axes, add a Z axis and updadte the
    coordinateTransformations
    2) Change the label name that is referenced, if a new name is provided
    """
    if len(attrs["multiscales"][0]["axes"]) == 3:
        pass
    # If we're getting a 2D image, we need to add a Z axis
    elif len(attrs["multiscales"][0]["axes"]) == 2:
        z_axis = attrs["multiscales"][0]["axes"][-1]
        z_axis["name"] = "z"
        attrs["multiscales"][0]["axes"] = [z_axis] + attrs["multiscales"][0]["axes"]
        for i, dataset in enumerate(attrs["multiscales"][0]["datasets"]):
            if len(dataset["coordinateTransformations"][0]["scale"]) == 2:
                attrs["multiscales"][0]["datasets"][i]["coordinateTransformations"][0][
                    "scale"
                ] = [z_pixel_size] + dataset["coordinateTransformations"][0]["scale"]
            else:
                raise NotImplementedError(
                    f"A dataset with 2 axes {attrs['multiscales'][0]['axes']}"
                    "must have coordinateTransformations with 2 scales. "
                    "Instead, it had "
                    f"{dataset['coordinateTransformations'][0]['scale']}"
                )
    else:
        raise NotImplementedError("The label image must have 2 or 3 axes")
    attrs["multiscales"][0]["name"] = new_label_name
    return attrs


def check_table_validity(new_table_names, old_table_names):
    """Validate table mapping between old & new tables."""
    if new_table_names and old_table_names:
        if len(new_table_names) != len(old_table_names):
            raise ValueError(
                "The number of new table names must match the number of old "
                f"table names. Instead, the task got {len(new_table_names)}"
                "new table names vs. {len(old_table_names)} old table names."
                "Check the task configuration, specifically `new_table_names`"
            )
        if len(set(new_table_names)) != len(new_table_names):
            raise ValueError(
                "The new table names must be unique. Instead, the task got "
                f"{new_table_names}"
            )


@validate_call
def convert_2D_segmentation_to_3D(
    zarr_url: str,
    label_name: str,
    level: int = 0,
    ROI_tables_to_copy: Optional[list[str]] = None,
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
        ROI_tables_to_copy: List of ROI table names to copy from 2D OME-Zarr
            to 3D OME-Zarr
        new_label_name: Optionally overwriting the name of the label in
            the 3D OME-Zarr
        new_table_names: Optionally overwriting the names of the ROI tables
            in the 3D OME-Zarr
        level: Level of the 2D OME-Zarr label to copy from
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
    if level != 0:
        raise NotImplementedError("Only level 0 is supported at the moment")
    zarr_3D_url = zarr_url.replace(f"{plate_suffix}.zarr", ".zarr")
    # Handle changes to image name
    # (would get easier if projections were subgroups!)
    if image_suffix_2D_to_remove:
        zarr_3D_url = zarr_3D_url.rstrip(image_suffix_2D_to_remove)
    if image_suffix_3D_to_add:
        zarr_3D_url += image_suffix_3D_to_add

    # FIXME: Check whether 3D Zarr actually exists

    if new_label_name is None:
        new_label_name = label_name
    if new_table_names is None:
        new_table_names = ROI_tables_to_copy

    check_table_validity(new_table_names, ROI_tables_to_copy)
    logger.info(
        f"Copying {label_name} from {zarr_url} to {zarr_3D_url} as "
        f"{new_label_name}."
    )

    # 1a) Load a 2D label image
    label_img = da.from_zarr(f"{zarr_url}/labels/{label_name}/{level}")
    chunks = list(label_img.chunksize)

    # 1b) Get number z planes & Z spacing from 3D OME-Zarr file
    with zarr.open(zarr_3D_url, mode="rw+") as zarr_img:
        zarr_3D = da.from_zarr(zarr_img[0])
        new_z_planes = zarr_3D.shape[-3]
        z_chunk_3d = zarr_3D.chunksize[-3]

    # TODO: Improve axis detection in ngio refactor?
    if z_chunks:
        chunks[-3] = z_chunks
    else:
        chunks[-3] = z_chunk_3d
    chunks = tuple(chunks)

    image_meta = load_NgffImageMeta(zarr_3D_url)
    z_pixel_size = image_meta.get_pixel_sizes_zyx(level=0)[0]

    # Prepare the output label group
    # Get the label_attrs correctly (removes hack below)
    label_attrs = get_zattrs(zarr_url=f"{zarr_url}/labels/{label_name}")
    label_attrs = make_zattrs_3D(label_attrs, z_pixel_size, new_label_name)
    output_label_group = prepare_label_group(
        image_group=zarr.group(zarr_3D_url),
        label_name=new_label_name,
        overwrite=overwrite,
        label_attrs=label_attrs,
        logger=logger,
    )

    logger.info(f"Helper function `prepare_label_group` returned {output_label_group=}")

    # 2) Create the 3D stack of the label image
    label_img_3D = da.stack([label_img.squeeze()] * new_z_planes)

    # 3) Save changed label image to OME-Zarr
    label_dtype = np.uint32
    store = zarr.storage.FSStore(f"{zarr_3D_url}/labels/{new_label_name}/0")
    new_label_array = zarr.create(
        shape=label_img_3D.shape,
        chunks=chunks,
        dtype=label_dtype,
        store=store,
        overwrite=False,
        dimension_separator="/",
    )

    da.array(label_img_3D).to_zarr(
        url=new_label_array,
    )
    logger.info(f"Saved {new_label_name} to 3D Zarr at full resolution")
    # 4) Build pyramids for label image
    label_meta = load_NgffImageMeta(f"{zarr_url}/labels/{label_name}")
    build_pyramid(
        zarrurl=f"{zarr_3D_url}/labels/{new_label_name}",
        overwrite=overwrite,
        num_levels=label_meta.num_levels,
        coarsening_xy=label_meta.coarsening_xy,
        chunksize=chunks,
        aggregation_function=np.max,
    )
    logger.info(f"Built a pyramid for the {new_label_name} label image")

    # 5) Copy ROI tables
    image_group = zarr.group(zarr_3D_url)
    if ROI_tables_to_copy:
        for i, ROI_table in enumerate(ROI_tables_to_copy):
            new_table_name = new_table_names[i]
            logger.info(f"Copying ROI table {ROI_table} as {new_table_name}")
            roi_an, table_attrs = read_table_and_attrs(zarr_url, ROI_table)
            nb_rois = len(roi_an.X)
            # Set the new Z values to span the whole ROI
            roi_an.X[:, 5] = np.array([z_pixel_size * new_z_planes] * nb_rois)

            write_table(
                image_group=image_group,
                table_name=new_table_name,
                table=roi_an,
                overwrite=overwrite,
                table_attrs=table_attrs,
            )

    logger.info("Finished 2D to 3D conversion")

    # Give the 3D image as an output so that filters are applied correctly
    image_list_updates = dict(
        image_list_updates=[
            dict(
                zarr_url=zarr_3D_url,
            )
        ]
    )
    return image_list_updates


if __name__ == "__main__":
    from fractal_tasks_core.tasks._utils import run_fractal_task

    run_fractal_task(
        task_function=convert_2D_segmentation_to_3D,
        logger_name=logger.name,
    )
