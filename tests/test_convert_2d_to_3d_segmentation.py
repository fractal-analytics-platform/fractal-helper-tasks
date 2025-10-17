"""Test copy 2D to 3D segmentation."""

from pathlib import Path

import ngio
import numpy as np
import pandas as pd
import pytest
from ngio.tables import GenericTable

from fractal_helper_tasks.convert_2D_segmentation_to_3D import (
    convert_2D_segmentation_to_3D,
)


def create_synthetic_data(zarr_url, zarr_url_3d, label_name, z_spacing=1.0):
    base_array = np.zeros(
        shape=(1, 1, 100, 100),
    )
    base_array_3d = np.zeros(
        shape=(1, 10, 100, 100),
    )
    label_array = np.zeros(
        shape=(1, 100, 100),
    )
    label_array[:, 20:40, 30:50] = 1

    ome_zarr_2d = ngio.create_ome_zarr_from_array(
        store=zarr_url,
        array=base_array,
        xy_pixelsize=0.5,
        z_spacing=1.0,
    )

    ngio.create_ome_zarr_from_array(
        store=zarr_url_3d,
        array=base_array_3d,
        xy_pixelsize=0.5,
        z_spacing=z_spacing,
    )

    label_img = ome_zarr_2d.derive_label(
        name=label_name,
        ref_image=ome_zarr_2d.get_image(),
        dtype="uint16",
    )
    label_img.set_array(label_array)
    label_img.consolidate()

    image_roi_table = ome_zarr_2d.build_image_roi_table(name="image_ROI_table")
    ome_zarr_2d.add_table(
        name="image_ROI_table",
        table=image_roi_table,
    )

    # Create a masking roi table in the 2D image
    masking_roi_table = ome_zarr_2d.build_masking_roi_table(label=label_name)
    ome_zarr_2d.add_table(
        name="masking_ROI_table",
        table=masking_roi_table,
    )


@pytest.mark.parametrize("new_label_name", [None, "nuclei_new", "test"])
def test_2d_to_3d_label_renaming(tmp_path: Path, new_label_name):
    """Test that the z-spacing is copied correctly."""
    zarr_url = str(tmp_path / "plate_mip.zarr" / "B" / "03" / "0")
    zarr_url_3d = str(tmp_path / "plate.zarr" / "B" / "03" / "0")
    label_name = "nuclei"

    create_synthetic_data(zarr_url, zarr_url_3d, label_name, z_spacing=1.0)

    convert_2D_segmentation_to_3D(
        zarr_url=zarr_url,
        label_name=label_name,
        tables_to_copy=["masking_ROI_table"],
        new_label_name=new_label_name,
    )
    ome_zarr_3d = ngio.open_ome_zarr_container(zarr_url_3d)
    if new_label_name is None:
        assert ome_zarr_3d.list_labels() == ["nuclei"]
    else:
        assert ome_zarr_3d.list_labels() == [new_label_name]


@pytest.mark.parametrize("level", ["0", "1", "2"])
def test_2d_to_3d_varying_levels(tmp_path: Path, level):
    """Test that the z-spacing is copied correctly."""
    zarr_url = str(tmp_path / "plate_mip.zarr" / "B" / "03" / "0")
    zarr_url_3d = str(tmp_path / "plate.zarr" / "B" / "03" / "0")
    label_name = "nuclei"

    create_synthetic_data(zarr_url, zarr_url_3d, label_name, z_spacing=1.0)

    convert_2D_segmentation_to_3D(
        zarr_url=zarr_url,
        level=level,
        label_name=label_name,
        tables_to_copy=["masking_ROI_table"],
    )
    ome_zarr_3d = ngio.open_ome_zarr_container(zarr_url_3d)
    label_img = ome_zarr_3d.get_label(name="nuclei")
    assert label_img.pixel_size.x == 0.5 * (2 ** int(level))


@pytest.mark.parametrize("new_table_names", [None, ["new_table_names"]])
def test_2d_to_3d_table_renaming(tmp_path: Path, new_table_names):
    """Test that the z-spacing is copied correctly."""
    zarr_url = str(tmp_path / "plate_mip.zarr" / "B" / "03" / "0")
    zarr_url_3d = str(tmp_path / "plate.zarr" / "B" / "03" / "0")
    label_name = "nuclei"

    create_synthetic_data(zarr_url, zarr_url_3d, label_name, z_spacing=1.0)

    convert_2D_segmentation_to_3D(
        zarr_url=zarr_url,
        label_name=label_name,
        tables_to_copy=["masking_ROI_table"],
        new_table_names=new_table_names,
    )
    ome_zarr_3d = ngio.open_ome_zarr_container(zarr_url_3d)
    if new_table_names is None:
        assert ome_zarr_3d.list_tables() == ["masking_ROI_table"]
    else:
        assert ome_zarr_3d.list_tables() == new_table_names


def test_2d_to_3d_table_copying(tmp_path: Path):
    """Test that the z-spacing is copied correctly."""
    zarr_url = str(tmp_path / "plate_mip.zarr" / "B" / "03" / "0")
    zarr_url_3d = str(tmp_path / "plate.zarr" / "B" / "03" / "0")
    label_name = "nuclei"

    create_synthetic_data(zarr_url, zarr_url_3d, label_name, z_spacing=1.0)
    ome_zarr_2d = ngio.open_ome_zarr_container(zarr_url)
    roi_table = ome_zarr_2d.get_table("masking_ROI_table")
    assert roi_table.table_type() == "masking_roi_table"
    assert roi_table.rois()[0].z_length == 1.0

    convert_2D_segmentation_to_3D(
        zarr_url=zarr_url,
        label_name=label_name,
        tables_to_copy=["masking_ROI_table", "image_ROI_table"],
    )
    ome_zarr_3d = ngio.open_ome_zarr_container(zarr_url_3d)
    # Validate correctness of tables
    assert ome_zarr_3d.list_tables() == ["masking_ROI_table", "image_ROI_table"]
    roi_table = ome_zarr_3d.get_table("masking_ROI_table")
    assert roi_table.table_type() == "masking_roi_table"
    rois = roi_table.rois()
    assert len(rois) == 1
    assert rois[0].name == "1"
    assert rois[0].x_length == 10.0
    # z_length goes from 1 to 10 because we have 10 z planes
    assert rois[0].z_length == 10.0

    roi_table = ome_zarr_3d.get_table("image_ROI_table")
    assert roi_table.table_type() == "roi_table"
    rois = roi_table.rois()
    assert len(rois) == 1
    assert rois[0].name == "image_ROI_table"
    assert rois[0].x_length == 50.0
    # z_length goes from 1 to 10 because we have 10 z planes
    assert rois[0].z_length == 10.0


@pytest.mark.parametrize("z", [0.5, 1.0, 2.0])
def test_2d_to_3d_z_spacing(tmp_path: Path, z):
    """Test that the z-spacing is copied correctly."""
    zarr_url = str(tmp_path / "plate_mip.zarr" / "B" / "03" / "0")
    zarr_url_3d = str(tmp_path / "plate.zarr" / "B" / "03" / "0")
    label_name = "nuclei"

    create_synthetic_data(zarr_url, zarr_url_3d, label_name, z)

    convert_2D_segmentation_to_3D(
        zarr_url=zarr_url,
        label_name=label_name,
        tables_to_copy=["masking_ROI_table"],
    )

    ome_zarr_3d = ngio.open_ome_zarr_container(zarr_url_3d)
    label_img_3d = ome_zarr_3d.get_label(name=label_name).get_array(mode="dask")
    assert label_img_3d.shape == (10, 100, 100)
    assert ome_zarr_3d.get_label(name=label_name).pixel_size.z == z


def test_2d_to_3d_real_data(tmp_zenodo_zarr: list[str]):
    print(tmp_zenodo_zarr)
    zarr_url = f"{tmp_zenodo_zarr[1]}/B/03/0"
    label_name = "nuclei"
    tables_to_copy = ["nuclei_ROI_table", "nuclei"]

    # Create a masking roi table in the 2D image
    ome_zarr_2d = ngio.open_ome_zarr_container(zarr_url)
    masking_roi_table = ome_zarr_2d.build_masking_roi_table(label_name)

    ome_zarr_2d.add_table(
        name=tables_to_copy[0],
        table=masking_roi_table,
    )

    convert_2D_segmentation_to_3D(
        zarr_url=zarr_url,
        label_name=label_name,
        tables_to_copy=tables_to_copy,
    )

    zarr_3D_label_url = f"{tmp_zenodo_zarr[0]}/B/03/0"
    # Check that the label has been copied correctly
    ome_zarr_3d = ngio.open_ome_zarr_container(zarr_3D_label_url)
    label_img_3d = ome_zarr_3d.get_label(name=label_name).get_array(mode="dask")
    assert label_img_3d.shape == (2, 540, 1280)
    assert (
        ome_zarr_3d.list_tables()
        == ["FOV_ROI_table", "well_ROI_table"] + tables_to_copy  # noqa RUF005
    )


def test_2d_to_3d_real_data_no_label_copy(tmp_zenodo_zarr: list[str]):
    print(tmp_zenodo_zarr)
    zarr_url = f"{tmp_zenodo_zarr[1]}/B/03/0"
    tables_to_copy = ["generic_table", "nuclei"]

    # Create a masking roi table in the 2D image
    ome_zarr_2d = ngio.open_ome_zarr_container(zarr_url)

    # Add a generic table to be copied over
    generic_table = GenericTable(
        table_data=pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    )

    ome_zarr_2d.add_table(
        name=tables_to_copy[0],
        table=generic_table,
    )

    convert_2D_segmentation_to_3D(
        zarr_url=zarr_url,
        label_name=None,
        tables_to_copy=tables_to_copy,
    )

    zarr_3D_label_url = f"{tmp_zenodo_zarr[0]}/B/03/0"
    # Check that the label has been copied correctly
    ome_zarr_3d = ngio.open_ome_zarr_container(zarr_3D_label_url)
    assert len(ome_zarr_3d.list_labels()) == 0
    assert (
        ome_zarr_3d.list_tables()
        == ["FOV_ROI_table", "well_ROI_table"] + tables_to_copy  # noqa RUF005
    )


# TODO: Test table content more carefully
