"""Test copy 2D to 3D segmentation."""

from pathlib import Path

import ngio
import numpy as np
import pytest

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

    # Create a masking roi table in the 2D image
    masking_roi_table = ome_zarr_2d.get_masked_image(label_name).build_image_roi_table(
        name="masking_ROI_table"
    )
    ome_zarr_2d.add_table(
        name="masking_ROI_table",
        table=masking_roi_table,
    )


# @pytest.mark.parametrize("new_label_name", [None, "nuclei_new"])
# def test_2d_to_3d_synthetic_data(tmp_path: Path, z=2.0):
#     """Test that the z-spacing is copied correctly."""
#     zarr_url = str(tmp_path / "plate_mip.zarr" / "B" / "03" / "0")
#     zarr_url_3d = str(tmp_path / "plate.zarr" / "B" / "03" / "0")
#     label_name = "nuclei"

#     create_synthetic_data(zarr_url, zarr_url_3d, label_name, z)

#     convert_2D_segmentation_to_3D(
#         zarr_url=zarr_url,
#         label_name=label_name,
#         tables_to_copy=["masking_ROI_table"],
#     )


# TODO: Parameterize this test with different z-spacing values
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

    # TODO: Check that tables are copied over correctly


def test_2d_to_3d_real_data(tmp_zenodo_zarr: list[str]):
    print(tmp_zenodo_zarr)
    zarr_url = f"{tmp_zenodo_zarr[1]}/B/03/0"
    label_name = "nuclei"
    tables_to_copy = ["nuclei_ROI_table", "nuclei"]

    # Create a masking roi table in the 2D image
    ome_zarr_2d = ngio.open_ome_zarr_container(zarr_url)
    masking_roi_table = ome_zarr_2d.get_masked_image("nuclei").build_image_roi_table(
        name=tables_to_copy[0]
    )

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

    # for table_name in roi_table_names:
    #     roi_table = ome_zarr_3d.get_roi_table(name=table_name)
    #     assert roi_table is not None
    #     assert isinstance(roi_table, zarr.core.Array)


# TODO: Add a feature table & have it copied over

# TODO: Add test with new table names

# TODO: Create a version of the test data where image suffixes need to be
# changed, run tests on those
