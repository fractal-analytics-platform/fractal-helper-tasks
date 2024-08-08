"""Test copy 2D to 3D segmentation."""

import dask.array as da
import pytest
import zarr

from fractal_helper_tasks.convert_2D_segmentation_to_3D import (
    convert_2D_segmentation_to_3D,
)


@pytest.mark.parametrize("new_label_name", [None, "nuclei_new"])
def test_2d_to_3d(tmp_zenodo_zarr: list[str], new_label_name):
    zarr_url = f"{tmp_zenodo_zarr[1]}/B/03/0"
    label_name = "nuclei"

    convert_2D_segmentation_to_3D(
        zarr_url=zarr_url,
        label_name=label_name,
        new_label_name=new_label_name,
    )

    if not new_label_name:
        new_label_name = label_name

    zarr_3D_label_url = f"{tmp_zenodo_zarr[0]}/B/03/0/labels/{new_label_name}"
    # Check that the label has been copied correctly
    with zarr.open(zarr_3D_label_url, mode="rw+") as zarr_img:
        zarr_3D = da.from_zarr(zarr_img[0])
        assert zarr_3D.shape == (2, 540, 1280)


# TODO: Add custom ROI tables to be copied to 3D

# TODO: Add a feature table & have it copied over

# TODO: Add test with new table names

# TODO: Create a version of the test data where image suffixes need to be
# changed, run tests on those
