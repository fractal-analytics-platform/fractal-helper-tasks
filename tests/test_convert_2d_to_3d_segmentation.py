"""Test copy 2D to 3D segmentation."""

import dask.array as da
import zarr

from fractal_helper_tasks.convert_2D_segmentation_to_3D import (
    convert_2D_segmentation_to_3D,
)


def test_2d_to_3d(tmp_zenodo_zarr: list[str]):
    zarr_url = f"{tmp_zenodo_zarr[1]}/B/03/0"
    label_name = "nuclei"

    convert_2D_segmentation_to_3D(
        zarr_url=zarr_url,
        label_name=label_name,
    )
    zarr_3D_label_url = f"{tmp_zenodo_zarr[0]}/B/03/0/labels/{label_name}"
    # Check that the label has been copied correctly
    with zarr.open(zarr_3D_label_url, mode="rw+") as zarr_img:
        zarr_3D = da.from_zarr(zarr_img[0])
        assert zarr_3D.shape == (2, 540, 1280)


# TODO: Add custom ROI tables to be copied to 3D

# TODO: Add test with new label name, new table names

# TODO: Create a version of the test data where image suffixes need to be
# changed, run tests on those
