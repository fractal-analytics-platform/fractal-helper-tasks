"""Test drop t dimension task."""

from pathlib import Path

import ngio
import numpy as np
import pytest

from fractal_helper_tasks.drop_t_dimension import (
    drop_t_dimension,
)


@pytest.mark.parametrize("overwrite_input", [False, True])
def test_drop_t_dimension(
    tmp_path: Path,
    overwrite_input: bool,
):
    zarr_url = str(tmp_path / "my_zarr.zarr")

    ngio.create_ome_zarr_from_array(
        store=zarr_url,
        array=np.zeros((1, 1, 1, 100, 100)),
        xy_pixelsize=0.5,
        z_spacing=1.0,
        axes_names="tczyx",
        overwrite=True,
    )

    drop_t_dimension(
        zarr_url=zarr_url,
        overwrite_input=overwrite_input,
    )

    if overwrite_input:
        result_url = zarr_url
    else:
        result_url = f"{zarr_url}_no_T"

    result_container = ngio.open_ome_zarr_container(result_url)
    assert result_container.image_meta.axes_handler.axes_names == (
        "c",
        "z",
        "y",
        "x",
    )
