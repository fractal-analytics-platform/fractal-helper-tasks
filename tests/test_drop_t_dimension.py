"""Test drop t dimension task."""

from pathlib import Path

import ngio
import numpy as np

from fractal_helper_tasks.drop_t_dimension import (
    drop_t_dimension,
)


def test_drop_t_dimension(
    tmp_path: Path,
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
        overwrite_input=False,
    )
    new_zarr_url = f"{zarr_url}_no_T"
    new_ome_zarr_container = ngio.open_ome_zarr_container(new_zarr_url)
    assert new_ome_zarr_container.image_meta.axes_mapper.on_disk_axes_names == [
        "c",
        "z",
        "y",
        "x",
    ]
