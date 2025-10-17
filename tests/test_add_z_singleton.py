"""Test drop t dimension task."""

from pathlib import Path

import ngio
import numpy as np
import pytest

from fractal_helper_tasks.add_z_singleton import (
    add_z_singleton,
)


@pytest.mark.parametrize(
    "orig_axes_names, target_axes_names, orig_dimensions, "
    "target_dimensions, overwrite_input",
    [
        ("cyx", ["c", "z", "y", "x"], (5, 100, 100), (5, 1, 100, 100), True),
        ("cyx", ["c", "z", "y", "x"], (5, 100, 100), (5, 1, 100, 100), False),
        ("yx", ["z", "y", "x"], (100, 100), (1, 100, 100), True),
        (
            "tcyx",
            ["t", "c", "z", "y", "x"],
            (3, 5, 100, 100),
            (3, 5, 1, 100, 100),
            True,
        ),
        ("tyx", ["t", "z", "y", "x"], (3, 100, 100), (3, 1, 100, 100), True),
    ],
)
def test_add_singleton(
    tmp_path: Path,
    orig_axes_names: str,
    target_axes_names: list[str],
    orig_dimensions: tuple[int],
    target_dimensions: tuple[int],
    overwrite_input: bool,
):
    zarr_url = str(tmp_path / "my_zarr.zarr")

    ngio.create_ome_zarr_from_array(
        store=zarr_url,
        array=np.zeros(orig_dimensions),
        xy_pixelsize=0.5,
        axes_names=orig_axes_names,
        overwrite=True,
    )

    suffix = "z_singleton"
    add_z_singleton(
        zarr_url=zarr_url,
        suffix=suffix,
        overwrite_input=overwrite_input,
    )

    if overwrite_input:
        new_zarr_url = zarr_url
    else:
        new_zarr_url = f"{zarr_url}_{suffix}"

    new_ome_zarr_container = ngio.open_ome_zarr_container(new_zarr_url)
    assert (
        list(new_ome_zarr_container.image_meta.axes_handler.axes_names)
        == target_axes_names
    )
    assert new_ome_zarr_container.get_image().pixel_size.z == 1.0
    assert new_ome_zarr_container.get_image().shape == target_dimensions
