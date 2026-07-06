import json
from pathlib import Path

import ngio
import numpy as np
import pytest

from fractal_helper_tasks.roi_selection_task import roi_selection_task


def _roi_json(**kwargs: object) -> str:
    """Build a JSON string from keyword arguments."""
    return json.dumps(kwargs)


def test_roi_selection_single_roi(test_data_3d_path: Path) -> None:
    """Test storing a single ROI."""
    roi_selection_task(
        zarr_url=test_data_3d_path.as_posix(),
        roi_corners=[
            _roi_json(name="roi_0", x1=0, y1=0, z1=0, x2=10, y2=10, z2=2),
        ],
        output_table_name="test_single_roi",
        overwrite=True,
    )
    container = ngio.open_ome_zarr_container(test_data_3d_path)
    table = container.get_generic_roi_table("test_single_roi")
    rois = table.rois()
    assert len(rois) == 1
    assert rois[0].name == "roi_0"


def test_roi_selection_multiple_rois(test_data_3d_path: Path) -> None:
    """Test storing multiple ROIs."""
    roi_selection_task(
        zarr_url=test_data_3d_path.as_posix(),
        roi_corners=[
            _roi_json(name="roi_0", x1=0, y1=0, z1=0, x2=10, y2=10, z2=2),
            _roi_json(name="roi_1", x1=20, y1=20, z1=0, x2=30, y2=30, z2=2),
        ],
        output_table_name="test_multi_roi",
        overwrite=True,
    )
    container = ngio.open_ome_zarr_container(test_data_3d_path)
    table = container.get_generic_roi_table("test_multi_roi")
    rois = table.rois()
    assert len(rois) == 2
    names = {r.name for r in rois}
    assert names == {"roi_0", "roi_1"}


def test_roi_selection_swapped_corners(test_data_3d_path: Path) -> None:
    """Test that swapped corners (x2 < x1) are handled correctly."""
    roi_selection_task(
        zarr_url=test_data_3d_path.as_posix(),
        roi_corners=[
            _roi_json(name="swapped", x1=10, y1=10, z1=2, x2=0, y2=0, z2=0),
        ],
        output_table_name="test_swapped",
        overwrite=True,
    )
    container = ngio.open_ome_zarr_container(test_data_3d_path)
    table = container.get_generic_roi_table("test_swapped")
    rois = table.rois()
    assert len(rois) == 1
    assert rois[0].get("x").length > 0
    assert rois[0].get("y").length > 0
    assert rois[0].get("z").length > 0


def test_roi_selection_overwrite_false_duplicate_name(test_data_3d_path: Path) -> None:
    """Test that overwrite=False raises when ROI name already exists in table."""
    roi_selection_task(
        zarr_url=test_data_3d_path.as_posix(),
        roi_corners=[
            _roi_json(name="roi_0", x1=0, y1=0, z1=0, x2=10, y2=10, z2=2),
        ],
        output_table_name="test_overwrite",
        overwrite=True,
    )
    with pytest.raises(FileExistsError, match="already exist"):
        roi_selection_task(
            zarr_url=test_data_3d_path.as_posix(),
            roi_corners=[
                _roi_json(name="roi_0", x1=0, y1=0, z1=0, x2=10, y2=10, z2=2),
            ],
            output_table_name="test_overwrite",
            overwrite=False,
        )


def test_roi_selection_append_new_rois(test_data_3d_path: Path) -> None:
    """Test that overwrite=False appends when ROI names are different."""
    roi_selection_task(
        zarr_url=test_data_3d_path.as_posix(),
        roi_corners=[
            _roi_json(name="roi_0", x1=0, y1=0, z1=0, x2=10, y2=10, z2=2),
        ],
        output_table_name="test_append",
        overwrite=True,
    )
    roi_selection_task(
        zarr_url=test_data_3d_path.as_posix(),
        roi_corners=[
            _roi_json(name="roi_1", x1=5, y1=5, z1=0, x2=15, y2=15, z2=2),
        ],
        output_table_name="test_append",
        overwrite=False,
    )
    container = ngio.open_ome_zarr_container(test_data_3d_path)
    table = container.get_generic_roi_table("test_append")
    rois = table.rois()
    assert len(rois) == 2
    names = {r.name for r in rois}
    assert names == {"roi_0", "roi_1"}


def test_roi_selection_overwrite_true(test_data_3d_path: Path) -> None:
    """Test that overwrite=True replaces the existing table."""
    roi_selection_task(
        zarr_url=test_data_3d_path.as_posix(),
        roi_corners=[
            _roi_json(name="roi_0", x1=0, y1=0, z1=0, x2=10, y2=10, z2=2),
        ],
        output_table_name="test_overwrite_ok",
        overwrite=True,
    )
    roi_selection_task(
        zarr_url=test_data_3d_path.as_posix(),
        roi_corners=[
            _roi_json(name="new_roi", x1=5, y1=5, z1=0, x2=15, y2=15, z2=2),
        ],
        output_table_name="test_overwrite_ok",
        overwrite=True,
    )
    container = ngio.open_ome_zarr_container(test_data_3d_path)
    table = container.get_generic_roi_table("test_overwrite_ok")
    rois = table.rois()
    assert len(rois) == 1
    assert rois[0].name == "new_roi"


def test_roi_selection_empty_list(test_data_3d_path: Path) -> None:
    """Test that an empty ROI list raises ValueError."""
    with pytest.raises(ValueError, match="No ROI corners provided"):
        roi_selection_task(
            zarr_url=test_data_3d_path.as_posix(),
            roi_corners=[],
            output_table_name="test_empty",
        )


def test_roi_selection_out_of_bounds_clamped(test_data_3d_path: Path, caplog) -> None:
    """Test that an ROI with in-bounds origin but overflowing extent is clamped."""
    # 3D fixture: 100x100 pixels at pixelsize 0.5 -> 50 physical units in x/y,
    # 10 z-slices at z_spacing 1.0 -> 10 physical units in z.
    with caplog.at_level("WARNING"):
        roi_selection_task(
            zarr_url=test_data_3d_path.as_posix(),
            roi_corners=[
                _roi_json(name="clamped", x1=0, y1=0, z1=0, x2=9999, y2=9999, z2=9999),
            ],
            output_table_name="test_clamped",
            overwrite=True,
        )
    assert "clamping length" in caplog.text
    container = ngio.open_ome_zarr_container(test_data_3d_path)
    table = container.get_generic_roi_table("test_clamped")
    roi = table.rois()[0]
    # Extent is clamped so the ROI ends exactly at the image bounds.
    assert roi.get("x").start + roi.get("x").length == pytest.approx(50)
    assert roi.get("y").start + roi.get("y").length == pytest.approx(50)
    assert roi.get("z").start + roi.get("z").length == pytest.approx(10)


def test_roi_selection_origin_out_of_bounds(test_data_3d_path: Path) -> None:
    """Test that an ROI whose origin is outside image bounds raises ValueError."""
    with pytest.raises(ValueError, match="origin is out of image bounds"):
        roi_selection_task(
            zarr_url=test_data_3d_path.as_posix(),
            roi_corners=[
                _roi_json(
                    name="oob_origin", x1=9000, y1=9000, z1=0, x2=9999, y2=9999, z2=2
                ),
            ],
            output_table_name="test_oob_origin",
            overwrite=True,
        )


def test_roi_selection_nonexistent_zarr(tmp_path: Path) -> None:
    """Test that a nonexistent zarr path raises FileNotFoundError."""
    fake_path = tmp_path / "nonexistent.zarr"
    with pytest.raises(FileNotFoundError):
        roi_selection_task(
            zarr_url=fake_path.as_posix(),
            roi_corners=[
                _roi_json(name="roi_0", x1=0, y1=0, z1=0, x2=10, y2=10, z2=2),
            ],
        )


def test_roi_selection_physical_coords(tmp_path: Path) -> None:
    """Test that physical XY coordinates are stored directly
    and Z indices are converted using a non-unit z_spacing."""
    zarr_url = tmp_path / "test_3d_zspacing.zarr"
    ngio.create_ome_zarr_from_array(
        store=str(zarr_url),
        array=np.zeros((1, 10, 100, 100)),
        pixelsize=0.5,
        z_spacing=5.0,
        axes_names="czyx",
        overwrite=True,
    )
    roi_selection_task(
        zarr_url=zarr_url.as_posix(),
        roi_corners=[
            _roi_json(name="phys_test", x1=0, y1=0, z1=0, x2=10, y2=20, z2=3),
        ],
        output_table_name="test_phys",
        overwrite=True,
    )
    container = ngio.open_ome_zarr_container(zarr_url)
    image = container.get_image(path="0")
    ps = image.pixel_size
    assert ps.z == pytest.approx(5.0)

    table = container.get_generic_roi_table("test_phys")
    roi = table.rois()[0]
    # X and Y are already physical from the viewer — stored as-is
    assert roi.get("x").length == pytest.approx(10)
    assert roi.get("y").length == pytest.approx(20)
    # Z is a slice index — multiplied by ps.z (5.0)
    assert roi.get("z").length == pytest.approx(4 * ps.z)


def test_roi_selection_2d(test_data_2d_path: Path) -> None:
    """Test ROI selection on a 2D-like image (Z axis with size 1)."""
    roi_selection_task(
        zarr_url=test_data_2d_path.as_posix(),
        roi_corners=[
            _roi_json(name="roi_2d", x1=0, y1=0, z1=0, z2=0, x2=10, y2=10),
        ],
        output_table_name="test_2d_roi",
        overwrite=True,
    )
    container = ngio.open_ome_zarr_container(test_data_2d_path)
    table = container.get_generic_roi_table("test_2d_roi")
    rois = table.rois()
    assert len(rois) == 1


def test_roi_selection_single_z_plane(test_data_3d_path: Path) -> None:
    """Test that z1==z2 produces a single-Z-plane ROI (length=1*ps.z)."""
    roi_selection_task(
        zarr_url=test_data_3d_path.as_posix(),
        roi_corners=[
            _roi_json(name="single_z", x1=0, y1=0, z1=0, x2=10, y2=10, z2=0),
        ],
        output_table_name="test_single_z",
        overwrite=True,
    )
    container = ngio.open_ome_zarr_container(test_data_3d_path)
    image = container.get_image(path="0")
    ps = image.pixel_size

    table = container.get_generic_roi_table("test_single_z")
    roi = table.rois()[0]
    assert roi.get("z").length == pytest.approx(1 * ps.z)
    assert roi.get("z").start == pytest.approx(0.0)
    # X and Y stored directly as physical
    assert roi.get("x").length == pytest.approx(10)
    assert roi.get("y").length == pytest.approx(10)


def test_roi_selection_3d_missing_z_raises(test_data_3d_path: Path) -> None:
    """Test that omitting Z on a 3D image raises ValueError."""
    with pytest.raises(ValueError, match="z1/z2 coordinates were not provided"):
        roi_selection_task(
            zarr_url=test_data_3d_path.as_posix(),
            roi_corners=[
                _roi_json(name="no_z", x1=0, y1=0, x2=10, y2=10),
            ],
            output_table_name="test_missing_z",
            overwrite=True,
        )


def test_roi_selection_t_ignored_no_t_axis(test_data_3d_path: Path, caplog) -> None:
    """Test that T values are ignored with warning on image without T axis."""
    with caplog.at_level("WARNING"):
        roi_selection_task(
            zarr_url=test_data_3d_path.as_posix(),
            roi_corners=[
                _roi_json(
                    name="t_ignored",
                    x1=0,
                    y1=0,
                    z1=0,
                    z2=2,
                    t1=0,
                    t2=3,
                    x2=10,
                    y2=10,
                ),
            ],
            output_table_name="test_t_ignored",
            overwrite=True,
        )
    assert "T values will be ignored" in caplog.text


def test_roi_selection_duplicate_names_raises(test_data_3d_path: Path) -> None:
    """Test that duplicate ROI names in input raise ValueError."""
    with pytest.raises(ValueError, match="Duplicate ROI names"):
        roi_selection_task(
            zarr_url=test_data_3d_path.as_posix(),
            roi_corners=[
                _roi_json(name="dup", x1=0, y1=0, z1=0, x2=10, y2=10, z2=2),
                _roi_json(name="dup", x1=5, y1=5, z1=0, x2=15, y2=15, z2=2),
            ],
            output_table_name="test_dup_names",
            overwrite=True,
        )
