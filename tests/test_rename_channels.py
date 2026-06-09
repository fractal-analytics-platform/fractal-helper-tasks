"""Tests for rename_channels task."""

import ngio
import pytest

from fractal_helper_tasks.rename_channels import rename_channels


@pytest.fixture
def two_zarrs(tmp_path):
    """Two synthetic OME-Zarr images with channels [DAPI, GFP, mCherry]."""
    zarr1 = str(tmp_path / "zarr1.zarr")
    zarr2 = str(tmp_path / "zarr2.zarr")
    for path in (zarr1, zarr2):
        ngio.create_synthetic_ome_zarr(
            store=path,
            shape=(3, 64, 64),
            axes_names=["c", "y", "x"],
            channels_meta=["DAPI", "GFP", "mCherry"],
            overwrite=True,
        )
    return zarr1, zarr2, str(tmp_path)


def test_basic_rename(two_zarrs):
    """All mapped channels are renamed in every image."""
    zarr1, zarr2, zarr_dir = two_zarrs

    rename_channels(
        zarr_urls=[zarr1, zarr2],
        zarr_dir=zarr_dir,
        channel_name_map={"DAPI": "HOECHST", "GFP": "Alexa488"},
    )

    for url in (zarr1, zarr2):
        labels = ngio.open_ome_zarr_container(url).get_image().channel_labels
        assert labels == ["HOECHST", "Alexa488", "mCherry"]


def test_partial_map_skips_missing(two_zarrs):
    """Map entries not matching any channel are silently skipped."""
    zarr1, zarr2, zarr_dir = two_zarrs

    rename_channels(
        zarr_urls=[zarr1, zarr2],
        zarr_dir=zarr_dir,
        channel_name_map={"DAPI": "HOECHST", "channel_99": "irrelevant"},
    )

    for url in (zarr1, zarr2):
        labels = ngio.open_ome_zarr_container(url).get_image().channel_labels
        assert labels == ["HOECHST", "GFP", "mCherry"]


def test_no_op_when_no_keys_match(two_zarrs):
    """Map with no matching keys leaves every image unchanged."""
    zarr1, zarr2, zarr_dir = two_zarrs
    original = ["DAPI", "GFP", "mCherry"]

    rename_channels(
        zarr_urls=[zarr1, zarr2],
        zarr_dir=zarr_dir,
        channel_name_map={"channel_0": "A", "channel_1": "B"},
    )

    for url in (zarr1, zarr2):
        labels = ngio.open_ome_zarr_container(url).get_image().channel_labels
        assert labels == original


def test_fail_on_missing_channels_raises(two_zarrs):
    """fail_on_missing_channels=True raises ValueError for absent channels."""
    zarr1, zarr2, zarr_dir = two_zarrs

    with pytest.raises(ValueError, match="channel_99"):
        rename_channels(
            zarr_urls=[zarr1, zarr2],
            zarr_dir=zarr_dir,
            channel_name_map={"DAPI": "HOECHST", "channel_99": "irrelevant"},
            fail_on_missing_channels=True,
        )
