"""Tests for fix_omero_metadata task."""

from typing import Any

import ngio
import pytest
import zarr
from ngio import NgffVersions
from ngio.utils import NgioValidationError
from pydantic import ValidationError

from fractal_helper_tasks.fix_omero_metadata import (
    FixOmeroMetadata,
    OmeroChannel,
    OmeroWindow,
    RewriteOmeroMetadata,
    fix_omero_metadata,
)

# Invalid omero metadata as written by some converters: no window field
INVALID_CHANNELS = [
    {
        "active": True,
        "coefficient": 1,
        "color": "3342ff",
        "inverted": False,
        "label": "DAPI",
        "wavelength_id": "A01_C01",
    },
    {
        "active": True,
        "coefficient": 1,
        "color": "33ff42",
        "inverted": False,
        "label": "AF488",
        "wavelength_id": "A02_C02",
    },
]

# zarr_dir is a standard Fractal argument that the task doesn't use
ZARR_DIR = "/unused"

UINT16_WINDOW = {"min": 0, "max": 65535, "start": 0, "end": 65535}
REWRITE_DEFAULTS = {"active": True, "coefficient": 1, "inverted": False}


def _create_zarr(path, ngff_version: NgffVersions = "0.4") -> str:
    url = str(path / "img.zarr")
    ngio.create_synthetic_ome_zarr(
        store=url,
        shape=(2, 64, 64),
        axes_names=["c", "y", "x"],
        channels_meta=["DAPI", "AF488"],
        ngff_version=ngff_version,
        overwrite=True,
    )
    return url


def _set_omero(url: str, omero: dict | None) -> None:
    """Overwrite (or remove if None) the raw omero metadata of an image."""
    group = zarr.open_group(url, mode="r+")
    attrs: dict[str, Any] = group.attrs.asdict()
    is_v05 = "ome" in attrs
    ome_attrs: dict[str, Any] = attrs["ome"] if is_v05 else attrs
    ome_attrs.pop("omero", None)
    if omero is not None:
        ome_attrs["omero"] = omero
    if is_v05:
        group.attrs["ome"] = ome_attrs
    else:
        group.attrs.put(ome_attrs)


def _get_omero(url: str) -> dict[str, Any]:
    """Read the raw omero metadata of an image (v0.4 or v0.5)."""
    attrs: dict[str, Any] = zarr.open_group(url).attrs.asdict()
    return attrs.get("ome", attrs)["omero"]


def _get_channels(url: str) -> list[dict[str, Any]]:
    return _get_omero(url)["channels"]


@pytest.fixture(params=["0.4", "0.5"])
def invalid_zarr(tmp_path, request) -> str:
    url = _create_zarr(tmp_path, request.param)
    _set_omero(url, {"channels": INVALID_CHANNELS, "name": "TBD", "id": 1})
    with pytest.raises(NgioValidationError):
        ngio.open_ome_zarr_container(url)
    return url


def test_fix_adds_missing_windows(invalid_zarr):
    """Default mode adds windows and keeps all existing metadata."""
    fix_omero_metadata(zarr_urls=[invalid_zarr], zarr_dir=ZARR_DIR)

    image = ngio.open_ome_zarr_container(invalid_zarr).get_image()
    assert image.channel_labels == ["DAPI", "AF488"]
    channels = _get_channels(invalid_zarr)
    for channel, original in zip(channels, INVALID_CHANNELS, strict=True):
        assert channel == {**original, "window": UINT16_WINDOW}


def test_fix_multiple_images(tmp_path):
    """All images passed to the task are processed."""
    urls = []
    for name in ("a", "b"):
        (tmp_path / name).mkdir()
        url = _create_zarr(tmp_path / name)
        _set_omero(url, {"channels": INVALID_CHANNELS})
        urls.append(url)

    fix_omero_metadata(zarr_urls=urls, zarr_dir=str(tmp_path))

    for url in urls:
        ngio.open_ome_zarr_container(url)
        for channel in _get_channels(url):
            assert channel["window"] == UINT16_WINDOW


def test_fix_keeps_other_omero_fields(invalid_zarr):
    """Top-level omero fields other than channels are kept."""
    fix_omero_metadata(zarr_urls=[invalid_zarr], zarr_dir=ZARR_DIR)
    omero = _get_omero(invalid_zarr)
    assert omero["name"] == "TBD"
    assert omero["id"] == 1


def test_fix_with_overrides(invalid_zarr):
    """Overrides are matched by label, unmatched labels are skipped."""
    fix_omero_metadata(
        zarr_urls=[invalid_zarr],
        zarr_dir=ZARR_DIR,
        mode=FixOmeroMetadata(
            channels=[
                OmeroChannel(
                    label="AF488",
                    color="00FF00",
                    window=OmeroWindow(start=100, end=2000),
                ),
                OmeroChannel(label="not_in_image", color="FF0000"),
            ]
        ),
    )

    dapi, af488 = _get_channels(invalid_zarr)
    assert dapi == {**INVALID_CHANNELS[0], "window": UINT16_WINDOW}
    assert af488["color"] == "00FF00"
    assert af488["window"] == {"min": 0, "max": 65535, "start": 100, "end": 2000}
    assert af488["wavelength_id"] == "A02_C02"
    ngio.open_ome_zarr_container(invalid_zarr)


def test_fix_partial_window_and_missing_color(tmp_path):
    """Existing window values are kept, only missing ones are defaulted."""
    url = _create_zarr(tmp_path)
    _set_omero(
        url,
        {
            "channels": [
                {"label": "DAPI", "window": {"start": 10, "end": 500}},
                {"label": "AF488", "color": "00FF00"},
            ]
        },
    )

    fix_omero_metadata(zarr_urls=[url], zarr_dir=ZARR_DIR)

    dapi, af488 = _get_channels(url)
    assert dapi["window"] == {"min": 0, "max": 65535, "start": 10, "end": 500}
    # Missing color gets a default based on the label
    assert dapi["color"] == "0000FF"
    assert af488["window"] == UINT16_WINDOW
    # Optional fields are not added if not present or provided
    assert "wavelength_id" not in dapi
    assert "active" not in af488
    ngio.open_ome_zarr_container(url)


def test_fix_sets_wavelength_id_if_provided(invalid_zarr):
    """wavelength_id is overwritten if provided, kept otherwise."""
    fix_omero_metadata(
        zarr_urls=[invalid_zarr],
        zarr_dir=ZARR_DIR,
        mode=FixOmeroMetadata(
            channels=[OmeroChannel(label="DAPI", wavelength_id="A05_C05")]
        ),
    )
    dapi, af488 = _get_channels(invalid_zarr)
    assert dapi["wavelength_id"] == "A05_C05"
    assert af488["wavelength_id"] == "A02_C02"


def test_fix_without_omero_fails(tmp_path):
    """Fix mode needs existing omero metadata."""
    url = _create_zarr(tmp_path)
    _set_omero(url, None)
    with pytest.raises(ValueError, match="Rewrite omero metadata"):
        fix_omero_metadata(zarr_urls=[url], zarr_dir=ZARR_DIR)


@pytest.mark.parametrize(
    "channels",
    [
        [{"label": "DAPI"}, {"color": "00FF00"}],
        [{"label": "DAPI"}, {"label": "DAPI"}],
    ],
)
def test_fix_missing_or_duplicate_labels_fails(tmp_path, channels):
    """Fix mode can't match overrides without unique existing labels."""
    url = _create_zarr(tmp_path)
    _set_omero(url, {"channels": channels})
    with pytest.raises(ValueError, match="Rewrite omero metadata"):
        fix_omero_metadata(zarr_urls=[url], zarr_dir=ZARR_DIR)


@pytest.mark.parametrize("has_omero", [True, False])
def test_rewrite(tmp_path, has_omero):
    """Rewrite mode replaces the omero metadata, with or without existing."""
    url = _create_zarr(tmp_path)
    _set_omero(url, {"channels": INVALID_CHANNELS} if has_omero else None)

    fix_omero_metadata(
        zarr_urls=[url],
        zarr_dir=ZARR_DIR,
        mode=RewriteOmeroMetadata(
            channels=[
                OmeroChannel(label="Hoechst", wavelength_id="A01_C01"),
                OmeroChannel(label="GFP", color="00FF00", window=OmeroWindow(end=1000)),
            ]
        ),
    )

    hoechst, gfp = _get_channels(url)
    # Default color is picked by ngio based on the label
    assert len(hoechst.pop("color")) == 6
    assert hoechst == {
        "label": "Hoechst",
        "wavelength_id": "A01_C01",
        "window": UINT16_WINDOW,
        **REWRITE_DEFAULTS,
    }
    assert gfp == {
        **REWRITE_DEFAULTS,
        "label": "GFP",
        "color": "00FF00",
        "window": {"min": 0, "max": 65535, "start": 0, "end": 1000},
    }
    image = ngio.open_ome_zarr_container(url).get_image()
    assert image.channel_labels == ["Hoechst", "GFP"]


def test_rewrite_wrong_number_of_channels(tmp_path):
    url = _create_zarr(tmp_path)
    with pytest.raises(ValueError, match="has 2 channels"):
        fix_omero_metadata(
            zarr_urls=[url],
            zarr_dir=ZARR_DIR,
            mode=RewriteOmeroMetadata(channels=[OmeroChannel(label="DAPI")]),
        )


def test_mode_from_dict(invalid_zarr):
    """The mode is selected via the discriminator when passed as JSON."""
    fix_omero_metadata(
        zarr_urls=[invalid_zarr],
        zarr_dir=ZARR_DIR,
        # Raw dict on purpose: validate_call parses it, like Fractal's JSON args
        mode={  # pyright: ignore[reportArgumentType]
            "mode": "Rewrite omero metadata",
            "channels": [{"label": "A"}, {"label": "B"}],
        },
    )
    assert [c["label"] for c in _get_channels(invalid_zarr)] == ["A", "B"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"channels": [OmeroChannel(label="A"), OmeroChannel(label="A")]},
        {"channels": [{"label": "A", "color": "#00FF00"}]},
    ],
)
def test_invalid_inputs(kwargs):
    """Duplicate labels and invalid colors are rejected."""
    with pytest.raises(ValidationError):
        FixOmeroMetadata(**kwargs)
