# Copyright 2026 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
# Claude
"""Task to fix or rewrite omero channel metadata in OME-Zarr images."""

import logging
from typing import Annotated, Any, Literal

import numpy as np
import zarr
from ngio.ome_zarr_meta.ngio_specs import NgioColors
from pydantic import BaseModel, Field, model_validator, validate_call

logger = logging.getLogger("fix_omero_metadata")

# Defaults for optional omero channel fields when writing from scratch
# (same values as written by ngio)
REWRITE_CHANNEL_DEFAULTS = {"active": True, "coefficient": 1, "inverted": False}


class OmeroWindow(BaseModel):
    """Display range of an omero channel.

    Attributes:
        min: Minimum value of the data range (e.g. 0 for uint16).
        max: Maximum value of the data range (e.g. 65535 for uint16).
        start: Lower bound of the display range.
        end: Upper bound of the display range.
    """

    min: float | None = None
    max: float | None = None
    start: float | None = None
    end: float | None = None


class OmeroChannel(BaseModel):
    """Omero metadata of a single channel.

    All fields except the label are optional. Unset fields keep their existing
    value (when fixing metadata) or are set to defaults if they are required
    by the OME-Zarr spec (color & window). Other existing omero fields (e.g.
    `active`, `inverted`) are kept when fixing metadata.

    Attributes:
        label: Name of the channel. Used to match the channel to existing
            omero channels when fixing metadata.
        wavelength_id: Fractal-specific wavelength identifier
            (e.g. `A01_C01`). Only set if provided or already present.
        color: Color of the channel as a 6 character hex string
            (e.g. `00FF00`).
        window: Display range of the channel.
    """

    label: str
    wavelength_id: str | None = None
    color: str | None = Field(default=None, pattern=r"^[0-9a-fA-F]{6}$")
    window: OmeroWindow = Field(default_factory=OmeroWindow)


def _check_unique_labels(channels: list[OmeroChannel]) -> None:
    labels = [channel.label for channel in channels]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Channel labels must be unique, got {labels}.")


class FixOmeroMetadata(BaseModel):
    """Fix existing omero metadata.

    Keeps all existing omero metadata, fills in missing required fields
    (color & window) with defaults and applies the provided overrides.

    Attributes:
        mode: Fix existing omero metadata.
        channels: Optional overrides for existing channels, matched by label.
            Overrides for labels not present in an image are skipped with a
            warning.
    """

    mode: Literal["Fix omero metadata"] = "Fix omero metadata"
    channels: list[OmeroChannel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_labels(self):
        _check_unique_labels(self.channels)
        return self


class RewriteOmeroMetadata(BaseModel):
    """Write new omero metadata from scratch.

    Replaces any existing omero metadata. Required fields that are not
    provided (color & window) are set to defaults. Channels are written with
    `active: true`, `coefficient: 1` and `inverted: false`.

    Attributes:
        mode: Rewrite omero metadata from scratch.
        channels: Metadata for each channel, in the order of the channels in
            the image. Must have as many entries as the image has channels.
    """

    mode: Literal["Rewrite omero metadata"] = "Rewrite omero metadata"
    channels: list[OmeroChannel] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_labels(self):
        _check_unique_labels(self.channels)
        return self


def _default_window(dtype: np.dtype) -> dict[str, float]:
    """Default window spanning the full range of the data type."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        dtype_min, dtype_max = int(info.min), int(info.max)
    else:
        # Floating point images have no meaningful dtype range
        dtype_min, dtype_max = 0, 1
    return {"min": dtype_min, "max": dtype_max, "start": dtype_min, "end": dtype_max}


def _build_channel(
    existing: dict[str, Any],
    override: OmeroChannel | None,
    label: str,
    default_window: dict[str, float],
) -> dict[str, Any]:
    """Merge existing channel metadata, user overrides and defaults."""
    channel = dict(existing)
    channel["label"] = label
    window = {**default_window, **existing.get("window", {})}
    if override is not None:
        channel.update(override.model_dump(exclude_none=True, exclude={"window"}))
        window.update(override.window.model_dump(exclude_none=True))
    channel["window"] = window
    if "color" not in channel:
        channel["color"] = NgioColors.semi_random_pick(label.lower()).value
    return channel


def _get_highest_res_array(
    group: zarr.Group, multiscales: list[dict[str, Any]]
) -> zarr.Array:
    """Get the highest-resolution array of an OME-Zarr image."""
    path = multiscales[0]["datasets"][0]["path"]
    array = group[path]
    if not isinstance(array, zarr.Array):
        raise ValueError(f"Expected an array at '{path}', found a group.")
    return array


def _num_channels(axes: list[dict[str, Any]], shape: tuple[int, ...]) -> int:
    """Size of the channel axis (1 if there is none)."""
    for idx, axis in enumerate(axes):
        if axis.get("type") == "channel" or axis.get("name") == "c":
            return shape[idx]
    return 1


def _update_omero(zarr_url: str, mode: FixOmeroMetadata | RewriteOmeroMetadata) -> None:
    """Fix or rewrite the omero metadata of a single OME-Zarr image."""
    group = zarr.open_group(zarr_url, mode="r+")
    # Raw JSON metadata, potentially invalid, so it is handled untyped
    attrs: dict[str, Any] = group.attrs.asdict()

    # OME-Zarr v0.5 nests the metadata under the "ome" key
    is_v05 = "ome" in attrs
    ome_attrs: dict[str, Any] = attrs["ome"] if is_v05 else attrs
    multiscales = ome_attrs["multiscales"]
    array = _get_highest_res_array(group, multiscales)
    default_window = _default_window(array.dtype)

    if isinstance(mode, RewriteOmeroMetadata):
        num_channels = _num_channels(multiscales[0]["axes"], array.shape)
        if len(mode.channels) != num_channels:
            raise ValueError(
                f"{zarr_url} has {num_channels} channels, but "
                f"{len(mode.channels)} channels were provided."
            )
        channels = [
            _build_channel(
                REWRITE_CHANNEL_DEFAULTS, override, override.label, default_window
            )
            for override in mode.channels
        ]
        omero = {"channels": channels}
    else:
        if "omero" not in ome_attrs:
            raise ValueError(
                f"{zarr_url} has no omero metadata to fix. Use the "
                "'Rewrite omero metadata' mode to create it from scratch."
            )
        omero = ome_attrs["omero"]
        existing_channels = omero.get("channels", [])
        overrides = {channel.label: channel for channel in mode.channels}
        existing_labels = [c.get("label") for c in existing_channels]
        # Overrides are matched by label, so existing labels must be unique
        if None in existing_labels or len(set(existing_labels)) != len(existing_labels):
            raise ValueError(
                f"{zarr_url} has omero channels with missing or duplicate "
                f"labels: {existing_labels}. Use the 'Rewrite omero metadata' "
                "mode to set the channel metadata from scratch."
            )
        missing = [label for label in overrides if label not in existing_labels]
        if missing:
            logger.warning(
                f"{zarr_url}: channels not found (skipped): {missing}. "
                f"Available channels: {existing_labels}."
            )
        omero["channels"] = [
            _build_channel(
                existing,
                overrides.get(existing["label"]),
                existing["label"],
                default_window,
            )
            for existing in existing_channels
        ]

    logger.info(f"{zarr_url}: writing omero metadata {omero}.")
    if is_v05:
        ome_attrs["omero"] = omero
        group.attrs["ome"] = ome_attrs
    else:
        group.attrs["omero"] = omero


@validate_call
def fix_omero_metadata(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    mode: Annotated[
        FixOmeroMetadata | RewriteOmeroMetadata, Field(discriminator="mode")
    ] = Field(default_factory=FixOmeroMetadata),
) -> None:
    """Fix or rewrite the omero channel metadata of OME-Zarr images.

    Works directly on the Zarr attributes, so it can repair images that ngio
    fails to open because of invalid omero metadata (e.g. channels missing
    the `window` field). Supports OME-Zarr v0.4 and v0.5.

    Required fields that are missing are set to defaults: `window` spans the
    range of the image data type (e.g. 0 - 65535 for uint16, 0 - 1 for float
    images) and `color` is picked based on the channel label.

    Args:
        zarr_urls: Paths to all OME-Zarr images to be processed.
            (standard argument for Fractal non-parallel tasks).
        zarr_dir: Path to the directory containing the OME-Zarr images.
            (standard argument for Fractal non-parallel tasks).
        mode: "Fix omero metadata" keeps existing metadata, fills in missing
            required fields and applies optional per-channel overrides
            (matched by label). Requires existing omero metadata.
            "Rewrite omero metadata" replaces the omero metadata with the
            provided channels (works with or without existing omero metadata).
    """
    logger.info(
        f"Running `fix_omero_metadata` on {len(zarr_urls)} images "
        f"in mode '{mode.mode}'."
    )
    for zarr_url in zarr_urls:
        _update_omero(zarr_url, mode)
    logger.info("Finished `fix_omero_metadata`.")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=fix_omero_metadata)
