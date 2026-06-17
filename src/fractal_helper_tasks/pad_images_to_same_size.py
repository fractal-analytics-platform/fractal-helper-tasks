# Copyright 2025 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
"""Task to pad all images to the same size by extending zarr metadata."""

import logging
import os
from pathlib import Path

import ngio
import zarr
from ngio.utils import NgioValidationError
from pydantic import validate_call

logger = logging.getLogger("pad_images_to_same_size")


def _check_is_hcs_plate(plate_root: str) -> None:
    """Raise ValueError if plate_root is not an HCS plate."""
    try:
        ngio.open_ome_zarr_plate(plate_root)
    except NgioValidationError as e:
        raise ValueError(
            f"The path '{plate_root}' does not appear to be an HCS plate. "
            "`pad_by_plate=True` requires all images to belong to an HCS plate. "
            f"Original error: {e}"
        ) from e


def _pad_image_labels(
    container,
    zarr_url: str,
    level_paths: list[str],
    spatial_targets_per_level: list[dict[str, int]],
) -> None:
    """Pad all labels in an image to match the padded image's spatial dimensions.

    Args:
        container: Open ngio OmeZarrContainer for the image.
        zarr_url: Path to the OME-Zarr image.
        level_paths: Pyramid level paths (e.g. ['0', '1', '2']).
        spatial_targets_per_level: Per-level dicts mapping axis name to target
            size, e.g. [{"z": 10, "y": 100, "x": 100}, ...].
    """
    label_names = container.list_labels()
    if not label_names:
        return

    img_pixel_size = container.get_image().pixel_size

    for label_name in label_names:
        label = container.get_label(label_name)

        lbl_pixel_size = label.pixel_size
        if (
            lbl_pixel_size.x != img_pixel_size.x
            or lbl_pixel_size.y != img_pixel_size.y
            or lbl_pixel_size.z != img_pixel_size.z
        ):
            raise ValueError(
                f"Label '{label_name}' in '{zarr_url}' has pixel size "
                f"{lbl_pixel_size} which does not match the image pixel size "
                f"{img_pixel_size}. Cannot pad label."
            )

        if len(label.meta.paths) != len(level_paths):
            raise ValueError(
                f"Label '{label_name}' in '{zarr_url}' has "
                f"{len(label.meta.paths)} pyramid levels, but the image has "
                f"{len(level_paths)}. Cannot pad label."
            )

        logger.info(f"{zarr_url}: padding label '{label_name}'.")
        for level_path, spatial_target in zip(
            level_paths, spatial_targets_per_level, strict=True
        ):
            lbl_at_level = container.get_label(label_name, path=level_path)
            current = list(lbl_at_level.shape)
            lbl_axes = lbl_at_level.axes_handler
            for axis, size in spatial_target.items():
                lbl_idx = lbl_axes.get_index(axis)
                if lbl_idx is not None:
                    current[lbl_idx] = size
            new_label_shape = tuple(current)
            array_path = os.path.join(zarr_url, "labels", label_name, level_path)
            zarr.open(array_path, mode="r+").resize(new_label_shape)
            logger.info(
                f"  Label '{label_name}' level {level_path}: "
                f"resized to {new_label_shape}"
            )


def _pad_group(zarr_urls: list[str], pad_labels: bool = True) -> None:
    """Pad all images in a group to the per-axis maximum size."""
    # Collect shapes at every pyramid level for every image
    containers = [ngio.open_ome_zarr_container(url) for url in zarr_urls]
    level_paths_per_image = [c.level_paths for c in containers]

    # Verify all images have the same number of pyramid levels
    n_levels = len(level_paths_per_image[0])
    for url, paths in zip(zarr_urls, level_paths_per_image, strict=True):
        if len(paths) != n_levels:
            raise ValueError(
                f"Image '{url}' has {len(paths)} pyramid levels, but other "
                f"images in the group have {n_levels}. All images must have "
                "the same number of pyramid levels."
            )

    # For each pyramid level, find the per-axis max shape across all images
    level_paths = level_paths_per_image[0]
    target_shapes_per_level: list[tuple[int, ...]] = []
    for level_path in level_paths:
        shapes_at_level = [c.get_image(path=level_path).shape for c in containers]
        # Max per axis independently
        target = tuple(
            max(s[i] for s in shapes_at_level) for i in range(len(shapes_at_level[0]))
        )
        target_shapes_per_level.append(target)

    # Resize each image that is smaller than the target at level 0
    target_shape_l0 = target_shapes_per_level[0]
    for url, container in zip(zarr_urls, containers, strict=True):
        img_l0 = container.get_image()
        current_shape_l0 = img_l0.shape
        axes_handler = img_l0.axes_handler

        z_idx = axes_handler.get_index("z")
        y_idx = axes_handler.get_index("y")
        x_idx = axes_handler.get_index("x")

        needs_padding = any(
            idx is not None and current_shape_l0[idx] < target_shape_l0[idx]
            for idx in [z_idx, y_idx, x_idx]
        )
        if not needs_padding:
            logger.info(f"{url}: already at target size, skipping.")
            continue

        logger.info(f"{url}: padding from {current_shape_l0} to {target_shape_l0}.")

        for level_path, target_shape in zip(
            level_paths, target_shapes_per_level, strict=True
        ):
            img_at_level = container.get_image(path=level_path)
            current = list(img_at_level.shape)

            for idx, _axis in [(z_idx, "z"), (y_idx, "y"), (x_idx, "x")]:
                if idx is not None:
                    current[idx] = target_shape[idx]

            new_shape = tuple(current)
            array_path = os.path.join(url, level_path)
            zarr.open(array_path, mode="r+").resize(new_shape)
            logger.info(f"  Level {level_path}: resized to {new_shape}")

        if pad_labels:
            spatial_targets = []
            for target_shape in target_shapes_per_level:
                spatial: dict[str, int] = {}
                for idx, axis in [(z_idx, "z"), (y_idx, "y"), (x_idx, "x")]:
                    if idx is not None:
                        spatial[axis] = target_shape[idx]
                spatial_targets.append(spatial)
            _pad_image_labels(container, url, level_paths, spatial_targets)


@validate_call
def pad_images_to_same_size(
    *,
    zarr_urls: list[str],
    zarr_dir: str,
    pad_by_plate: bool = False,
    pad_labels: bool = True,
) -> None:
    """Pad all images to the same size by extending zarr array metadata.

    Extends the shape metadata in the zarr arrays so that all images report
    the same size. No data is copied; regions outside the original array
    return the fill value (0) when read.

    Args:
        zarr_urls: Paths to all OME-Zarr images to be processed.
            (standard argument for Fractal non-parallel tasks).
        zarr_dir: Path to the directory containing the OME-Zarr images.
            (standard argument for Fractal non-parallel tasks).
        pad_by_plate: If True, images are grouped by their parent HCS plate
            and padded to the maximum size within each plate separately.
            All images must belong to an HCS plate when this is enabled.
            If False, all images are padded to the global maximum size.
        pad_labels: If True, label images within each OME-Zarr are also
            padded to match the padded image's spatial dimensions. Labels
            must have the same pixel size and number of pyramid levels as
            the image; a ValueError is raised otherwise.
    """
    logger.info(
        f"Running `pad_images_to_same_size` on {len(zarr_urls)} images, "
        f"{pad_by_plate=}, {pad_labels=}."
    )

    if pad_by_plate:
        # Group by plate root (3 levels up from image: image → col → row → plate)
        groups: dict[str, list[str]] = {}
        for url in zarr_urls:
            plate_root = str(Path(url.rstrip("/")).parents[2])
            groups.setdefault(plate_root, []).append(url)

        for plate_root, group_urls in groups.items():
            logger.info(
                f"Processing plate '{plate_root}' with {len(group_urls)} images."
            )
            _check_is_hcs_plate(plate_root)
            _pad_group(group_urls, pad_labels)
    else:
        _pad_group(zarr_urls, pad_labels)

    logger.info("Finished `pad_images_to_same_size`.")


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(task_function=pad_images_to_same_size)
