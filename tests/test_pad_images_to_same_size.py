"""Tests for pad_images_to_same_size task."""

from pathlib import Path

import ngio
from ngio.ome_zarr_meta.ngio_specs._ngio_hcs import ImageInWellPath

from fractal_helper_tasks.pad_images_to_same_size import pad_images_to_same_size


def create_plate_with_images(
    plate_path: Path,
    well_images: list[tuple[str, str, str, tuple[int, ...]]],
    axes_names: str = "czyx",
) -> list[str]:
    """Create an HCS plate with images of given shapes.

    Args:
        plate_path: Path for the plate zarr.
        well_images: List of (row, column, image_path, shape) tuples.
        axes_names: Axes names for the images.

    Returns:
        List of zarr_url strings for the created images.
    """
    plate = ngio.create_empty_plate(
        store=str(plate_path),
        name=plate_path.stem,
        images=[
            ImageInWellPath(row=row, column=col, path=path)
            for row, col, path, _ in well_images
        ],
        overwrite=True,
    )

    zarr_urls = []
    for row, col, path, shape in well_images:
        store = plate.get_image_store(row=row, column=col, image_path=path)
        ngio.create_synthetic_ome_zarr(
            store=store,
            shape=shape,
            axes_names=axes_names,
            overwrite=True,
        )
        zarr_urls.append(str(plate_path / row / col / path))

    return zarr_urls


def test_pad_single_plate(tmp_path: Path):
    """Two wells in one plate: smaller one gets padded to match the larger."""
    plate_path = tmp_path / "plate.zarr"

    zarr_urls = create_plate_with_images(
        plate_path,
        [
            ("B", "03", "0", (1, 10, 50, 50)),
            ("B", "04", "0", (1, 10, 100, 100)),
        ],
    )

    pad_images_to_same_size(zarr_urls=zarr_urls, zarr_dir=str(plate_path.parent))

    for url in zarr_urls:
        container = ngio.open_ome_zarr_container(url)
        assert container.get_image().shape == (1, 10, 100, 100), (
            f"{url} has unexpected shape {container.get_image().shape}"
        )
        for label_name in container.list_labels():
            lbl_shape = container.get_label(label_name).shape
            assert lbl_shape[-2] == 100, (
                f"{url} label '{label_name}' y-dim: {lbl_shape}"
            )
            assert lbl_shape[-1] == 100, (
                f"{url} label '{label_name}' x-dim: {lbl_shape}"
            )

    # Verify all pyramid levels also match
    levels = ngio.open_ome_zarr_container(zarr_urls[0]).level_paths
    for level in levels:
        shapes = [
            ngio.open_ome_zarr_container(url).get_image(path=level).shape
            for url in zarr_urls
        ]
        assert len(set(shapes)) == 1, (
            f"Level {level}: shapes differ across wells: {shapes}"
        )


def test_pad_by_plate(tmp_path: Path):
    """Two plates with different max sizes; pad_by_plate keeps them separate."""
    plate_a_path = tmp_path / "plate_a.zarr"
    plate_b_path = tmp_path / "plate_b.zarr"

    urls_a = create_plate_with_images(
        plate_a_path,
        [
            ("B", "03", "0", (1, 5, 80, 90)),
            ("B", "04", "0", (1, 5, 100, 100)),
        ],
    )
    urls_b = create_plate_with_images(
        plate_b_path,
        [
            ("B", "03", "0", (1, 8, 120, 130)),
            ("B", "04", "0", (1, 8, 140, 150)),
        ],
    )

    pad_images_to_same_size(
        zarr_urls=urls_a + urls_b,
        zarr_dir=str(tmp_path),
        pad_by_plate=True,
    )

    # Plate A: both wells should be padded to plate A's max (1, 5, 100, 100)
    for url in urls_a:
        shape = ngio.open_ome_zarr_container(url).get_image().shape
        assert shape == (1, 5, 100, 100), f"plate_a {url}: got {shape}"

    # Plate B: both wells should be padded to plate B's max (1, 8, 140, 150)
    for url in urls_b:
        shape = ngio.open_ome_zarr_container(url).get_image().shape
        assert shape == (1, 8, 140, 150), f"plate_b {url}: got {shape}"
