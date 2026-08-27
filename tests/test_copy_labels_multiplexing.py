"""Tests for copy_labels_to_multiplexing_acquisitions compound task."""

from pathlib import Path

import ngio
import pytest
from ngio import ImageInWellPath

from fractal_helper_tasks.copy_labels_multiplexing import copy_labels_multiplexing
from fractal_helper_tasks.copy_labels_multiplexing_init import (
    copy_labels_multiplexing_init,
)

# Wells and acquisitions used across tests
WELLS_ACQS = [
    ("B", "03", "0", 0),
    ("B", "03", "1", 1),
    ("B", "03", "2", 2),
    ("B", "04", "0", 0),
    ("B", "04", "1", 1),
    ("B", "04", "2", 2),
]


def _create_multiplexing_plate(plate_path: Path) -> tuple[list[str], list[str]]:
    """Create a plate with 2 wells x 3 acquisitions, each with synthetic images.

    Returns:
        zarr_urls: All image URLs in the plate.
        label_names: Labels present in the synthetic images (from acquisition 0).
    """
    plate = ngio.create_empty_plate(
        store=str(plate_path),
        name=plate_path.stem,
        images=[
            ImageInWellPath(row=row, column=col, path=path, acquisition_id=acq_id)
            for row, col, path, acq_id in WELLS_ACQS
        ],
        overwrite=True,
    )

    zarr_urls = []
    for row, col, path, _ in WELLS_ACQS:
        store = plate.get_image_store(row=row, column=col, image_path=path)
        ngio.create_synthetic_ome_zarr(
            store=store,
            shape=(1, 64, 64),
            axes_names=["c", "y", "x"],
            overwrite=True,
        )
        zarr_urls.append(str(plate_path / row / col / path))

    # Collect label names from any reference (acquisition 0) image
    ref_url = str(plate_path / "B" / "03" / "0")
    label_names = ngio.open_ome_zarr_container(ref_url).list_labels()
    return zarr_urls, label_names


@pytest.fixture
def plate_with_labels_deleted(tmp_path: Path):
    """Plate where acquisitions 1 and 2 have had their labels removed."""
    plate_path = tmp_path / "plate.zarr"
    zarr_urls, label_names = _create_multiplexing_plate(plate_path)

    # Delete labels from acquisitions 1 and 2
    for row, col, path, acq_id in WELLS_ACQS:
        if acq_id in (1, 2):
            container = ngio.open_ome_zarr_container(str(plate_path / row / col / path))
            for name in list(container.list_labels()):
                container.delete_label(name, missing_ok=True)

    # Verify they are gone
    for row, col, path, acq_id in WELLS_ACQS:
        if acq_id in (1, 2):
            remaining = ngio.open_ome_zarr_container(
                str(plate_path / row / col / path)
            ).list_labels()
            assert remaining == [], (
                f"Expected no labels in {row}/{col}/{path}, got {remaining}"
            )

    return zarr_urls, label_names, str(tmp_path)


def test_copy_labels_restores_all_acquisitions(plate_with_labels_deleted):
    """Labels copied from acquisition 0 should appear in acquisitions 1 and 2."""
    zarr_urls, label_names, zarr_dir = plate_with_labels_deleted

    result = copy_labels_multiplexing_init(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        reference_acquisition=0,
        label_names=label_names,
    )

    # 2 wells x 2 non-reference acquisitions = 4 compute jobs
    assert len(result["parallelization_list"]) == 4

    for item in result["parallelization_list"]:
        copy_labels_multiplexing(
            zarr_url=item["zarr_url"],
            init_args=item["init_args"],
        )

    # All acquisitions in both wells should now have all labels
    for row, col, path, _acq_id in WELLS_ACQS:
        url = str(Path(zarr_dir) / "plate.zarr" / row / col / path)
        container = ngio.open_ome_zarr_container(url)
        for label_name in label_names:
            assert label_name in container.list_labels(), (
                f"Label '{label_name}' missing from {url}"
            )


def test_copy_labels_only_targets_non_reference(plate_with_labels_deleted):
    """Reference acquisition (0) should not appear as a target."""
    zarr_urls, label_names, zarr_dir = plate_with_labels_deleted

    result = copy_labels_multiplexing_init(
        zarr_urls=zarr_urls,
        zarr_dir=zarr_dir,
        reference_acquisition=0,
        label_names=label_names,
    )

    target_urls = [item["zarr_url"] for item in result["parallelization_list"]]
    ref_urls = [url for url in zarr_urls if url.endswith("/0")]
    for ref_url in ref_urls:
        assert ref_url not in target_urls, (
            f"Reference URL {ref_url} should not appear as a target"
        )
