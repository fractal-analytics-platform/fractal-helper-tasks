"""Tests for delete_images_labels_tables task."""

from pathlib import Path

import ngio
import pandas as pd
import pytest
from ngio import ImageInWellPath, create_empty_plate
from ngio.tables import FeatureTable

from fractal_helper_tasks.delete_images_labels_tables import (
    delete_images_labels_tables,
)


def _add_feature_table(container, name: str, label_name: str) -> None:
    df = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.Index([1, 2], name="label"),
    )
    container.add_table(
        name=name,
        table=FeatureTable(table_data=df, reference_label=label_name),
        overwrite=True,
    )


def _create_image(zarr_url: str):
    """Create a synthetic OME-Zarr (includes nuclei and nuclei_mask labels)."""
    return ngio.create_synthetic_ome_zarr(
        store=zarr_url,
        shape=(1, 64, 64),
        axes_names=["c", "y", "x"],
        overwrite=True,
    )


@pytest.fixture
def two_zarrs(tmp_path: Path):
    """
    Image 1 (zarr1): labels=[nuclei, nuclei_mask], tables=[table_a, table_b]
    Image 2 (zarr2): labels=[nuclei, nuclei_mask], tables=[table_b, table_c]
    """
    zarr1 = str(tmp_path / "img1.zarr")
    zarr2 = str(tmp_path / "img2.zarr")

    # Image 1: synthetic (has nuclei + nuclei_mask labels by default) + 2 tables
    c1 = _create_image(zarr1)
    _add_feature_table(c1, "table_a", "nuclei")
    _add_feature_table(c1, "table_b", "nuclei")

    # Image 2: synthetic only (nuclei + nuclei_mask labels) + 2 different tables
    c2 = _create_image(zarr2)
    _add_feature_table(c2, "table_b", "nuclei")
    _add_feature_table(c2, "table_c", "nuclei")

    return zarr1, zarr2


def test_delete_partial(two_zarrs, tmp_path):
    """Delete nuclei_mask label and table_a/table_c; verify remaining content."""
    zarr1, zarr2 = two_zarrs

    delete_images_labels_tables(
        zarr_urls=[zarr1, zarr2],
        zarr_dir=str(tmp_path),
        labels_to_delete=["nuclei_mask"],
        tables_to_delete=["table_a", "table_c"],
    )

    c1 = ngio.open_ome_zarr_container(zarr1)
    # nuclei_mask deleted, nuclei kept
    assert "nuclei" in c1.list_labels()
    assert "nuclei_mask" not in c1.list_labels()
    # table_a deleted, table_b kept, table_c was absent (no error)
    assert "table_a" not in c1.list_tables()
    assert "table_b" in c1.list_tables()

    c2 = ngio.open_ome_zarr_container(zarr2)
    # nuclei_mask deleted, nuclei kept
    assert "nuclei" in c2.list_labels()
    assert "nuclei_mask" not in c2.list_labels()
    # table_a was absent in zarr2 (no error), table_c deleted, table_b kept
    assert "table_c" not in c2.list_tables()
    assert "table_b" in c2.list_tables()


def test_delete_all_labels(two_zarrs, tmp_path):
    """Delete all labels from both images."""
    zarr1, zarr2 = two_zarrs

    delete_images_labels_tables(
        zarr_urls=[zarr1, zarr2],
        zarr_dir=str(tmp_path),
        labels_to_delete=["nuclei", "nuclei_mask"],
    )

    for url in [zarr1, zarr2]:
        assert ngio.open_ome_zarr_container(url).list_labels() == []


def test_delete_none_matching(two_zarrs, tmp_path):
    """Requesting deletion of non-existent items leaves images unchanged."""
    zarr1, zarr2 = two_zarrs

    delete_images_labels_tables(
        zarr_urls=[zarr1, zarr2],
        zarr_dir=str(tmp_path),
        labels_to_delete=["nonexistent_label"],
        tables_to_delete=["nonexistent_table"],
    )

    c1 = ngio.open_ome_zarr_container(zarr1)
    assert {"nuclei", "nuclei_mask"}.issubset(c1.list_labels())
    assert {"table_a", "table_b"}.issubset(c1.list_tables())

    c2 = ngio.open_ome_zarr_container(zarr2)
    assert {"nuclei", "nuclei_mask"}.issubset(c2.list_labels())
    assert {"table_b", "table_c"}.issubset(c2.list_tables())


@pytest.fixture
def plate_zarr(tmp_path: Path):
    """
    Plate with well B/03 (images 0 and 1) and well C/05 (image 0).
    Every image has labels=[nuclei, nuclei_mask] and tables=[table_a].
    """
    plate_url = str(tmp_path / "plate.zarr")
    images = [
        ImageInWellPath(row="B", column="03", path="0"),
        ImageInWellPath(row="B", column="03", path="1"),
        ImageInWellPath(row="C", column="05", path="0"),
    ]
    create_empty_plate(store=plate_url, name="plate", images=images)
    zarr_urls = []
    for image in images:
        zarr_url = f"{plate_url}/{image.row}/{image.column}/{image.path}"
        container = _create_image(zarr_url)
        _add_feature_table(container, "table_a", "nuclei")
        zarr_urls.append(zarr_url)
    return plate_url, zarr_urls


def test_no_image_deletion_returns_none(two_zarrs, tmp_path):
    """Without image deletion, the task does not update the image list."""
    zarr1, zarr2 = two_zarrs

    output = delete_images_labels_tables(
        zarr_urls=[zarr1, zarr2],
        zarr_dir=str(tmp_path),
        labels_to_delete=["nuclei_mask"],
    )

    assert output is None


def test_delete_standalone_image(two_zarrs, tmp_path):
    """Deleting an image outside of a plate removes it from disk."""
    zarr1, zarr2 = two_zarrs

    output = delete_images_labels_tables(
        zarr_urls=[zarr1, zarr2],
        zarr_dir=str(tmp_path),
        images_names_to_delete=["img1.zarr"],
    )

    assert output == {"image_list_removals": [zarr1]}
    assert not Path(zarr1).exists()
    c2 = ngio.open_ome_zarr_container(zarr2)
    assert {"nuclei", "nuclei_mask"}.issubset(c2.list_labels())
    assert {"table_b", "table_c"}.issubset(c2.list_tables())


def test_delete_image_in_plate(plate_zarr, tmp_path):
    """Deleting an image in a plate removes it from disk and well metadata."""
    plate_url, zarr_urls = plate_zarr

    output = delete_images_labels_tables(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        images_names_to_delete=["1"],
    )

    assert output == {"image_list_removals": [f"{plate_url}/B/03/1"]}
    assert not Path(f"{plate_url}/B/03/1").exists()
    plate = ngio.open_ome_zarr_plate(plate_url)
    assert plate.get_well("B", "03").paths() == ["0"]
    assert plate.get_well("C", "05").paths() == ["0"]
    assert sorted(plate.wells_paths()) == ["B/03", "C/05"]
    # Remaining images are untouched
    for url in [f"{plate_url}/B/03/0", f"{plate_url}/C/05/0"]:
        container = ngio.open_ome_zarr_container(url)
        assert {"nuclei", "nuclei_mask"}.issubset(container.list_labels())
        assert "table_a" in container.list_tables()


def test_delete_image_empties_well(plate_zarr, tmp_path):
    """Image names match in every well; emptied wells leave the plate."""
    plate_url, zarr_urls = plate_zarr

    output = delete_images_labels_tables(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        images_names_to_delete=["0"],
    )

    assert output == {
        "image_list_removals": [f"{plate_url}/B/03/0", f"{plate_url}/C/05/0"]
    }
    assert not Path(f"{plate_url}/B/03/0").exists()
    assert not Path(f"{plate_url}/C/05/0").exists()
    assert Path(f"{plate_url}/B/03/1").exists()
    plate = ngio.open_ome_zarr_plate(plate_url)
    assert plate.wells_paths() == ["B/03"]
    assert plate.get_well("B", "03").paths() == ["1"]


def test_delete_images_then_labels_tables(plate_zarr, tmp_path):
    """Labels and tables are only processed on the images that remain."""
    plate_url, zarr_urls = plate_zarr

    delete_images_labels_tables(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        images_names_to_delete=["1"],
        labels_to_delete=["nuclei_mask"],
        tables_to_delete=["table_a"],
    )

    assert not Path(f"{plate_url}/B/03/1").exists()
    for url in [f"{plate_url}/B/03/0", f"{plate_url}/C/05/0"]:
        container = ngio.open_ome_zarr_container(url)
        assert container.list_labels() == ["nuclei"]
        assert "table_a" not in container.list_tables()


def test_delete_image_rerun(plate_zarr, tmp_path):
    """Re-running on the updated image list finds nothing to delete."""
    plate_url, zarr_urls = plate_zarr

    output = delete_images_labels_tables(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        images_names_to_delete=["1"],
    )
    # Fractal removes the deleted images from the image list
    assert output is not None
    remaining_zarr_urls = [
        url for url in zarr_urls if url not in output["image_list_removals"]
    ]

    output = delete_images_labels_tables(
        zarr_urls=remaining_zarr_urls,
        zarr_dir=str(tmp_path),
        images_names_to_delete=["1"],
    )

    assert output is None
    plate = ngio.open_ome_zarr_plate(plate_url)
    assert plate.get_well("B", "03").paths() == ["0"]
    assert plate.get_well("C", "05").paths() == ["0"]
    assert Path(f"{plate_url}/B/03/0").exists()
    assert Path(f"{plate_url}/C/05/0").exists()


def test_delete_image_already_deleted(plate_zarr, tmp_path):
    """A stale image list entry for an already deleted image is removed.

    E.g. if a previous run deleted the data but its image list update was
    never applied. The task should not fail and should still report the
    image for removal from the image list.
    """
    plate_url, zarr_urls = plate_zarr

    for _ in range(2):
        output = delete_images_labels_tables(
            zarr_urls=zarr_urls,
            zarr_dir=str(tmp_path),
            images_names_to_delete=["1"],
        )

    assert output == {"image_list_removals": [f"{plate_url}/B/03/1"]}
    plate = ngio.open_ome_zarr_plate(plate_url)
    assert plate.get_well("B", "03").paths() == ["0"]


def test_delete_image_not_in_well_metadata(plate_zarr, tmp_path):
    """An image on disk but missing from the well metadata is still deleted."""
    plate_url, zarr_urls = plate_zarr
    plate = ngio.open_ome_zarr_plate(plate_url)
    plate.remove_image(row="B", column="03", image_path="1")

    delete_images_labels_tables(
        zarr_urls=zarr_urls,
        zarr_dir=str(tmp_path),
        images_names_to_delete=["1"],
    )

    assert not Path(f"{plate_url}/B/03/1").exists()
    plate = ngio.open_ome_zarr_plate(plate_url)
    assert plate.get_well("B", "03").paths() == ["0"]
