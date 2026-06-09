"""Tests for delete_labels_tables task."""

from pathlib import Path

import ngio
import pandas as pd
import pytest
from ngio.tables import FeatureTable

from fractal_helper_tasks.delete_labels_tables import delete_labels_tables


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

    delete_labels_tables(
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

    delete_labels_tables(
        zarr_urls=[zarr1, zarr2],
        zarr_dir=str(tmp_path),
        labels_to_delete=["nuclei", "nuclei_mask"],
    )

    for url in [zarr1, zarr2]:
        assert ngio.open_ome_zarr_container(url).list_labels() == []


def test_delete_none_matching(two_zarrs, tmp_path):
    """Requesting deletion of non-existent items leaves images unchanged."""
    zarr1, zarr2 = two_zarrs

    delete_labels_tables(
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
