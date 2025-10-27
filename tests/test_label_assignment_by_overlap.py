"""Test label assignment by overlap task."""

from pathlib import Path

import ngio
import numpy as np
import pandas as pd
import pytest
from ngio.tables import FeatureTable

from fractal_helper_tasks.label_assignment_by_overlap import (
    label_assignment_by_overlap,
)


@pytest.mark.parametrize(
    ["overlap_threshold", "expected_assignment"],
    [(0.8, [1, 2, 3, np.nan, 2]), (1.0, [1, 2, 3, np.nan, np.nan])],
)
def test_label_assignment_by_overlap_new_table(
    tmp_path: Path,
    overlap_threshold: float,
    expected_assignment: list[int],
):
    zarr_url = str(tmp_path / "my_zarr.zarr")
    orig_dimensions = (1, 100, 100)
    orig_axes_names = ["c", "y", "x"]
    parent_label_name = "parent_label"
    child_label_name = "child_label"
    overlap_table_name = "overlap_table"

    ome_zarr_container = ngio.create_ome_zarr_from_array(
        store=zarr_url,
        array=np.zeros(orig_dimensions),
        xy_pixelsize=0.5,
        axes_names=orig_axes_names,
        overwrite=True,
    )

    parent_label_array = np.zeros(orig_dimensions[1:], dtype=np.uint32)
    parent_label_array[0:50, 0:50] = 1
    parent_label_array[0:50, 50:100] = 2
    parent_label_array[50:100, 0:50] = 3

    child_label_array = np.zeros(orig_dimensions[1:], dtype=np.uint32)
    child_label_array[10:30, 10:30] = 1
    child_label_array[10:30, 70:90] = 2
    child_label_array[70:90, 10:30] = 3
    child_label_array[70:90, 70:90] = 4
    child_label_array[10:30, 45:80] = 5

    parent_label = ome_zarr_container.derive_label(
        name=parent_label_name,
    )
    parent_label.set_array(patch=parent_label_array)

    child_label = ome_zarr_container.derive_label(
        name=child_label_name,
    )
    child_label.set_array(patch=child_label_array)

    label_assignment_by_overlap(
        zarr_url=zarr_url,
        parent_label_name=parent_label_name,
        child_label_name=child_label_name,
        overlap_threshold=overlap_threshold,
        overlap_table_name=overlap_table_name,
    )

    new_ome_zarr_container = ngio.open_ome_zarr_container(zarr_url)
    assert overlap_table_name in new_ome_zarr_container.list_tables()
    overlap_table = new_ome_zarr_container.get_table_as(
        name=overlap_table_name,
        table_cls=FeatureTable,
    ).dataframe
    assert len(overlap_table) == 5
    expected_output_columns = [
        f"{child_label_name}_{parent_label_name}_overlap",
        f"{parent_label_name}_label",
    ]
    assert overlap_table.columns.tolist() == expected_output_columns
    assigned_overlaps = overlap_table[f"{parent_label_name}_label"].to_numpy(
        dtype=float
    )
    assert np.allclose(assigned_overlaps, expected_assignment, equal_nan=True)


def test_label_assignment_by_overlap_existing_feature_table(
    tmp_path: Path,
):
    zarr_url = str(tmp_path / "my_zarr.zarr")
    orig_dimensions = (1, 100, 100)
    orig_axes_names = ["c", "y", "x"]
    parent_label_name = "parent_label"
    child_label_name = "child_label"
    overlap_table_name = "overlap_table"

    ome_zarr_container = ngio.create_ome_zarr_from_array(
        store=zarr_url,
        array=np.zeros(orig_dimensions),
        xy_pixelsize=0.5,
        axes_names=orig_axes_names,
        overwrite=True,
    )

    parent_label_array = np.zeros(orig_dimensions[1:], dtype=np.uint32)
    parent_label_array[0:50, 0:50] = 1
    parent_label_array[0:50, 50:100] = 2
    parent_label_array[50:100, 0:50] = 3

    child_label_array = np.zeros(orig_dimensions[1:], dtype=np.uint32)
    child_label_array[10:30, 10:30] = 1
    child_label_array[10:30, 70:90] = 2
    child_label_array[70:90, 10:30] = 3
    child_label_array[70:90, 70:90] = 4
    child_label_array[10:30, 45:80] = 5

    parent_label = ome_zarr_container.derive_label(
        name=parent_label_name,
    )
    parent_label.set_array(patch=parent_label_array)

    child_label = ome_zarr_container.derive_label(
        name=child_label_name,
    )
    child_label.set_array(patch=child_label_array)

    # Create existing tables
    base_table = pd.DataFrame(
        {
            "some_other_measurement": [10, 20, 30, 40, 50],
        },
        index=[1, 2, 3, 4, 5],
    )
    base_table.index.name = "label"

    feature_table = FeatureTable(
        table_data=base_table,
        reference_label=child_label_name,
    )
    ome_zarr_container.add_table(
        name=overlap_table_name,
        table=feature_table,
        overwrite=True,
    )

    label_assignment_by_overlap(
        zarr_url=zarr_url,
        parent_label_name=parent_label_name,
        child_label_name=child_label_name,
        overlap_threshold=1.0,
        overlap_table_name=overlap_table_name,
    )

    new_ome_zarr_container = ngio.open_ome_zarr_container(zarr_url)
    assert overlap_table_name in new_ome_zarr_container.list_tables()
    overlap_table = new_ome_zarr_container.get_table_as(
        name=overlap_table_name,
        table_cls=FeatureTable,
    ).dataframe
    assert overlap_table.shape == (5, 3)


def test_label_assignment_by_overlap_existing_roi_table(
    tmp_path: Path,
):
    zarr_url = str(tmp_path / "my_zarr.zarr")
    orig_dimensions = (1, 100, 100)
    orig_axes_names = ["c", "y", "x"]
    parent_label_name = "parent_label"
    child_label_name = "child_label"
    overlap_table_name = "image_roi_table"

    ome_zarr_container = ngio.create_ome_zarr_from_array(
        store=zarr_url,
        array=np.zeros(orig_dimensions),
        xy_pixelsize=0.5,
        axes_names=orig_axes_names,
        overwrite=True,
    )

    parent_label_array = np.zeros(orig_dimensions[1:], dtype=np.uint32)
    parent_label_array[0:50, 0:50] = 1
    parent_label_array[0:50, 50:100] = 2
    parent_label_array[50:100, 0:50] = 3

    child_label_array = np.zeros(orig_dimensions[1:], dtype=np.uint32)
    child_label_array[10:30, 10:30] = 1
    child_label_array[10:30, 70:90] = 2
    child_label_array[70:90, 10:30] = 3
    child_label_array[70:90, 70:90] = 4
    child_label_array[10:30, 45:80] = 5

    parent_label = ome_zarr_container.derive_label(
        name=parent_label_name,
    )
    parent_label.set_array(patch=parent_label_array)

    child_label = ome_zarr_container.derive_label(
        name=child_label_name,
    )
    child_label.set_array(patch=child_label_array)

    # Create existing table
    roi_table = ome_zarr_container.build_image_roi_table(name=overlap_table_name)

    ome_zarr_container.add_table(
        name=overlap_table_name,
        table=roi_table,
        overwrite=True,
    )
    with pytest.raises(ValueError):
        label_assignment_by_overlap(
            zarr_url=zarr_url,
            parent_label_name=parent_label_name,
            child_label_name=child_label_name,
            overlap_threshold=1.0,
            overlap_table_name=overlap_table_name,
        )
