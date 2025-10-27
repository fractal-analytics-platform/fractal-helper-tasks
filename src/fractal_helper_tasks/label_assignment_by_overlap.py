# Copyright 2024 (C) BioVisionCenter, University of Zurich
#
# Original authors:
# Joel Lüthi <joel.luethi@uzh.ch>
# Derived from APx label assignment task
# https://github.com/Apricot-Therapeutics/APx_fractal_task_collection/blob/main/src/apx_fractal_task_collection/tasks/label_assignment_by_overlap.py
# Simplified to avoid compound task and updated to use ngio
"""Task to assign labels based on overlap between two label images."""

import logging
from typing import Optional

import ngio
import numpy as np
import pandas as pd
from ngio.tables import FeatureTable
from pydantic import validate_call
from skimage.measure import regionprops_table

logger = logging.getLogger(__name__)


def label_overlap(
    regionmask: np.ndarray,
    intensity_image: np.ndarray,
) -> list[float, float]:
    """Calculates label overlap between 2 numpy arrays.

    Scikit-image regionprops_table extra_properties function to compute
    max overlap between two label images. Based on APX implementation.

    regionmask: 2D numpy array of labels, where each label corresponds to a
        child object.
    intensity_image: 2D numpy array with parent objects.
    """
    parent_labels = np.where(regionmask > 0, intensity_image, 0)

    labels, counts = np.unique(parent_labels[parent_labels > 0], return_counts=True)

    if len(labels > 0):
        # if there is a tie in the overlap, the first label is selected
        max_label = labels[np.argmax(counts)]
        max_label_area = counts.max()
        child_area = regionmask[regionmask > 0].size
        overlap = max_label_area / child_area
    else:
        max_label = np.nan
        overlap = np.nan

    return [max_label, overlap]


def assign_objects(
    parent_label: np.ndarray,
    child_label: np.ndarray,
    overlap_threshold=1.0,
) -> pd.DataFrame:
    """Assigns objects of child label to parent label based on overlap.

    Calculate the overlap between labels in label_a and label_b,
    and return a DataFrame of matching labels. Based on APX implementation.

    label_a:  4D numpy array.
    label_b:  4D numpy array.
    overlap_threshold: float, the minimum fraction of child label object that
        must be contained in parent label object to be considered a match.
    """
    parent_label = np.squeeze(parent_label)
    child_label = np.squeeze(child_label)

    t = pd.DataFrame(
        regionprops_table(
            child_label,
            parent_label,
            properties=["label"],
            extra_properties=[label_overlap],
        )
    )

    t.columns = ["child_label", "parent_label", "overlap"]
    t.loc[t.overlap < overlap_threshold, "parent_label"] = np.nan
    t["parent_label"] = t["parent_label"].astype("Int32")
    t.set_index("child_label", inplace=True)

    return t


@validate_call
def label_assignment_by_overlap(
    *,
    # Default arguments for fractal tasks:
    zarr_url: str,
    # Task-specific arguments:
    parent_label_name: str,
    child_label_name: str,
    overlap_threshold: float = 1.0,
    overlap_table_name: Optional[str] = None,
    level_path: Optional[str] = None,
):
    """Assign labels to each other based on overlap.

    Takes a parent label image and a child label image and calculates
    overlaps between their labels. Child labels will be assigned to parent
    labels based on an overlap threshold.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        parent_label_name: Name of the parent label.
        child_label_name: Name of the child label. This label will be assigned
            to the parent label based on overlap. The parent label will appear
            in the child feature table as the "(parent_label_name)_label"
            column in the obs table of the anndata table.
        overlap_threshold: The minimum percentage (between 0 and 1) of child
            label object that must be contained in parent label object to
             be considered a match.
        overlap_table_name: Name of the feature table to which the overlap
            should be added. If the feature table already exists, the overlap
            measurements are added to it. Otherwise, an overlap table is
            created. If no name was given, a new table named parent_label_name
            + child_label_name + '_overlap' is created.
        level_path: Resolution of the label image to calculate overlap. Full
            resolution label is used by default. Typically overriden with
            0 (full resolution), 1 (half resolution), etc.

    """
    ome_zarr_container = ngio.open_ome_zarr_container(zarr_url)
    child_label_container = ome_zarr_container.get_label(
        name=child_label_name, path=level_path
    )
    child_label = child_label_container.get_array()

    # if there are no child labels, assignments will be all NaN
    if np.unique(child_label).size == 1:
        assignments = pd.DataFrame(
            {"parent_label": pd.NA, "overlap": pd.NA}, index=pd.Index([])
        )
        logger.info(
            f"Label image was empty for child label {child_label_name}. "
            "No labels could be matched."
        )

    else:
        # Load the parent label image at the resolution of the child label image
        parent_label = ome_zarr_container.get_label(
            name=parent_label_name,
            pixel_size=child_label_container.pixel_size,
        ).get_array()
        # make the assignment
        logger.info(
            "Calculating label assignments with overlap threshold "
            f"{overlap_threshold}."
        )
        assignments = assign_objects(
            parent_label,
            child_label,
            overlap_threshold,
        )

    parent_label_column_name = f"{parent_label_name}_label"
    overlap_column_name = f"{child_label_name}_{parent_label_name}_overlap"

    assignments.rename(
        columns={
            "parent_label": parent_label_column_name,
            "overlap": overlap_column_name,
        },
        inplace=True,
    )

    # Check if the feature table already exists
    if overlap_table_name in ome_zarr_container.list_tables():
        base_table_container = ome_zarr_container.get_table(
            name=overlap_table_name,
        )
        if base_table_container.table_type() != "feature_table":
            raise ValueError(
                f"The existing table {overlap_table_name} is not a "
                "FeatureTable. Cannot add overlap measurements to it."
            )
        base_table = base_table_container.dataframe
        # If the table already contains the overlap measurement, drop it
        if parent_label_column_name in base_table.columns:
            base_table.drop(
                columns=[
                    parent_label_column_name,
                    overlap_column_name,
                ],
                inplace=True,
            )

    else:
        if overlap_table_name is None:
            overlap_table_name = f"{parent_label_name}_{child_label_name}_overlap"

        # Initialize a new empty table with label index
        labels = np.unique(child_label)[1:]  # FIXME: Handle empty label image
        base_table = pd.DataFrame(index=labels)
        base_table.index.name = "label"

    # merge with child feature obs data
    merged_data = pd.merge(
        base_table, assignments, left_on="label", right_index=True, how="left"
    )

    merged_table = FeatureTable(
        table_data=merged_data,
        reference_label=child_label_name,
    )

    ome_zarr_container.add_table(
        name=overlap_table_name,
        table=merged_table,
        overwrite=True,
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=label_assignment_by_overlap,
        logger_name=logger.name,
    )
