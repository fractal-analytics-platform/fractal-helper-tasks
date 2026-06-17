"""Task to store user-defined ROIs into an OME-Zarr ROI table."""

import json
import logging
from typing import Annotated

from ngio import Image, OmeZarrContainer, open_ome_zarr_container
from ngio.common import Roi
from ngio.tables.v1._roi_table import RoiTableV1
from pydantic import BaseModel, Field, validate_call


def _roi_axis(roi: Roi, axis: str) -> tuple[float | None, float | None]:
    """Return (start, length) for *axis*, or (None, None) if absent."""
    s = roi.get(axis)
    if s is None:
        return None, None
    return s.start, s.length


logger = logging.getLogger(__name__)

# Regex pattern to validate JSON-like ROI corner strings.
ROI_CORNERS_PATTERN = (
    r'^\{\s*(?:"\w+"\s*:\s*(?:"[^"]*"|\d+(?:\.\d+)?)\s*,?\s*)+\}$'  # basic JSON pattern
)
# more specific altrernative:
# r'^\{\s*"name"\s*:\s*"[^"]+"(?:\s*,\s*"[xyzt][12]"\s*:\s*\d+(?:\.\d+)?){4,8}\s*\}$'

RoiCornersStr = Annotated[str, Field(pattern=ROI_CORNERS_PATTERN)]


def parse_roi_corners_str(roi_str: str) -> "RoiCorners":
    """Parse a JSON string into a RoiCorners model.

    Args:
        roi_str: A JSON string with keys matching RoiCorners fields.

    Returns:
        A validated RoiCorners instance.

    Raises:
        ValueError: If the string is not valid JSON or does not match
            the RoiCorners schema.
    """
    try:
        data = json.loads(roi_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in ROI corners string: {e}") from e
    return RoiCorners(**data)


class RoiCorners(BaseModel):
    """Two corners defining a rectangular ROI.

    X and Y are in physical coordinates (e.g. micrometers) as reported by
    the viewer (deck.gl info.coordinate includes the modelMatrix transform).
    Z and T are slice/frame indices.

    Attributes:
        name: Identifier for the ROI.
        x1: X physical coordinate of the first corner.
        y1: Y physical coordinate of the first corner.
        z1: Z slice index of the first corner (optional, required if image has Z axis).
        t1: T frame index of the first corner (optional, required if image has T axis).
        x2: X physical coordinate of the second corner.
        y2: Y physical coordinate of the second corner.
        z2: Z slice index of the second corner (optional, required if image has Z axis).
        t2: T frame index of the second corner (optional, required if image has T axis).
    """

    name: str
    x1: float
    y1: float
    z1: float | None = None
    t1: float | None = None
    x2: float
    y2: float
    z2: float | None = None
    t2: float | None = None


def validate_roi_within_image(roi: Roi, image: Image) -> None:
    """Validate that a physical-space ROI falls within the image bounds.

    Args:
        roi: The ROI in physical coordinates.
        image: The ngio Image to validate against.

    Raises:
        ValueError: If the ROI extends outside the image bounds.
    """
    ps = image.pixel_size
    dims = image.dimensions
    image_x = dims.get("x") * ps.x
    image_y = dims.get("y") * ps.y
    n_z = dims.get("z")
    image_z = n_z * ps.z if n_z is not None else None
    n_t = dims.get("t")
    image_t = n_t * ps.t if n_t is not None else None

    rx, rx_len = _roi_axis(roi, "x")
    ry, ry_len = _roi_axis(roi, "y")
    rz, rz_len = _roi_axis(roi, "z")
    rt, rt_len = _roi_axis(roi, "t")

    errors: list[str] = []
    if rx is not None and rx_len is not None:
        if rx < 0 or rx + rx_len > image_x:
            errors.append(
                f"X range [{rx}, {rx + rx_len}] exceeds image bounds [0, {image_x}]"
            )
    if ry is not None and ry_len is not None:
        if ry < 0 or ry + ry_len > image_y:
            errors.append(
                f"Y range [{ry}, {ry + ry_len}] exceeds image bounds [0, {image_y}]"
            )
    if image_z is not None and rz is not None and rz_len is not None:
        if rz < 0 or rz + rz_len > image_z:
            errors.append(
                f"Z range [{rz}, {rz + rz_len}] exceeds image bounds [0, {image_z}]"
            )
    if image_t is not None and rt is not None and rt_len is not None:
        if rt < 0 or rt + rt_len > image_t:
            errors.append(
                f"T range [{rt}, {rt + rt_len}] exceeds image bounds [0, {image_t}]"
            )
    if errors:
        raise ValueError(
            f"ROI '{roi.name}' is out of image bounds: {'; '.join(errors)}"
        )


def corners_to_roi(corners: RoiCorners, image: Image) -> Roi:
    """Convert physical-space corners to a physical-space Roi.

    The viewer (deck.gl) provides coordinates already in physical units
    (e.g. micrometers) via its modelMatrix transform, so no pixel-size
    conversion is needed for X and Y.  For Z and T, when the two corners
    are equal the user intends a single plane/timepoint, so the length
    is set to one physical step (1 * pixel_size).

    Validates that Z/T coordinates are provided when the image has the
    corresponding axis, and warns (ignoring the values) when coordinates
    are provided but the image lacks that axis.

    Args:
        corners: The two corners in physical coordinates (from the viewer).
        image: The ngio Image, used for axis validation and Z/T pixel size.

    Returns:
        A Roi object in physical coordinates.

    Raises:
        ValueError: If the image has Z or T axis but the corresponding
            corner coordinates are not provided, or if the resulting ROI
            falls outside the image bounds.
    """
    ps = image.pixel_size
    has_z = image.has_axis("z")
    has_t = image.has_axis("t")
    z_provided = corners.z1 is not None and corners.z2 is not None
    t_provided = corners.t1 is not None and corners.t2 is not None

    # --- Z axis validation ---
    if has_z and not z_provided:
        raise ValueError(
            f"ROI '{corners.name}': image has a Z axis but z1/z2 coordinates "
            "were not provided."
        )
    if not has_z and z_provided:
        logger.warning(
            "[WARNING] ROI '%s': z1/z2 coordinates were provided but the image has no Z"
            " axis. Z values will be ignored.",
            corners.name,
        )

    # --- T axis validation ---
    if has_t and not t_provided:
        raise ValueError(
            f"ROI '{corners.name}': image has a T axis but t1/t2 coordinates"
            "were not provided."
        )
    if not has_t and t_provided:
        logger.warning(
            "[WARNING] ROI '%s': t1/t2 coordinates were provided but the image has no T"
            " axis. T values will be ignored.",
            corners.name,
        )

    # Build slices dict for Roi.from_values
    # X and Y are already in physical units from the viewer (deck.gl modelMatrix).
    slices: dict[str, tuple[float, float]] = {
        "x": (min(corners.x1, corners.x2), abs(corners.x2 - corners.x1)),
        "y": (min(corners.y1, corners.y2), abs(corners.y2 - corners.y1)),
    }
    # Z and T are slice/frame indices from the viewer, so they need conversion.
    if has_z and z_provided:
        z_len = (abs(corners.z2 - corners.z1) + 1) * ps.z
        slices["z"] = (
            min(corners.z1, corners.z2) * ps.z,
            z_len,
        )
    if has_t and t_provided:
        t_len = (abs(corners.t2 - corners.t1) + 1) * ps.t
        slices["t"] = (
            min(corners.t1, corners.t2) * ps.t,
            t_len,
        )

    roi = Roi.from_values(slices=slices, name=corners.name)
    validate_roi_within_image(roi, image)
    return roi


@validate_call
def roi_selection_task(
    *,
    # Fractal parameters
    zarr_url: str,  # fractal does not support Path so we use str
    # Task-specific arguments
    roi_corners: list[RoiCornersStr],
    output_table_name: str = "user_ROIs",
    overwrite: bool = False,
) -> None:
    """Store user-defined ROIs into an OME-Zarr ROI table.

    This task receives a list of ROIs defined by two corners in physical
    coordinates (from the viewer, e.g. deck.gl info.coordinate which
    includes the modelMatrix transform) and writes them as a
    physical-coordinate ROI table into the OME-Zarr container.

    Args:
        zarr_url: Path or url to the individual OME-Zarr image to be processed.
            (standard argument for Fractal tasks, managed by Fractal server).
        roi_corners: List of ROIs as JSON strings, each defining two corners
            in physical coordinates (from the viewer).
        output_table_name: Name of the output ROI table to create.
        overwrite: Whether to overwrite the output ROI table if it exists.
    """
    zarr_url = zarr_url.rstrip("/")
    logger.info(f"{zarr_url=}")

    if len(roi_corners) == 0:
        raise ValueError("No ROI corners provided.")

    parsed_corners = [parse_roi_corners_str(s) for s in roi_corners]

    names = [c.name for c in parsed_corners]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(
            f"Duplicate ROI names in input: {duplicates}. "
            "Each ROI must have a unique name."
        )

    ome_zarr: OmeZarrContainer = open_ome_zarr_container(zarr_url)
    logger.info(f"{ome_zarr=}")

    image: Image = ome_zarr.get_image()
    logger.info(f"{image=}")

    rois: list[Roi] = []
    for corners in parsed_corners:
        roi = corners_to_roi(corners, image)
        logger.info(f"Converted {corners.name}: {roi}")
        rois.append(roi)

    table_exists = output_table_name in ome_zarr.list_tables()

    if overwrite or not table_exists:
        table = RoiTableV1(rois=rois)
        ome_zarr.add_table(output_table_name, table, overwrite=overwrite)
    else:
        # Append to existing table; raise if any new ROI name already exists
        existing_table: RoiTableV1 = ome_zarr.get_table_as(
            output_table_name, RoiTableV1
        )
        existing_names = {r.name for r in existing_table.rois()}
        new_names = {r.name for r in rois}
        conflicts = existing_names & new_names
        if conflicts:
            raise FileExistsError(
                f"ROI(s) {conflicts} already exist in table "
                f"'{output_table_name}'. Set overwrite=True to overwrite."
            )
        existing_table.add(rois)
        ome_zarr.add_table(output_table_name, existing_table, overwrite=True)

    logger.info(
        f"Saved {len(rois)} ROI(s) to table '{output_table_name}' at {zarr_url=}"
    )


if __name__ == "__main__":
    from fractal_task_tools.task_wrapper import run_fractal_task

    run_fractal_task(
        task_function=roi_selection_task,
        logger_name=logger.name,
    )
