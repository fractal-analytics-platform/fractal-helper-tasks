### Purpose
- Stores user-defined ROIs into an OME-Zarr ROI table.
- Each ROI is defined by two corners in physical coordinates (e.g. micrometers), as reported by the viewer (deck.gl `info.coordinate`, which already includes the `modelMatrix` transform). X and Y are stored as-is, while Z and T are interpreted as slice/frame indices and converted using the image pixel size.
- Corners may be given in any order, then the task normalizes them to a positive-extent ROI.

### Outputs
- A physical-coordinate ROI table (default name `user_ROIs`) written into the OME-Zarr container.
- With `overwrite=False` and an existing table, new ROIs are appended. A name that already exists raises an error.
- With `overwrite=True`, the table is replaced.

### Limitations
- ROIs must fall within the image bounds, otherwise the task raises an error.
- Z/T coordinates are required when the image has the corresponding axis, and are ignored (with a warning) when provided for an axis the image does not have.
- ROI names must be unique within a single task call.
