### Purpose
- Deletes specified label images and/or tables from a list of OME-Zarr images.
- Useful for cleanup steps, e.g. removing intermediate segmentation results or tables that are no longer needed.
- Items in the deletion lists that are absent from a given image are silently skipped — it is safe to pass the same list across images that have different labels or tables.

### Outputs
- The specified labels and tables are removed from each image in-place.

### Limitations
- Edits are **irreversible** — deleted labels and tables cannot be recovered.
