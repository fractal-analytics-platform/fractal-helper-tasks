### Purpose
- Renames channels in OME-Zarr images by supplying a mapping of old names to new names.
- Only updates Zarr metadata — no image arrays are rewritten, so the operation is fast regardless of image size.
- The mapping may contain more entries than a given image has channels (e.g. in multiplexing, where different images carry different channel subsets). Channels not found in an image are skipped with a warning by default.

### Outputs
- Updated channel names in the Zarr metadata of each image.

### Limitations
- Only renames channels; does not reorder them or change other channel metadata (colors, wavelength IDs, display ranges).
- Edits Zarr metadata in-place and irreversibly.
