### Purpose
- Rechunks OME-Zarr to new chunking parameters: Changes whether the array is stored as many small files or few larger files.
- Optionally applies the same rechunking to label images.
- Optionally rebuilds pyramids from scratch in the rechunked image.

### Outputs
- A **new Zarr image** written alongside the original, with `_{suffix}` appended to the name (default: `_rechunked`).

### Limitations
- Does not support in-place rechunking — always writes a new Zarr; there is no `overwrite_input` option.
