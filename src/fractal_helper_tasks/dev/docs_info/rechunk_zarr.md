### Purpose
- Rechunks OME-Zarr to new chunking parameters: Changes whether the array is stored as many small files or few larger files.
- Optionally applies the same rechunking to label images.
- Optionally rebuilds pyramids from scratch in the rechunked image.

### Outputs
- By default (`overwrite_input=True`): rechunks **in place** — the original Zarr is replaced with the rechunked version.
- With `overwrite_input=False`: writes a new Zarr alongside the original, with `_{suffix}` appended to the name (default suffix: `_rechunked`).

### Limitations
- In-place mode uses a temporary rename (`_tmp`) followed by deletion of the original; this is not atomic and requires sufficient disk space to hold both copies during the operation.
