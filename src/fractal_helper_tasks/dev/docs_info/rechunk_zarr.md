### Purpose
- Rechunks OME-Zarr to new chunking parameters: Changes whether the array is stored as many small files or few larger files.
- Optionally applies the same rechunking to label images.

### Outputs
- A **new Zarr image** that is rechunked.
