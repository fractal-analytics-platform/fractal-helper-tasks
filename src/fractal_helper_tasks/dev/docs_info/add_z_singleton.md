### Purpose
- Creates a **singleton Z dimension** in a 2D OME-Zarr image. Useful when 2D images don't have a singleton Z dimension but downstream tasks require it.
- Overwrites the input image by default (`overwrite_input=True`). Set `overwrite_input=False` to keep the original and write the result to a new Zarr with a configurable suffix instead.

### Outputs
- A **new Zarr image** with the singleton Z dimension 

### Limitations
- Only processes 2D OME-Zarr images without a **Z-axis**.  
- Does not copy associated **label images** to the new Zarr structure.  