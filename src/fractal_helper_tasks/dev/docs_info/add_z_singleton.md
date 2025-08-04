### Purpose
- Creates a **singleton Z dimension** in a 2D OME-Zarr image. Useful when 2D images don't have a singleton Z dimension but downstream tasks require it.
- Optionally overwrites the input image if `overwrite_input` is set to True.

### Outputs
- A **new Zarr image** with the singleton Z dimension 

### Limitations
- Only processes 2D OME-Zarr images without a **Z-axis**.  
- Does not copy associated **label images** to the new Zarr structure.  