### Purpose
- Removes a **singleton time (T) dimension** from an OME-Zarr image.  
- Creates a new OME-Zarr image with updated metadata and dimensions.
- Optionally overwrites the input image if `overwrite_input` is set to True.

### Outputs
- A **new Zarr image** without the singleton T-dimension, stored with a configurable suffix.  

### Limitations
- Only processes OME-Zarr images where the **T-axis is the first axis**.  
- Assumes the T-dimension is **singleton**; does not process non-singleton time axes.  
- Does not copy associated **label images** to the new Zarr structure.  