### Purpose
- Converts a **2D segmentation** image into a **3D segmentation** by replicating the 2D segmentation across Z-slices.  
- Supports OME-Zarr datasets where **2D and 3D images** share the same base name but differ by suffixes.  
- Optionally copies associated ROI tables and adjusts them to align with the replicated Z-dimensions.  

### Outputs
- A **3D segmentation label image** saved with a new name.  
- Updated **ROI tables** adjusted for Z-dimensions (optional).  

### Limitations
- Only supports **same-base 2D and 3D Zarr names**; full flexibility in file names is not yet implemented.  
- Assumes **2D OME-Zarr images** and corresponding 3D images are stored in the same base folder and just differ with a suffix before the .zarr.  
