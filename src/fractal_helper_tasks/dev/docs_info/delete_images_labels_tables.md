### Purpose
- Deletes whole OME-Zarr images, label images and/or tables from a list of OME-Zarr images.
- Useful for cleanup steps, e.g. removing intermediate images (such as the raw images or intermediate processing images), intermediate segmentation results or outdated tables.
- **Images** are selected by name, i.e. the last part of their path (e.g. `0` or `1_registered` for an image at `plate.zarr/B/03/1_registered`). Every image in the input list whose name matches is deleted. In an HCS plate, a name like `0` therefore matches the image with that name **in every well**. If you want an image to be deleted, it needs to be included in the selection of images sent to the task and pass the images_names_to_delete filter.
- **Labels and tables** are deleted by name from all remaining images. Items in the deletion lists that are absent from a given image are silently skipped, so it is safe to pass the same list across images that have different labels or tables.

### Outputs
- Deleted images are removed from disk, from the well metadata of their HCS plate (if they are part of one) and from the Fractal image list.
- Wells that no longer contain any image are removed from the plate metadata. Their (empty) folders stay on disk.
- The specified labels and tables are removed from each remaining image in-place.

### Limitations
- ⚠️ **Use this task with care: all deletions are permanent and cannot be undone.** Deleted images, labels and tables cannot be recovered. Double-check the names you enter, and filter the input image list (e.g. by well or by type) if you only want to act on some of the images.
- Image names are matched exactly against the last part of each image path; there is no pattern matching.
- Image deletion is applied before label and table deletion. Labels and tables are only deleted from images that were not deleted.
