### Purpose
- Pads all wells in an HCS plate to the same size by extending smaller wells to match the maximum well dimensions across the plate.
- Designed for **search-first acquisitions**, where different wells in a multi-well plate have different image shapes. Unequal well sizes cause display issues in plate viewers such as ViZarr, the napari plate overview, and MoBie.
- Padding is performed by **updating Zarr metadata only** — the underlying image arrays are not rewritten, making this operation fast regardless of image size.
- Optionally pads segmentation labels alongside the intensity images.

### Outputs
- Updated Zarr metadata for all images (and labels) in the plate, with consistent well dimensions across the plate.

### Limitations
- Edits Zarr metadata **in-place and irreversibly** — there is no undo. Run on a copy if the original metadata must be preserved.
- Best used **after conversion and before segmentation**: padding before segmentation ensures that all downstream processing operates on consistently sized images. The task can also pad existing segmentation labels, but does not handle the case where labels have a different pixel resolution than the intensity image.
- Only works on **HCS plates**.
