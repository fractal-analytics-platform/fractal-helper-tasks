### Purpose
- Copies labels from a reference acquisition to all other acquisitions in the same HCS well.
- Designed for multiplexing workflows where segmentation is computed on one acquisition and needs to be propagated to the remaining acquisitions.
- Copies labels as-is — pixel coordinates are not transformed during the copy. This task must therefore run **after** registration has been applied to the non-reference images, so that all acquisitions are already in the same coordinate space.

### Outputs
- The specified label images are written to each non-reference acquisition in every submitted well.

### Limitations
- Only works on **HCS plates** — images must be part of a well with acquisition metadata (standard OME-NGFF HCS layout).
- Requires **exactly one image per well** with the reference acquisition ID in the submission. If multiple images belonging to the reference acquisition are given as input to the task, the task raises an error, as the copy target would be ambiguous.
- Labels are copied without any spatial transformation. Running this task before registration will produce misaligned segmentations.
