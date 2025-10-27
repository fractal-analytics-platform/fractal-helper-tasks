### Purpose
- Assigns labels of child label to the parent label based on overlap.
- Uses a threshold and fills NA for labels with no sufficient overlap.
- When multiple parent labels overlap with a child label, assigns the parent label with the maximum overlap.

### Outputs
- A new FeatureTable or an addition to an existing FeatureTable with the overlap measurements & assignments.

### Limitations
- Only processes OME-Zarrs where both parent and child label are in the same OME-Zarr image (in some multiplexing scenarios, those labels are in different images).