### Purpose
- Fixes or rewrites the omero channel metadata (channel names, colors, display ranges) of OME-Zarr images.
- **Fix omero metadata** (default): keeps all existing omero metadata and fills in missing fields required by the OME-Zarr spec, e.g. channels missing the `window` field. Optionally applies per-channel overrides (color, window, wavelength_id), matched to existing channels by label.
- **Rewrite omero metadata**: replaces the omero metadata with the provided channels, one entry per channel in the image. Order matters: List the channels in the same order as they are provided in the Zarr. Channels are written with `active: true`, `coefficient: 1` and `inverted: false`. Works whether or not omero metadata already exists.
- Missing required fields get defaults: the `window` spans the range of the image data type (e.g. 0 - 65535 for uint16, 0 - 1 for float images) and the `color` is picked based on the channel label. The Fractal-specific `wavelength_id` is only set if provided or already present.
- Edits the Zarr attributes directly, so it works on images that ngio cannot open due to invalid omero metadata.

### Outputs
- Updated omero metadata in the Zarr attributes of each image.

### Limitations
- Only handles omero metadata; does not fix other metadata problems.
- Fix mode requires existing omero metadata with unique channel labels; otherwise use rewrite mode to set it from scratch.
- Edits Zarr metadata in-place and irreversibly.
