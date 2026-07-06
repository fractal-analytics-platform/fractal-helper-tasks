# fractal-helper-tasks

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
![Python version](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/fractal-analytics-platform/fractal-helper-tasks/build_and_test.yml?branch=main)
[![codecov](https://codecov.io/gh/fractal-analytics-platform/fractal-helper-tasks/graph/badge.svg?token=ednmg2GzOw)](https://codecov.io/gh/fractal-analytics-platform/fractal-helper-tasks)

Collection of Fractal helper tasks for working with OME-Zarr images in [Fractal](https://fractal-analytics-platform.github.io/) workflows.

## Tasks

### Image dimension tasks

**Drop T Dimension** — Removes a singleton T dimension from an OME-Zarr image. Optionally overwrites the input in place.

**Add Z Singleton** — Adds a singleton Z dimension to a 2D OME-Zarr, producing a 3D-compatible image. Overwrites the input in place by default.

**Rechunk Zarr** — Rechunks an OME-Zarr image with user-defined chunk sizes per axis (e.g. `{"y": 4000, "x": 4000}`). Rebuilds pyramids and optionally applies the same rechunking to all label images.

**Pad Images to Same Size (HCS)** — Extends Zarr array shape metadata so that all images in the plate share the same spatial dimensions (ZYX). No data is copied; regions outside the original extent return the fill value. Optionally pads label images alongside each image. Useful to fix display issues (e.g. ViZarr, napari plate overview, MoBie) caused by wells of unequal sizes in search-first acquisitions.

### Segmentation-related tasks

**Convert 2D Segmentation to 3D** — Replicates a 2D label image along the Z axis of the corresponding 3D OME-Zarr. Useful when segmentation was run on a projected image but needs to be stored back in the 3D image. Supports copying associated feature tables.

**Label Assignment by Overlap** — Assigns child labels to parent labels based on spatial overlap. Stores results as a feature table with configurable overlap threshold.

**Copy Labels to Multiplexing Acquisitions (HCS)** — Copies label images from a reference acquisition to all other acquisitions in the same HCS well. Must be run after registration has been applied to the non-reference images, so all acquisitions share the same coordinate space.

### Metadata and cleanup tasks

**Rename Channels** — Renames channels by supplying a mapping of old → new names. Only updates Zarr metadata; no arrays are rewritten. The mapping may contain more entries than a given image has channels, making it convenient for multiplexing workflows where different images carry different channel subsets.

**Delete Labels and Tables** — Deletes specified label images and/or tables from OME-Zarr images. Entries absent from a given image are silently skipped.

### ROI tasks

**ROI Selection** — Stores user-defined ROIs into an OME-Zarr ROI table. Each ROI is given by two corners in physical coordinates (as reported by the Vizarr viewer ROI plugin); X and Y are stored as-is, while Z and T are treated as slice/frame indices and converted using the image pixel size. Corners may be supplied in any order, ROIs whose origin lies within the image but whose extent overflows are clamped to the image bounds (with a warning) while an origin outside the bounds raises an error, and results are written to a configurable table (default `user_ROIs`) that can be appended to or overwritten.


## Installation instructions
You can install this task package on Fractal. It's recommended to install it via pixi:
1. Download the tar.gz from the release page
2. Add it to Tasks -> Manage tasks -> Pixi -> Upload tar.gz file

Alternatively, you can install it via the whl file or locally install the package as:
```
git clone https://github.com/fractal-analytics-platform/fractal-helper-tasks
cd fractal-helper-tasks
pip install .
```

## Development instructions

To create the manifest:
```
pixi run create-manifest
```

Refer to the developers-guide in the [Fractal template repo](https://github.com/fractal-analytics-platform/fractal-tasks-template/blob/main/DEVELOPERS_GUIDE.md) for more detailed instructions.
