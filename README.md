# fractal-helper-tasks

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
![Python version](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/fractal-analytics-platform/fractal-helper-tasks/build_and_test.yml?branch=main)
[![codecov](https://codecov.io/gh/fractal-analytics-platform/fractal-helper-tasks/graph/badge.svg?token=ednmg2GzOw)](https://codecov.io/gh/fractal-analytics-platform/fractal-helper-tasks)

Collection of Fractal helper tasks for working with OME-Zarr images in [Fractal](https://fractal-analytics-platform.github.io/) workflows.

## Tasks

**Drop T Dimension** — Removes a singleton T dimension from an OME-Zarr image. Optionally overwrites the input in place.

**Add Z Singleton** — Adds a singleton Z dimension to a 2D OME-Zarr, producing a 3D-compatible image. Optionally overwrites the input in place.

**Rechunk Zarr** — Rechunks an OME-Zarr image with user-defined chunk sizes per axis (e.g. `{"y": 4000, "x": 4000}`). Rebuilds pyramids and optionally applies the same rechunking to all label images.

**Convert 2D Segmentation to 3D** — Replicates a 2D label image along the Z axis of the corresponding 3D OME-Zarr. Useful when segmentation was run on a projected image but needs to be stored back in the 3D image. Supports copying associated feature tables.

**Label Assignment by Overlap** — Assigns child labels to parent labels based on spatial overlap. Stores results as a feature table with configurable overlap threshold.

**Pad Images to Same Size** — Extends zarr array shape metadata so that all images in the workflow share the same spatial dimensions (ZYX). No data is copied; regions outside the original extent return the fill value. Can group images by HCS plate (`pad_by_plate=True`) and optionally pads label images alongside each image (`pad_labels=True`, default). Useful to make e.g. the napari viewer work on the plate level for search-first images.

## Development instructions

To create the manifest:
```
pixi run create-manifest
```

Refer to the developers-guide in the [Fractal template repo](https://github.com/fractal-analytics-platform/fractal-tasks-template/blob/main/DEVELOPERS_GUIDE.md) for more detailed instructions.
