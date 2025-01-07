"""Test copy 2D to 3D segmentation."""

import ngio
import pytest

from fractal_helper_tasks.rechunk_zarr import (
    rechunk_zarr,
)


@pytest.mark.parametrize(
    "chunk_sizes, output_chunk_sizes",
    [
        ({"x": 1000, "y": 1000}, [1, 1, 1000, 1000]),
        ({"X": 1000, "Y": 1000}, [1, 1, 1000, 1000]),
        ({"x": 6000, "y": 6000}, [1, 1, 2160, 5120]),
        ({}, [1, 1, 2160, 2560]),
        ({"x": None, "y": None}, [1, 1, 2160, 2560]),
        ({"z": 10}, [1, 1, 2160, 2560]),
        ({"Z": 10}, [1, 1, 2160, 2560]),
    ],
)
def test_rechunk_2d(tmp_zenodo_zarr: list[str], chunk_sizes, output_chunk_sizes):
    zarr_url = f"{tmp_zenodo_zarr[1]}/B/03/0"

    rechunk_zarr(
        zarr_url=zarr_url,
        chunk_sizes=chunk_sizes,
    )

    chunks = ngio.NgffImage(zarr_url).get_image().on_disk_dask_array.chunks
    chunk_sizes = [c[0] for c in chunks]
    assert chunk_sizes == output_chunk_sizes


@pytest.mark.parametrize(
    "chunk_sizes, output_chunk_sizes",
    [
        ({"x": None, "y": None}, [1, 1, 2160, 2560]),
        ({"z": 10}, [1, 2, 2160, 2560]),
    ],
)
def test_rechunk_3d(tmp_zenodo_zarr: list[str], chunk_sizes, output_chunk_sizes):
    zarr_url = f"{tmp_zenodo_zarr[0]}/B/03/0"

    rechunk_zarr(
        zarr_url=zarr_url,
        chunk_sizes=chunk_sizes,
    )

    chunks = ngio.NgffImage(zarr_url).get_image().on_disk_dask_array.chunks
    chunk_sizes = [c[0] for c in chunks]
    assert chunk_sizes == output_chunk_sizes


@pytest.mark.parametrize(
    "rechunk_labels, output_chunk_sizes",
    [
        (True, [1, 300, 300]),
        (False, [1, 540, 1280]),
    ],
)
def test_rechunk_labels(tmp_zenodo_zarr: list[str], rechunk_labels, output_chunk_sizes):
    zarr_url = f"{tmp_zenodo_zarr[1]}/B/03/0"
    chunk_sizes = {"x": 300, "y": 300}

    rechunk_zarr(
        zarr_url=zarr_url,
        chunk_sizes=chunk_sizes,
        rechunk_labels=rechunk_labels,
    )
    chunks = (
        ngio.NgffImage(zarr_url)
        .labels.get_label(name="nuclei", path="0")
        .on_disk_dask_array.chunks
    )
    chunk_sizes = [c[0] for c in chunks]
    assert chunk_sizes == output_chunk_sizes


@pytest.mark.parametrize(
    "chunk_sizes, error_axes",
    [
        ({"test": 1000, "y": 1000}, "test"),
        ({"u": 1000}, "u"),
    ],
)
def test_invalid_axis(tmp_zenodo_zarr: list[str], chunk_sizes, error_axes):
    zarr_url = f"{tmp_zenodo_zarr[1]}/B/03/0"

    with pytest.raises(ValueError) as e:
        rechunk_zarr(
            zarr_url=zarr_url,
            chunk_sizes=chunk_sizes,
        )
    assert f"Axis {error_axes} is not supported" in str(e.value)


def test_rechunk_no_overwrite_input(tmp_zenodo_zarr: list[str]):
    zarr_url = f"{tmp_zenodo_zarr[1]}/B/03/0"
    suffix = "rechunked_custom"
    new_zarr_url = f"{zarr_url}_{suffix}"
    overwrite_input = False
    chunk_sizes = {"x": 1000, "y": 1000}
    output_chunk_sizes = [1, 1, 1000, 1000]
    original_chunk_sizes = [1, 1, 2160, 2560]

    output = rechunk_zarr(
        zarr_url=zarr_url,
        chunk_sizes=chunk_sizes,
        suffix=suffix,
        overwrite_input=overwrite_input,
    )
    expected_output = dict(
        image_list_updates=[
            dict(
                zarr_url=new_zarr_url,
                origin=zarr_url,
                types=dict(rechunked=True),
            )
        ],
        filters=dict(types=dict(rechunked=True)),
    )
    assert expected_output == output

    # Existing zarr should be unchanged, but new zarr should have
    # expected chunking
    chunks = ngio.NgffImage(zarr_url).get_image().on_disk_dask_array.chunks
    chunk_sizes = [c[0] for c in chunks]
    assert chunk_sizes == original_chunk_sizes

    chunks = ngio.NgffImage(new_zarr_url).get_image().on_disk_dask_array.chunks
    chunk_sizes = [c[0] for c in chunks]
    assert chunk_sizes == output_chunk_sizes
