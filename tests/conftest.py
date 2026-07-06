import os
import shutil
from pathlib import Path

import ngio
import numpy as np
import pooch
import pytest


@pytest.fixture
def test_data_3d_path(tmp_path: Path) -> Path:
    """A synthetic 3D (CZYX) OME-Zarr image for ROI selection tests."""
    zarr_url = tmp_path / "test_3d.zarr"
    ngio.create_ome_zarr_from_array(
        store=str(zarr_url),
        array=np.zeros((1, 10, 100, 100)),
        pixelsize=0.5,
        z_spacing=1.0,
        axes_names="czyx",
        overwrite=True,
    )
    return zarr_url


@pytest.fixture
def test_data_2d_path(tmp_path: Path) -> Path:
    """A synthetic 2D-like (CZYX with singleton Z) OME-Zarr image."""
    zarr_url = tmp_path / "test_2d.zarr"
    ngio.create_ome_zarr_from_array(
        store=str(zarr_url),
        array=np.zeros((1, 1, 100, 100)),
        pixelsize=0.5,
        z_spacing=1.0,
        axes_names="czyx",
        overwrite=True,
    )
    return zarr_url


@pytest.fixture(scope="session")
def testdata_path() -> Path:
    TEST_DIR = Path(__file__).parent
    return TEST_DIR / "data/"


@pytest.fixture(scope="session")
def zenodo_zarr(testdata_path: Path) -> list[str]:
    """
    This takes care of multiple steps:

    1. Download/unzip two Zarr containers (3D and MIP) from Zenodo, via pooch
    2. Copy the two Zarr containers into tests/data
    3. Modify the Zarrs in tests/data, to add whatever is not in Zenodo
    """

    # 1 Download Zarrs from Zenodo
    DOI = "10.5281/zenodo.10257149"
    DOI_slug = DOI.replace("/", "_").replace(".", "_")
    platenames = ["plate.zarr", "plate_mip.zarr"]
    rootfolder = testdata_path / DOI_slug
    folders = [rootfolder / plate for plate in platenames]

    registry = {
        "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr.zip": None,
        "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr.zip": None,
    }
    base_url = f"doi:{DOI}"
    POOCH = pooch.create(
        pooch.os_cache("pooch") / DOI_slug,
        base_url,
        registry=registry,
        retry_if_failed=10,
        allow_updates=False,
    )
    downloader = pooch.DOIDownloader(
        headers={
            "User-Agent": (
                f"pooch/{pooch.__version__} "
                "(https://github.com/fractal-analytics-platform/fractal-helper-tasks)"
            )
        }
    )

    for ind, file_name in enumerate(
        [
            "20200812-CardiomyocyteDifferentiation14-Cycle1.zarr",
            "20200812-CardiomyocyteDifferentiation14-Cycle1_mip.zarr",
        ]
    ):
        # 1) Download/unzip a single Zarr from Zenodo
        file_paths = POOCH.fetch(
            f"{file_name}.zip",
            processor=pooch.Unzip(extract_dir=file_name),
            downloader=downloader,
        )
        zarr_full_path = file_paths[0].split(file_name)[0] + file_name
        folder = folders[ind]

        # 2) Copy the downloaded Zarr into tests/data
        if os.path.isdir(str(folder)):
            shutil.rmtree(str(folder))
        shutil.copytree(Path(zarr_full_path) / file_name, folder, ignore=shutil.ignore_patterns(".DS_Store"))
    return [str(f) for f in folders]


@pytest.fixture(scope="function")
def tmp_zenodo_zarr(zenodo_zarr: list[str], tmpdir: Path) -> list[str]:
    """Generates a copy of the zenodo zarrs in a tmpdir"""
    zenodo_mip_path = str(tmpdir / Path(zenodo_zarr[1]).name)
    zenodo_path = str(tmpdir / Path(zenodo_zarr[0]).name)
    shutil.copytree(zenodo_zarr[0], zenodo_path, ignore=shutil.ignore_patterns(".DS_Store"))
    shutil.copytree(zenodo_zarr[1], zenodo_mip_path, ignore=shutil.ignore_patterns(".DS_Store"))
    return [zenodo_path, zenodo_mip_path]
