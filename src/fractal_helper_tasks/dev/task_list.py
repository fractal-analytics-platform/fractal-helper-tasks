"""Fractal Task list for Fractal Helper Tasks."""

from fractal_tasks_core.dev.task_models import NonParallelTask, ParallelTask

TASK_LIST = [
    NonParallelTask(
        name="Drop T Dimension",
        executable="drop_t_dimension.py",
        meta={"cpus_per_task": 2, "mem": 8000},
        output_types=dict(has_t=False),
        tags=["Singleton time dimension"],
        docs_info="file:docs_info/drop_t_dimension.md",
    ),
    ParallelTask(
        input_types=dict(is_3D=False),
        output_types=dict(is_3D=True),
        name="Convert 2D segmentation to 3D",
        executable="convert_2D_segmentation_to_3D.py",
        meta={"cpus_per_task": 2, "mem": 8000},
        tags=[
            "Mixed modality",
            "2D to 3D workflows",
        ],
        docs_info="file:docs_info/2d_to_3d.md",
    ),
    ParallelTask(
        name="Rechunk OME-Zarr",
        executable="rechunk_zarr.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        tags=[
            "Rechunking",
            "Many files",
        ],
        docs_info="file:docs_info/rechunk_zarr.md",
    ),
]
