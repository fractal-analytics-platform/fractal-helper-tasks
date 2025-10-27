"""Fractal Task list for Fractal Helper Tasks."""

from fractal_task_tools.task_models import ParallelTask

AUTHORS = "Joel Luethi"
DOCS_LINK = "https://github.com/jluethi/fractal-helper-tasks"
TASK_LIST = [
    ParallelTask(
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
    ParallelTask(
        name="Add Z Singleton Dimension",
        executable="add_z_singleton.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        input_types=dict(is_3D=False),
        tags=["Singleton Z dimension"],
        docs_info="file:docs_info/drop_t_dimension.md",
    ),
    ParallelTask(
        name="Assign Label by Overlap",
        executable="label_assignment_by_overlap.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        tags=["Label assignment", "Label processing"],
        docs_info="file:docs_info/label_assignment_by_overlap.md",
    ),
]
