"""Fractal Task list for Fractal Helper Tasks."""

from fractal_task_tools.task_models import (
    CompoundTask,
    NonParallelTask,
    ParallelTask,
)

AUTHORS = "Joel Luethi"


DOCS_LINK = "https://github.com/fractal-analytics-platform/fractal-helper-tasks"


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
    NonParallelTask(
        name="Pad Images to Same Size",
        executable="pad_images_to_same_size.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        tags=["Padding", "HCS plate"],
        docs_info="file:docs_info/pad_images_to_same_size.md",
    ),
    NonParallelTask(
        name="Delete Labels and Tables",
        executable="delete_labels_tables.py",
        meta={"cpus_per_task": 1, "mem": 1000},
        tags=["Labels", "Tables", "Cleanup"],
        docs_info="file:docs_info/delete_labels_tables.md",
    ),
    NonParallelTask(
        name="Rename Channels",
        executable="rename_channels.py",
        meta={"cpus_per_task": 1, "mem": 1000},
        tags=["Channels", "Metadata"],
        docs_info="file:docs_info/rename_channels.md",
    ),
    CompoundTask(
        name="Copy Labels to Multiplexing Acquisitions (HCS)",
        executable_init="copy_labels_multiplexing_init.py",
        meta_init={"cpus_per_task": 1, "mem": 4000},
        executable="copy_labels_multiplexing.py",
        meta={"cpus_per_task": 1, "mem": 4000},
        tags=["HCS", "Multiplexing", "Labels"],
        docs_info="file:docs_info/copy_labels_multiplexing.md",
    ),
]
