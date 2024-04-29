"""Fractal Task list for Fractal Helper Tasks."""

from fractal_tasks_core.dev.task_models import NonParallelTask, ParallelTask

TASK_LIST = [
    NonParallelTask(
        name="Drop T Dimension",
        executable="drop_t_dimension.py",
        meta={"cpus_per_task": 2, "mem": 8000},
    ),
    ParallelTask(
        name="Convert 2D segmentation to 3D",
        executable="convert_2D_segmentation_to_3D.py",
        meta={"cpus_per_task": 2, "mem": 8000},
    ),
]
