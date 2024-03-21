"""Dev script to test the Drop T dimension task."""
from fractal_helper_tasks.drop_t_dimension import drop_t_dimension

input_paths = [
    "/Users/joel/Desktop/dyn_CHX_pos1_mmIntOrg_121z_0-2um_640-1000ms-200g_561-1000ms-200g__488-1000ms-200g_405-100ms-200g_11.zarr"
]
output_path = ""
component = "0"

drop_t_dimension(
    input_paths=input_paths,
    output_path=output_path,
    component=component,
    metadata={},
    suffix="no_T",
    overwrite_input=False,
)
