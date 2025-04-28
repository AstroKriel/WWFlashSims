## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import h5py
import numpy
from jormi.ww_io import io_manager


## ###############################################################
## FUNCTIONS
## ###############################################################
def read_grid_properties(file_path):
  io_manager.does_file_exist(file_path=file_path, raise_error=True)
  def _extract_properties(_h5file, dataset_name):
    return {
      str(key).split("'")[1].strip() : value
      for key, value in _h5file[dataset_name]
    }
  ## check that the file is the right type and has the right structure before proceeding
  properties = {}
  try:
    with h5py.File(file_path, "r") as h5file:
      properties["plasma_datasets"] = [
        dataset_name
        for dataset_name in h5file.keys()
        if any(
          dataset_name.startswith(prefix)
          for prefix in ("mag", "vel", "dens", "cur")
        )
      ]
      properties["int_scalars"]    = _extract_properties(h5file, "integer scalars")
      properties["int_properties"] = _extract_properties(h5file, "integer runtime parameters")
  except KeyError as exception:
    print(f"The group {exception} was not found in: {file_path}.")
    return {}
  except Exception as exception:
    print(f"An unexpected error occurred: {exception}")
    return {}
  if len(properties["plasma_datasets"]) == 0: print(f"No plasma datasets found in: {file_path}")
  try:
    output_num    = properties["int_scalars"]["plotfilenumber"]
    dataset_names = properties["plasma_datasets"]
    num_blocks    = numpy.int32(properties["int_scalars"]["globalnumblocks"])
    num_blocks_x  = numpy.int32(properties["int_properties"]["iprocs"])
    num_blocks_y  = numpy.int32(properties["int_properties"]["jprocs"])
    num_blocks_z  = numpy.int32(properties["int_properties"]["kprocs"])
    num_cells_per_block_x = numpy.int32(properties["int_scalars"]["nxb"])
    num_cells_per_block_y = numpy.int32(properties["int_scalars"]["nyb"])
    num_cells_per_block_z = numpy.int32(properties["int_scalars"]["nzb"])
    num_cells_per_block   = num_cells_per_block_x * num_cells_per_block_y * num_cells_per_block_z
    num_cells             = num_blocks * num_cells_per_block
    return {
      "output_num"            : output_num,
      "dataset_names"         : dataset_names,
      "num_blocks"            : num_blocks,
      "num_blocks_x"          : num_blocks_x,
      "num_blocks_y"          : num_blocks_y,
      "num_blocks_z"          : num_blocks_z,
      "num_cells_per_block_x" : num_cells_per_block_x,
      "num_cells_per_block_y" : num_cells_per_block_y,
      "num_cells_per_block_z" : num_cells_per_block_z,
      "num_cells"             : num_cells,
    }
  except KeyError as missing_key:
    print(f"Missing key `{missing_key}` in the extracted properties from: {file_path}")
    return {}

def _reformat_flash_sfield(
    sfield              : numpy.ndarray,
    num_blocks          : tuple[int, int, int],
    num_cells_per_block : tuple[int, int, int],
  ):
  ## input Fortran-style field (column-major: z, y, x) with shape:
  ## [total_number_of_blocks, num_cells_per_block_z, num_cells_per_block_y, num_cells_per_block_x]
  ## where total_number_of_blocks = num_blocks_x * num_blocks_y * num_blocks_z
  ## reshape this field to separate the unified block-structure into its individual block components:
  ## total_number_of_blocks -> [num_blocks_z, num_blocks_y, num_blocks_x]
  sfield = sfield.reshape(
    num_blocks[2], num_blocks[1], num_blocks[0],
    num_cells_per_block[2], num_cells_per_block[1], num_cells_per_block[0]
  )
  ## interleave blocks with their cells
  sfield = numpy.transpose(sfield, (0, 3, 1, 4, 2, 5))
  ## merge block and cell dimensions
  sfield_sorted = sfield.reshape(
    num_blocks[2] * num_cells_per_block[2],
    num_blocks[1] * num_cells_per_block[1],
    num_blocks[0] * num_cells_per_block[0],
  )
  ## convert from Fortran-style [z, y, x] to C-style [x, y, z] cell ordering
  return sfield_sorted.transpose((2, 1, 0))

def read_flash_field(
    file_path       : str,
    dataset_name    : str,
    grid_properties : dict | None = None,
  ) -> numpy.ndarray:
  if grid_properties is None:
    grid_properties = read_grid_properties(file_path)
    if not grid_properties: raise ValueError(f"FLASH grid properties could not be read from: {file_path}")
  num_blocks = (
    grid_properties["num_blocks_x"],
    grid_properties["num_blocks_y"],
    grid_properties["num_blocks_z"],
  )
  num_cells_per_block = (
    grid_properties["num_cells_per_block_x"],
    grid_properties["num_cells_per_block_y"],
    grid_properties["num_cells_per_block_z"],
  )
  matched_dataset_names = [
    _dataset_name
    for _dataset_name in grid_properties["dataset_names"]
    if _dataset_name.startswith(dataset_name)
  ]
  if len(matched_dataset_names) == 0: raise KeyError(f"No datasets found starting with `{dataset_name}` in file {file_path}")
  with h5py.File(file_path, "r") as h5file:
    raw_fields = [
      numpy.array(h5file[_dataset_name])
      for _dataset_name in sorted(matched_dataset_names)
    ]
  reformatted_fields = [
    _reformat_flash_sfield(sfield, num_blocks, num_cells_per_block)
    for sfield in raw_fields
  ]
  if len(matched_dataset_names) == 1:
    sfield = reformatted_fields[0]
    return sfield
  else:
    vfield = numpy.stack(reformatted_fields, axis=0)
    return vfield


## END OF MODULE