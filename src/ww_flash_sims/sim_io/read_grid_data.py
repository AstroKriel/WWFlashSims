## { MODULE

##
## === DEPENDENCIES ===
##

import h5py
import numpy
from jormi.ww_io import manage_io as io_manager

##
## === FUNCTIONS ===
##


def read_grid_properties(file_path):
    io_manager.does_file_exist(file_path=file_path, raise_error=True)

    def _extract_properties(_h5file, dataset_name):
        return {str(key).split("'")[1].strip(): value for key, value in _h5file[dataset_name]}

    ## check that the file is the right type and has the right structure before proceeding
    properties = {}
    try:
        with h5py.File(file_path, "r") as hdf5_file:
            properties["plasma_datasets"] = [
                dataset_name for dataset_name in hdf5_file.keys()
                if any(dataset_name.startswith(prefix) for prefix in ("mag", "vel", "dens", "cur"))
            ]
            properties["int_scalars"] = _extract_properties(hdf5_file, "integer scalars")
            properties["int_properties"] = _extract_properties(
                hdf5_file,
                "integer runtime parameters",
            )
    except KeyError as exception:
        print(f"The group {exception} was not found in: {file_path}.")
        return {}
    except Exception as exception:
        print(f"An unexpected error occurred: {exception}")
        return {}
    if len(properties["plasma_datasets"]) == 0: print(f"No plasma datasets found in: {file_path}")
    try:
        output_num = properties["int_scalars"]["plotfilenumber"]
        dataset_names = properties["plasma_datasets"]
        num_blocks = numpy.int32(properties["int_scalars"]["globalnumblocks"])
        num_blocks_x = numpy.int32(properties["int_properties"]["iprocs"])
        num_blocks_y = numpy.int32(properties["int_properties"]["jprocs"])
        num_blocks_z = numpy.int32(properties["int_properties"]["kprocs"])
        num_cells_per_block_x = numpy.int32(properties["int_scalars"]["nxb"])
        num_cells_per_block_y = numpy.int32(properties["int_scalars"]["nyb"])
        num_cells_per_block_z = numpy.int32(properties["int_scalars"]["nzb"])
        num_cells_per_block = num_cells_per_block_x * num_cells_per_block_y * num_cells_per_block_z
        num_cells = num_blocks * num_cells_per_block
        return {
            "output_num": output_num,
            "dataset_names": dataset_names,
            "num_blocks": num_blocks,
            "num_blocks_x": num_blocks_x,
            "num_blocks_y": num_blocks_y,
            "num_blocks_z": num_blocks_z,
            "num_cells_per_block_x": num_cells_per_block_x,
            "num_cells_per_block_y": num_cells_per_block_y,
            "num_cells_per_block_z": num_cells_per_block_z,
            "num_cells": num_cells,
        }
    except KeyError as missing_key:
        print(f"Missing key `{missing_key}` in the extracted properties from: {file_path}")
        return {}


def _reformat_flash_sfield_v1(
    sfield: numpy.ndarray,
    num_blocks: tuple[int, int, int],
    num_cells_per_block: tuple[int, int, int],
    force_use: bool = False,
):
    """
  Deprecated FLASH scalar-field reformatter:
  Uses explicit block indexing; memory-friendly but overly verbose and loop-based.
  """
    if not force_use:
        print(
            "Warning: depreciated FLASH reformatter was called. Using the up-to-date and optimised version instead.",
        )
        return _reformat_flash_sfield_v3(sfield, num_blocks, num_cells_per_block)
    ## initialise output array with fortran-style axis ordering: [z, y, x]
    sfield_sorted = numpy.zeros(
        shape=(
            num_cells_per_block[2] * num_blocks[2],
            num_cells_per_block[1] * num_blocks[1],
            num_cells_per_block[0] * num_blocks[0],
        ),
        dtype=numpy.float32,
        order="C",  # use C-contiguous memory layout (for consistency with numpy operation)
    )
    ## copy each block into its corresponding region
    block_index = 0
    for index_block_z in range(num_blocks[2]):
        for index_block_y in range(num_blocks[1]):
            for index_block_x in range(num_blocks[0]):
                sfield_sorted[
                    index_block_z * num_cells_per_block[2]:(index_block_z + 1) * num_cells_per_block[2],
                    index_block_y * num_cells_per_block[1]:(index_block_y + 1) * num_cells_per_block[1],
                    index_block_x * num_cells_per_block[0]:(index_block_x + 1) * num_cells_per_block[0],
                ] = sfield[block_index, :, :, :]
                block_index += 1
    ## reorder axis from fortran-style [z, y, x] to C-style [x, y, z]
    return numpy.transpose(sfield_sorted, (2, 1, 0))


def _reformat_flash_sfield_v2(
    sfield: numpy.ndarray,
    num_blocks: tuple[int, int, int],
    num_cells_per_block: tuple[int, int, int],
    force_use: bool = False,
):
    """
  Deprecated FLASH scalar-field reformatter:
  Vectorised (avoids loops), but poor memory layout leads to inefficient cache usage.
  """
    if not force_use:
        print(
            "Warning: depreciated FLASH reformatter was called. Using the up-to-date and optimised version instead.",
        )
        return _reformat_flash_sfield_v3(sfield, num_blocks, num_cells_per_block)
    ## see version 3 for an explanation
    sfield_sorted = sfield.reshape(
        num_blocks[2],
        num_blocks[1],
        num_blocks[0],
        num_cells_per_block[2],
        num_cells_per_block[1],
        num_cells_per_block[0],
    )
    ## warning: this layout is not memory-contiguous along the merge axes; leads to inefficient cache usage during the reshape
    sfield_sorted = numpy.transpose(sfield_sorted, (2, 5, 1, 4, 0, 3))
    sfield_sorted = sfield_sorted.reshape(
        num_blocks[0] * num_cells_per_block[0],
        num_blocks[1] * num_cells_per_block[1],
        num_blocks[2] * num_cells_per_block[2],
    )
    return sfield_sorted


def _reformat_flash_sfield_v3(
    sfield: numpy.ndarray,
    num_blocks: tuple[int, int, int],
    num_cells_per_block: tuple[int, int, int],
):
    """
  Highly optimised FLASH scalar-field reformatter:
  Vectorised, cache- and memory-efficient (intermediate arrays are views, only the final output is copied).
  """
    ## separate the unified block structure into individual block components
    ## reshape from flat: [num_blocks, z, y, x]
    ## to structured: [num_blocks_z, num_blocks_y, num_blocks_x, cells_per_z, cells_per_y, cells_per_x]
    ## where num_blocks = num_blocks_x * num_blocks_y * num_blocks_z
    sfield = sfield.reshape(
        num_blocks[2],
        num_blocks[1],
        num_blocks[0],
        num_cells_per_block[2],
        num_cells_per_block[1],
        num_cells_per_block[0],
    )
    ## interleave block and cell dimensions in cache-friendly order
    sfield = numpy.transpose(sfield, (0, 3, 1, 4, 2, 5))
    ## merge block and cell dimensions
    sfield_sorted = sfield.reshape(
        num_blocks[2] * num_cells_per_block[2],
        num_blocks[1] * num_cells_per_block[1],
        num_blocks[0] * num_cells_per_block[0],
    )
    ## convert axis ordering from fortran-style [z, y, x] to C-style [x, y, z]
    sfield_sorted = sfield_sorted.transpose((2, 1, 0))
    return sfield_sorted
    # ## ensure the result is c-contiguous and owns its data
    # return numpy.ascontiguousarray(sfield_sorted)


def read_flash_field(
    file_path: str,
    dataset_name: str,
    grid_properties: dict | None = None,
) -> numpy.ndarray:
    if grid_properties is None:
        grid_properties = read_grid_properties(file_path)
        if not grid_properties:
            raise ValueError(f"FLASH grid properties could not be read from: {file_path}")
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
        _dataset_name for _dataset_name in grid_properties["dataset_names"] if _dataset_name.startswith(dataset_name)
    ]
    if len(matched_dataset_names) == 0:
        raise KeyError(f"No datasets found starting with `{dataset_name}` in file {file_path}")
    with h5py.File(file_path, "r") as hdf5_file:
        raw_fields = [numpy.array(hdf5_file[_dataset_name]) for _dataset_name in sorted(matched_dataset_names)]
    reformatted_fields = [_reformat_flash_sfield_v3(sfield, num_blocks, num_cells_per_block) for sfield in raw_fields]
    if len(matched_dataset_names) == 1:
        sfield = reformatted_fields[0]
        return sfield
    else:
        vfield = numpy.stack(reformatted_fields, axis=0)
        return vfield


## } MODULE
