## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import h5py
import numpy

## local
from ww_flash_sims.sim_io import _read_snapshot_metadata

##
## === FUNCTIONS
##


def _reformat_flash_sfield(
    sfield: numpy.ndarray,
    num_blocks: tuple[int, int, int],
    num_cells_per_block: tuple[int, int, int],
) -> numpy.ndarray:
    ## vectorised and cache-friendly: intermediate arrays are views; only the final output is copied
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


def read_flash_field(
    file_path: str | Path,
    dataset_name: str,
    grid_properties: dict[str, Any] | None = None,
) -> numpy.ndarray:
    """
    Load a named field from a FLASH HDF5 output file as a sorted ndarray.

    Returns a 3D scalar array if a single dataset matches `dataset_name`,
    or a 4D array stacked along axis 0 if multiple datasets match.
    """
    if grid_properties is None:
        grid_properties = _read_snapshot_metadata.read_grid_properties(file_path)
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
        _dataset_name for _dataset_name in grid_properties["dataset_names"]
        if _dataset_name.startswith(dataset_name)
    ]
    if len(matched_dataset_names) == 0:
        raise KeyError(f"No datasets found starting with `{dataset_name}` in file {file_path}")
    with h5py.File(file_path, "r") as hdf5_file:
        raw_fields = [
            numpy.array(hdf5_file[_dataset_name]) for _dataset_name in sorted(matched_dataset_names)
        ]
    reformatted_fields = [
        _reformat_flash_sfield(sfield, num_blocks, num_cells_per_block) for sfield in raw_fields
    ]
    if len(matched_dataset_names) == 1:
        sfield = reformatted_fields[0]
        return sfield
    vfield = numpy.stack(reformatted_fields, axis=0)
    return vfield


## } MODULE
