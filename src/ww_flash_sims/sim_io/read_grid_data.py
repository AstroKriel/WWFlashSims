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

## personal (local)
from jormi.ww_fields.fields_3d import domain_types
from jormi.ww_fields.fields_3d import field_types

##
## === FUNCTIONS
##


def read_grid_properties(
    file_path: str | Path,
) -> dict[str, Any]:
    """
    Read block, cell, and domain metadata from a FLASH HDF5 output file.

    Returns an empty dict and prints a warning if the file cannot be read
    or required keys are missing.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"No FLASH file found: {file_path}")

    def _extract_properties(
        _h5file: Any,
        dataset_name: str,
    ) -> dict[str, Any]:
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
            try:
                properties["real_properties"] = _extract_properties(
                    hdf5_file,
                    "real runtime parameters",
                )
            except KeyError:
                properties["real_properties"] = {}
    except KeyError as exception:
        print(f"The group {exception} was not found in: {file_path}.")
        return {}
    except Exception as exception:
        print(f"An unexpected error occurred: {exception}")
        return {}
    if len(properties["plasma_datasets"]) == 0:
        print(f"No plasma datasets found in: {file_path}")
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
        real_props = properties["real_properties"]
        x_min = float(real_props.get("xmin", -0.5))
        x_max = float(real_props.get("xmax",  0.5))
        y_min = float(real_props.get("ymin", -0.5))
        y_max = float(real_props.get("ymax",  0.5))
        z_min = float(real_props.get("zmin", -0.5))
        z_max = float(real_props.get("zmax",  0.5))
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
            "domain_bounds": ((x_min, x_max), (y_min, y_max), (z_min, z_max)),
        }
    except KeyError as missing_key:
        print(f"Missing key `{missing_key}` in the extracted properties from: {file_path}")
        return {}


def compute_cell_widths(
    *,
    grid_properties: dict[str, Any],
) -> tuple[float, float, float]:
    """Derive uniform cell widths from grid properties and domain bounds."""
    num_cells_x = grid_properties["num_blocks_x"] * grid_properties["num_cells_per_block_x"]
    num_cells_y = grid_properties["num_blocks_y"] * grid_properties["num_cells_per_block_y"]
    num_cells_z = grid_properties["num_blocks_z"] * grid_properties["num_cells_per_block_z"]
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = grid_properties["domain_bounds"]
    return (
        (x_max - x_min) / num_cells_x,
        (y_max - y_min) / num_cells_y,
        (z_max - z_min) / num_cells_z,
    )


def read_uniform_domain(
    grid_properties: dict[str, Any],
) -> domain_types.UniformDomain_3D:
    """Construct a UniformDomain_3D from grid properties."""
    num_cells_x = grid_properties["num_blocks_x"] * grid_properties["num_cells_per_block_x"]
    num_cells_y = grid_properties["num_blocks_y"] * grid_properties["num_cells_per_block_y"]
    num_cells_z = grid_properties["num_blocks_z"] * grid_properties["num_cells_per_block_z"]
    return domain_types.UniformDomain_3D(
        periodicity=(True, True, True),
        resolution=(num_cells_x, num_cells_y, num_cells_z),
        domain_bounds=grid_properties["domain_bounds"],
    )


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
    else:
        vfield = numpy.stack(reformatted_fields, axis=0)
        return vfield


def read_flash_sfield(
    file_path: str | Path,
    dataset_name: str,
    *,
    field_label: str,
    grid_properties: dict[str, Any] | None = None,
) -> field_types.ScalarField_3D:
    """Load a named scalar field from a FLASH HDF5 output file as a ScalarField_3D."""
    if grid_properties is None:
        grid_properties = read_grid_properties(file_path)
        if not grid_properties:
            raise ValueError(f"FLASH grid properties could not be read from: {file_path}")
    sarray = read_flash_field(
        file_path,
        dataset_name,
        grid_properties=grid_properties,
    )
    udomain = read_uniform_domain(grid_properties)
    return field_types.ScalarField_3D.from_3d_sarray(
        sarray_3d=sarray,
        udomain_3d=udomain,
        field_label=field_label,
    )


def read_flash_vfield(
    file_path: str | Path,
    dataset_name: str,
    *,
    field_label: str,
    grid_properties: dict[str, Any] | None = None,
) -> field_types.VectorField_3D:
    """
    Load a named vector field from a FLASH HDF5 output file as a VectorField_3D.

    Expects exactly three component datasets matching `dataset_name` (e.g. `magx`,
    `magy`, `magz` for `dataset_name="mag"`), stacked in alphabetical order along axis 0.
    """
    if grid_properties is None:
        grid_properties = read_grid_properties(file_path)
        if not grid_properties:
            raise ValueError(f"FLASH grid properties could not be read from: {file_path}")
    varray = read_flash_field(
        file_path,
        dataset_name,
        grid_properties=grid_properties,
    )
    udomain = read_uniform_domain(grid_properties)
    return field_types.VectorField_3D.from_3d_varray(
        varray_3d=varray,
        udomain_3d=udomain,
        field_label=field_label,
    )


## } MODULE
