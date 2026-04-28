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
from jormi.ww_fields.fields_3d import domain_models

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
        x_min = float(
            real_props.get(
                "xmin",
                -0.5,
            ),
        )
        x_max = float(
            real_props.get(
                "xmax",
                0.5,
            ),
        )
        y_min = float(
            real_props.get(
                "ymin",
                -0.5,
            ),
        )
        y_max = float(
            real_props.get(
                "ymax",
                0.5,
            ),
        )
        z_min = float(
            real_props.get(
                "zmin",
                -0.5,
            ),
        )
        z_max = float(
            real_props.get(
                "zmax",
                0.5,
            ),
        )
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


def read_uniform_domain(
    grid_properties: dict[str, Any],
) -> domain_models.UniformDomain_3D:
    """Construct a UniformDomain_3D from grid properties."""
    num_cells_x = grid_properties["num_blocks_x"] * grid_properties["num_cells_per_block_x"]
    num_cells_y = grid_properties["num_blocks_y"] * grid_properties["num_cells_per_block_y"]
    num_cells_z = grid_properties["num_blocks_z"] * grid_properties["num_cells_per_block_z"]
    return domain_models.UniformDomain_3D(
        periodicity=(True, True, True),
        resolution=(num_cells_x, num_cells_y, num_cells_z),
        domain_bounds=grid_properties["domain_bounds"],
    )


## } MODULE
