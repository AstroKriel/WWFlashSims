## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path
from typing import Any

## third-party
import numpy

## personal
from jormi.ww_fields.fields_3d import domain_types, field_types

## local
from ww_flash_sims.sim_io import _read_grid_data

##
## === SNAPSHOT CLASS
##


class FlashSnapshot:
    """Interface for loading fields from a single FLASH HDF5 snapshot."""

    def __init__(
        self,
        snap_path: str | Path,
    ) -> None:
        self.snapshot_path: Path = Path(snap_path)
        self._grid_properties: dict[str, Any] | None = None
        self._udomain: domain_types.UniformDomain_3D | None = None

    ##
    ## --- INFRASTRUCTURE
    ##

    def _get_grid_properties(
        self,
    ) -> dict[str, Any]:
        if self._grid_properties is None:
            self._grid_properties = _read_grid_data.read_grid_properties(
                file_path=self.snapshot_path
            )
            if not self._grid_properties:
                raise ValueError(
                    f"FLASH grid properties could not be read from: {self.snapshot_path}"
                )
        return self._grid_properties

    def _get_udomain(
        self,
    ) -> domain_types.UniformDomain_3D:
        if self._udomain is None:
            self._udomain = _read_grid_data.read_uniform_domain(
                self._get_grid_properties()
            )
        return self._udomain

    def _load_field(
        self,
        *,
        dataset_name: str,
        expected_num_components: int,
    ) -> numpy.ndarray:
        grid_properties = self._get_grid_properties()
        array = _read_grid_data.read_flash_field(
            self.snapshot_path,
            dataset_name,
            grid_properties=grid_properties,
        )
        if expected_num_components == 1:
            if array.ndim != 3:
                raise ValueError(
                    f"Expected a scalar field for `{dataset_name}`; "
                    f"got shape {array.shape}: matched multiple components.",
                )
        elif array.ndim != 4 or array.shape[0] != expected_num_components:
            raise ValueError(
                f"Expected {expected_num_components} components for `{dataset_name}`; "
                f"got shape {array.shape}.",
            )
        return array

    ##
    ## --- FIELD LOADERS
    ##

    def load_3d_magnetic_vfield(
        self,
    ) -> field_types.VectorField_3D:
        """Load magnetic field: vec(b)."""
        varray_3d = self._load_field(
            dataset_name="mag",
            expected_num_components=3,
        )
        return field_types.VectorField_3D.from_3d_varray(
            varray_3d=varray_3d,
            udomain_3d=self._get_udomain(),
            field_label=r"\vec{b}",
        )

    def load_3d_velocity_vfield(
        self,
    ) -> field_types.VectorField_3D:
        """Load velocity field: vec(v)."""
        varray_3d = self._load_field(
            dataset_name="vel",
            expected_num_components=3,
        )
        return field_types.VectorField_3D.from_3d_varray(
            varray_3d=varray_3d,
            udomain_3d=self._get_udomain(),
            field_label=r"\vec{v}",
        )

    def load_3d_density_sfield(
        self,
    ) -> field_types.ScalarField_3D:
        """Load gas density: rho."""
        sarray_3d = self._load_field(
            dataset_name="dens",
            expected_num_components=1,
        )
        return field_types.ScalarField_3D.from_3d_sarray(
            sarray_3d=sarray_3d,
            udomain_3d=self._get_udomain(),
            field_label=r"\rho",
        )


## } MODULE
