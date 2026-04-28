## { MODULE

##
## === DEPENDENCIES
##

## stdlib
import argparse
from collections.abc import Callable
from typing import Any, TypedDict
from pathlib import Path

## personal
from jormi import ww_lists
from jormi.ww_fields import cartesian_axes

## local
from ww_flash_sims.sim_io.load_snapshot import FlashSnapshot

##
## === DEFAULT COLORMAPS
##

SEQUENTIAL_CMAP = "cmr.arctic"
DIVERGING_CMAP = "cmr.iceburn"

##
## === FIELD REGISTRY
##


class FieldEntry(TypedDict):
    loader: Callable[..., Any]
    cmap: str


def _field_entry(
    loader: Callable[..., Any],
    cmap: str,
) -> FieldEntry:
    return {
        "loader": loader,
        "cmap": cmap,
    }


FLASH_FIELD_LOOKUP: dict[str, FieldEntry] = {
    "mag": _field_entry(
        loader=FlashSnapshot.load_3d_magnetic_vfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "vel": _field_entry(
        loader=FlashSnapshot.load_3d_velocity_vfield,
        cmap=SEQUENTIAL_CMAP,
    ),
    "rho": _field_entry(
        loader=FlashSnapshot.load_3d_density_sfield,
        cmap=SEQUENTIAL_CMAP,
    ),
}

##
## === VALIDATION
##


def validate_fields(
    field_names: list[str] | tuple[str, ...] | None,
) -> None:
    valid_field_names = set(
        FLASH_FIELD_LOOKUP.keys(),
    )
    if not field_names or not set(field_names).issubset(valid_field_names):
        raise ValueError(f"provide fields via -f from: {sorted(valid_field_names)}.")


##
## === PARSER
##


def base_parser(
    allow_vfields: bool = True,
) -> argparse.ArgumentParser:
    """
    Shared argument parser for diagnostic scripts.

    Returns a base parser intended to be used as a parent via `parents=[base_parser()]`.

    Parameters
    ---
    - `allow_vfields`:
        If `True`, adds `--comps/-c` and `--axes/-a` for vector field components and slice axes.
    """
    field_list = ww_lists.as_string(
        elems=sorted(
            FLASH_FIELD_LOOKUP.keys(),
        ),
    )
    axis_list = ww_lists.as_string(
        elems=list(cartesian_axes.VALID_3D_AXIS_LABELS),
    )
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--sim-dir",
        "-d",
        type=lambda path: Path(path).expanduser().resolve(),
        required=True,
        help="Path to a FLASH simulation directory containing snapshot files.",
    )
    parser.add_argument(
        "--out-dir",
        "-o",
        type=lambda path: Path(path).expanduser().resolve(),
        default=None,
        help="Output directory for figures. Defaults to --sim-dir.",
    )
    parser.add_argument(
        "--fields",
        "-f",
        nargs="+",
        default=None,
        help=f"Fields to plot. Options: {field_list}",
    )
    if allow_vfields:
        parser.add_argument(
            "--comps",
            "-c",
            nargs="+",
            default=None,
            help=f"Vector field components to show. Options: {axis_list}",
        )
        parser.add_argument(
            "--axes",
            "-a",
            nargs="+",
            default=None,
            help=f"Axes to slice along. Options: {axis_list}",
        )
    return parser


## } MODULE
