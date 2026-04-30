## { SCRIPT

##
## === DEPENDENCIES
##

## stdlib
import argparse
from collections.abc import Callable
from typing import Any
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy
from matplotlib.axes import Axes as mpl_Axes

## personal
from jormi.ww_fields import cartesian_axes
from jormi.ww_fields.fields_3d import domain_models, field_models
from jormi.ww_io import manage_io, manage_log
from jormi.ww_plots import add_color, annotate_axis, manage_plots, plot_data
from jormi.ww_validation import validate_types
from jormi.ww_types import box_positions

## local
from ww_flash_sims.sim_io.load_snapshot import FlashSnapshot
import flash_fields

##
## === DATA STRUCTURES
##


@dataclass(frozen=True)
class FieldComp:
    data_3d: numpy.ndarray
    label: str


AxisBounds = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class SlicedField:
    data_2d: numpy.ndarray
    label: str
    axis_bounds: AxisBounds


##
## === SNAPSHOT HELPERS
##


def collect_snapshot_paths(
    sim_dir: Path,
) -> list[Path]:
    return manage_io.filter_directory(
        sim_dir,
        prefix="Turb_hdf5_plt_cnt_",
        num_parts=5,
        include_folders=False,
    )


def get_snapshot_index(
    snapshot_path: Path,
) -> str:
    return snapshot_path.name.removeprefix("Turb_hdf5_plt_cnt_")


##
## === FIELD PROCESSING
##


def _parse_axes(
    *,
    axes: tuple[str, ...] | list[str] | None,
) -> tuple[cartesian_axes.CartesianAxis_3D, ...]:
    if axes is None:
        return tuple(cartesian_axes.DEFAULT_3D_AXES_ORDER)
    parsed_axes: list[cartesian_axes.CartesianAxis_3D] = []
    for axis_name in validate_types.as_tuple(param=axes):
        try:
            parsed_axes.append(
                cartesian_axes.as_axis(axis=axis_name),
            )
        except (TypeError, ValueError):
            raise ValueError(
                f"invalid axis: `{axis_name}`; choose from: {list(cartesian_axes.VALID_3D_AXIS_LABELS)}.",
            )
    return tuple(parsed_axes)


def get_slice_bounds(
    *,
    uniform_domain: domain_models.UniformDomain_3D,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
) -> AxisBounds:
    (x0_min, x0_max), (x1_min, x1_max), (x2_min, x2_max) = uniform_domain.domain_bounds
    if axis_to_slice == cartesian_axes.CartesianAxis_3D.X2:
        return ((x0_min, x0_max), (x1_min, x1_max))
    if axis_to_slice == cartesian_axes.CartesianAxis_3D.X1:
        return ((x0_min, x0_max), (x2_min, x2_max))
    return ((x1_min, x1_max), (x2_min, x2_max))


def get_slice_labels(
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
) -> tuple[str, str]:
    axes_in_plane = [ax for ax in cartesian_axes.DEFAULT_3D_AXES_ORDER if ax != axis_to_slice]
    return (
        axes_in_plane[0].axis_label
        if "$" in axes_in_plane[0].axis_label else f"${axes_in_plane[0].axis_label}$",
        axes_in_plane[1].axis_label
        if "$" in axes_in_plane[1].axis_label else f"${axes_in_plane[1].axis_label}$",
    )


def slice_field(
    *,
    data_3d: numpy.ndarray,
    axis_to_slice: cartesian_axes.CartesianAxis_3D,
    uniform_domain: domain_models.UniformDomain_3D,
) -> SlicedField:
    num_cells_x0, num_cells_x1, num_cells_x2 = data_3d.shape
    if axis_to_slice == cartesian_axes.CartesianAxis_3D.X2:
        data_2d = data_3d[:, :, num_cells_x2 // 2]
    elif axis_to_slice == cartesian_axes.CartesianAxis_3D.X1:
        data_2d = data_3d[:, num_cells_x1 // 2, :]
    else:
        data_2d = data_3d[num_cells_x0 // 2, :, :]
    label_parts = [
        rf"{ax.axis_label}=L_{ax.axis_index}/2" if ax == axis_to_slice else ax.axis_label
        for ax in cartesian_axes.DEFAULT_3D_AXES_ORDER
    ]
    label = "$(" + ", ".join(label_parts) + ")$"
    return SlicedField(
        data_2d=data_2d,
        label=label,
        axis_bounds=get_slice_bounds(
            uniform_domain=uniform_domain,
            axis_to_slice=axis_to_slice,
        ),
    )


def get_field_comps(
    *,
    field: field_models.ScalarField_3D | field_models.VectorField_3D,
    field_name: str,
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...],
) -> list[FieldComp]:
    if isinstance(field, field_models.ScalarField_3D):
        return [
            FieldComp(
                data_3d=field_models.extract_3d_sarray(field),
                label=field_models.get_label(field),
            ),
        ]
    if isinstance(field, field_models.VectorField_3D):
        if not comps_to_plot:
            raise ValueError(
                f"vector field `{field_name}` requires at least one component; none provided via -c.",
            )
        varray_3d = field_models.extract_3d_varray(field)
        return [
            FieldComp(
                data_3d=varray_3d[cartesian_axes.get_axis_index(comp_axis)],
                label=field_models.get_vcomp_label(field, comp_axis=comp_axis),
            ) for comp_axis in comps_to_plot
        ]
    raise ValueError(f"unrecognised field type for `{field_name}`.")


##
## === FIGURE RENDERING
##


def plot_slice(
    *,
    ax: mpl_Axes,
    field_slice: SlicedField,
    label: str,
    cmap_name: str,
) -> None:
    min_value = float(numpy.nanmin(field_slice.data_2d))
    max_value = float(numpy.nanmax(field_slice.data_2d))
    plot_data.plot_2d_array(
        ax=ax,
        array_2d=field_slice.data_2d,
        data_format="xy",
        axis_aspect_ratio="equal",
        axis_bounds=field_slice.axis_bounds,
        cbar_bounds=(min_value, max_value),
        palette_config=add_color.SequentialConfig(palette_name=cmap_name),
        add_cbar=True,
        cbar_label=label,
        cbar_side=box_positions.Positions.Side.Right,
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.5,
        y_pos=0.95,
        x_alignment=box_positions.MPLPositions.Align.Center.Center,
        y_alignment=box_positions.MPLPositions.Align.Side.Top,
        label=f"min = {min_value:.2e}\nmax = {max_value:.2e}",
        text_size=14,
        box_alpha=0.5,
    )
    annotate_axis.add_text(
        ax=ax,
        x_pos=0.5,
        y_pos=0.05,
        x_alignment=box_positions.MPLPositions.Align.Center.Center,
        y_alignment=box_positions.MPLPositions.Align.Side.Bottom,
        label=field_slice.label,
        text_size=14,
        box_alpha=0.5,
    )


def render_snapshot(
    *,
    snapshot_path: Path,
    field_name: str,
    field_loader: Callable[..., Any],
    comps_to_plot: tuple[cartesian_axes.CartesianAxis_3D, ...],
    axes_to_slice: tuple[cartesian_axes.CartesianAxis_3D, ...],
    cmap_name: str,
    out_dir: Path,
) -> None:
    snapshot = FlashSnapshot(snapshot_path)
    field = field_loader(snapshot)
    uniform_domain = snapshot._get_udomain()
    field_comps = get_field_comps(
        field=field,
        field_name=field_name,
        comps_to_plot=comps_to_plot,
    )
    num_rows = len(field_comps)
    num_cols = len(axes_to_slice)
    fig, axs_grid = manage_plots.create_figure_grid(
        num_rows=num_rows,
        num_cols=num_cols,
        x_spacing=0.75,
        y_spacing=0.25,
    )
    for row_index, field_comp in enumerate(field_comps):
        for col_index, axis_to_slice in enumerate(axes_to_slice):
            ax = axs_grid[row_index][col_index]
            field_slice = slice_field(
                data_3d=field_comp.data_3d,
                axis_to_slice=axis_to_slice,
                uniform_domain=uniform_domain,
            )
            plot_slice(
                ax=ax,
                field_slice=field_slice,
                label=field_comp.label,
                cmap_name=cmap_name,
            )
            x_label, y_label = get_slice_labels(axis_to_slice)
            if (num_rows == 1) or (row_index == num_rows - 1):
                ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
    snapshot_index = get_snapshot_index(snapshot_path)
    fig_path = out_dir / f"{field_name}_slice_{snapshot_index}.png"
    manage_plots.save_figure(
        fig=fig,
        fig_path=fig_path,
    )


##
## === PROGRAM MAIN
##


def main() -> None:
    manage_log.set_block_width_mode(manage_log.BlockWidthMode.PRACTICAL)
    parser = argparse.ArgumentParser(
        description="Plot midplane slices of FLASH field components.",
        parents=[flash_fields.base_parser(allow_vfields=True)],
    )
    user_args = parser.parse_args()
    sim_dir: Path = user_args.sim_dir
    out_dir: Path = user_args.out_dir if user_args.out_dir is not None else sim_dir
    flash_fields.validate_fields(user_args.fields)
    fields_to_plot: tuple[str, ...] = tuple(user_args.fields)
    comps_to_plot = _parse_axes(axes=user_args.comps)
    axes_to_slice = _parse_axes(axes=user_args.axes)
    if not sim_dir.is_dir():
        raise FileNotFoundError(f"simulation directory not found: {sim_dir}.")
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    snapshot_paths = collect_snapshot_paths(sim_dir)
    if not snapshot_paths:
        manage_log.log_hint(text=f"no snapshots found under {sim_dir}.")
        return
    manage_log.log_note(text=f"found {len(snapshot_paths)} snapshot(s) under {sim_dir}.")
    for field_name in fields_to_plot:
        field_meta = flash_fields.FLASH_FIELD_LOOKUP[field_name]
        for snapshot_path in snapshot_paths:
            manage_log.log_task(text=f"plotting `{field_name}`: {snapshot_path.name}")
            render_snapshot(
                snapshot_path=snapshot_path,
                field_name=field_name,
                field_loader=field_meta["loader"],
                comps_to_plot=comps_to_plot,
                axes_to_slice=axes_to_slice,
                cmap_name=field_meta["cmap"],
                out_dir=out_dir,
            )


##
## === ENTRY POINT
##

if __name__ == "__main__":
    main()

## } SCRIPT
