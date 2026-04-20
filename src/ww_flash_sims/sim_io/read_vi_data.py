## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from dataclasses import dataclass
from pathlib import Path

## third-party
import numpy

## personal
from jormi import ww_lists

##
## === CLASSES
##


@dataclass
class VIData:
    """
    VI time-series data extracted from a FLASH `.dat` file.

    Fields
    ---
    - `times`:
        1D array of simulation times, normalised by `time_norm`.

    - `values`:
        1D array of field values corresponding to each time.

    - `dataset_name`:
        Name of the dataset column as it appears in the file header.

    - `file_path`:
        Path to the source `.dat` file.
    """

    times: numpy.ndarray
    values: numpy.ndarray
    dataset_name: str
    file_path: Path

    def __post_init__(
        self,
    ):
        if len(self.times) != len(self.values):
            raise ValueError(
                f"`times` and `values` must have the same length; "
                f"got {len(self.times)} and {len(self.values)}."
            )


##
## === FUNCTIONS
##


def read_vi_data(
    directory: str | Path,
    file_name: str = "Turb.dat",
    dataset_name: str | None = None,
    dataset_index: int | None = None,
    time_norm: float = 1.0,
    start_time: float = 0.0,
    end_time: float | None = None,
    raise_error: bool = False,
    print_header: bool = False,
) -> VIData:
    """
    Read a named dataset from a FLASH volume-integrated `.dat` file.

    Restart overlaps are handled automatically: only the latest contiguous
    time sequence is kept. Raises if no valid data is found or the requested
    dataset cannot be resolved.

    Parameters
    ---
    - `dataset_name`:
        Shorthand alias for common datasets: `"kin"`, `"mag"`, `"mach"`.
        Mutually exclusive with `dataset_index`.

    - `dataset_index`:
        Direct zero-based column index. Mutually exclusive with `dataset_name`.

    - `end_time`:
        Upper bound of the time window in normalised units.
        Defaults to the last valid time in the file if `None`.

    - `raise_error`:
        If `True`, raises on zero-valued field entries at non-zero times.
        If `False`, warns and skips them.

    - `print_header`:
        If `True`, prints the available dataset names and raises without returning data.
    """
    file_path = Path(directory) / file_name
    if not file_path.is_file():
        raise FileNotFoundError(f"No .dat file found: {file_path}")
    file_lines = _read_file_lines(file_path)
    header_names = file_lines[0].split()
    num_datasets = len(header_names)
    if print_header:
        _print_header(file_path, header_names)
        raise ValueError("print_header mode does not return data.")
    dataset_index = _resolve_dataset_index(
        file_path=file_path,
        dataset_index=dataset_index,
        dataset_name=dataset_name,
        header_names=header_names,
    )
    times, values = _extract_data(
        lines=file_lines[1:],
        num_datasets=num_datasets,
        dataset_index=dataset_index,
        time_norm=time_norm,
        raise_error=raise_error,
    )
    if len(times) == 0:
        raise ValueError(f"No valid data extracted from: {file_path}")
    end_time = end_time if (end_time is not None) else times[-1]
    start_idx = ww_lists.get_index_of_closest_value(
        values=times,
        target=start_time,
    )
    end_idx = ww_lists.get_index_of_closest_value(
        values=times,
        target=end_time,
    )
    if start_idx == end_idx:
        end_idx = min(end_idx + 1, len(times))
    return VIData(
        times=numpy.array(times[start_idx:end_idx]),
        values=numpy.array(values[start_idx:end_idx]),
        dataset_name=header_names[dataset_index],
        file_path=file_path,
    )


def _read_file_lines(
    file_path: str | Path,
) -> list[str]:
    with open(file_path, "r") as file_pointer:
        return file_pointer.readlines()


def _print_header(
    file_path: str | Path,
    header_names: list[str],
) -> None:
    print(f"Available datasets in: {file_path}")
    for dataset_index, dataset_name in enumerate(header_names):
        print(f"\tindex: {dataset_index:2d} - name: {dataset_name}")


def _resolve_dataset_index(
    file_path: str | Path,
    dataset_index: int | None,
    dataset_name: str | None,
    header_names: list[str],
) -> int:
    if dataset_index is not None:
        return dataset_index
    if dataset_name is None:
        raise ValueError("You need to either provide `dataset_index` or `dataset_name`.")
    lookup_dataset_index = {
        "kin": 9,
        "mag": 11,
        "mach": 13,
    }
    dataset_name = dataset_name.lower()
    if dataset_name not in lookup_dataset_index:
        _print_header(file_path, header_names)
        raise ValueError(
            f"`{dataset_name}` is an invalid dataset. "
            f"Choose from: {ww_lists.as_string(list(lookup_dataset_index.keys()))}, or provide `dataset_index` directly.",
        )
    return lookup_dataset_index[dataset_name]


def _extract_data(
    lines: list[str],
    num_datasets: int,
    dataset_index: int,
    time_norm: float,
    raise_error: bool,
) -> tuple[list[float], list[float]]:
    ## iterates in reverse: when a simulation was restarted, only the most recent run's data is kept
    time_index = 0
    prev_time = numpy.inf
    times, values = [], []
    for line in reversed(lines):
        tokens = line.strip().split()
        if len(tokens) != num_datasets:
            continue
        ## skip comment lines
        if "#" in tokens[time_index] or "#" in tokens[dataset_index]:
            continue
        try:
            time_val = float(tokens[time_index]) / time_norm
            data_val = float(tokens[dataset_index])
        except ValueError:
            continue
        ## discard entries from earlier runs (time has rewound past the current front)
        if time_val < prev_time:
            if data_val == 0.0 and time_val > 0:
                message = f"field[{dataset_index}] = 0.0 at time = {time_val:.3f}"
                if raise_error:
                    raise ValueError(f"Error: {message}")
                print(f"Warning: {message}")
                continue
            times.append(time_val)
            values.append(data_val)
            prev_time = time_val
    return list(reversed(times)), list(reversed(values))


## } MODULE
