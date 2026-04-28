## { MODULE

##
## === DEPENDENCIES
##

## stdlib
from pathlib import Path

## third-party
import numpy

## personal
from jormi import ww_lists


def read_file_lines(
    file_path: str | Path,
) -> list[str]:
    with open(file_path, "r") as file_pointer:
        return file_pointer.readlines()


def print_header(
    file_path: str | Path,
    header_names: list[str],
) -> None:
    print(f"Available datasets in: {file_path}")
    for dataset_index, dataset_name in enumerate(header_names):
        print(f"\tindex: {dataset_index:2d} - name: {dataset_name}")


def resolve_dataset_index(
    file_path: str | Path,
    dataset_index: int | None,
    dataset_name: str | None,
    header_names: list[str],
) -> int:
    if dataset_index is not None:
        return dataset_index
    if dataset_name is None:
        raise ValueError(
            "You need to either provide `dataset_index` or `dataset_name`.",
        )
    lookup_dataset_index = {
        "kin": 9,
        "mag": 11,
        "mach": 13,
    }
    dataset_name = dataset_name.lower()
    if dataset_name not in lookup_dataset_index:
        print_header(file_path, header_names)
        keys_string = ww_lists.as_string(elems=list(lookup_dataset_index.keys()))
        raise ValueError(
            f"`{dataset_name}` is an invalid dataset. "
            f"Choose from: {keys_string}, or provide `dataset_index` directly.",
        )
    return lookup_dataset_index[dataset_name]


def extract_data(
    lines: list[str],
    num_datasets: int,
    dataset_index: int,
    time_norm: float,
    raise_error: bool,
) -> tuple[list[float], list[float]]:
    ## iterates in reverse: when a simulation was restarted, only the most recent run's data is used
    time_index = 0
    prev_time = numpy.inf
    time_values, data_values = [], []
    for line in reversed(lines):
        line_content = line.strip().split()
        if len(line_content) != num_datasets:
            continue
        ## skip comment lines
        if "#" in line_content[time_index] or "#" in line_content[dataset_index]:
            continue
        try:
            time_value = float(line_content[time_index]) / time_norm
            data_value = float(line_content[dataset_index])
        except ValueError:
            continue
        ## discard entries from earlier runs (time has rewound past the current front)
        if time_value < prev_time:
            if data_value == 0.0 and time_value > 0:
                message = f"field[{dataset_index}] = 0.0 at time = {time_value:.3f}"
                if raise_error:
                    raise ValueError(f"Error: {message}")
                print(f"Warning: {message}")
                continue
            time_values.append(time_value)
            data_values.append(data_value)
            prev_time = time_value
    return (
        list(reversed(time_values)),
        list(reversed(data_values)),
    )


## } MODULE
