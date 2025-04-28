## START OF MODULE


## ###############################################################
## DEPENDENCIES
## ###############################################################
import numpy
from pathlib import Path
from jormi.utils import list_utils
from jormi.ww_io import io_manager


## ###############################################################
## FUNCTIONS
## ###############################################################
def read_vi_data(
  directory     : str | Path,
  file_name     : str = "Turb.dat",
  dataset_name  : str | None = None,
  dataset_index : int | None = None,
  time_norm     : float = 1.0,
  start_time    : float = 0.0,
  end_time      : float | None = None,
  raise_error   : bool = False,
  print_header  : bool = False,
) -> tuple[list[float], list[float]]:
  file_path = io_manager.combine_file_path_parts([ directory, file_name ])
  io_manager.does_file_exist(file_path=file_path, raise_error=True)
  file_lines   = _read_file_lines(file_path)
  header_names = file_lines[0].split()
  num_datasets = len(header_names)
  if print_header:
    _print_header(file_path, header_names)
    return [], []
  dataset_index = _resolve_dataset_index(
    file_path     = file_path,
    dataset_index = dataset_index,
    dataset_name  = dataset_name,
    header_names  = header_names,
  )
  times, values = _extract_data(
    lines         = file_lines[1:],
    num_datasets  = num_datasets,
    dataset_index = dataset_index,
    time_norm     = time_norm,
    raise_error   = raise_error
  )
  if len(times) == 0: return [], []
  end_time  = end_time if (end_time is not None) else times[-1]
  start_idx = list_utils.get_index_of_closest_value(times, start_time)
  end_idx   = list_utils.get_index_of_closest_value(times, end_time)
  if start_idx == end_idx: end_idx = numpy.min(end_idx+1, len(times))
  return numpy.array(times[start_idx:end_idx]), numpy.array(values[start_idx:end_idx])

def _read_file_lines(file_path: str | Path) -> list[str]:
  with open(file_path, "r") as file_pointer:
    return file_pointer.readlines()

def _print_header(
    file_path    : str | Path,
    header_names : list[str]
  ):
  print(f"Available datasets in: {file_path}")
  for dataset_index, dataset_name in enumerate(header_names):
    print(f"\tindex: {dataset_index:2d} - name: {dataset_name}")

def _resolve_dataset_index(
  file_path     : str | Path,
  dataset_index : int | None,
  dataset_name  : str | None,
  header_names  : list[str],
) -> int:
  if dataset_index is not None: return dataset_index
  if dataset_name is None: raise ValueError("You need to either provide `dataset_index` or `dataset_name`.")
  lookup_dataset_index = {
    "kin"  : 9,
    "mag"  : 11,
    "mach" : 13,
  }
  dataset_name = dataset_name.lower()
  if dataset_name not in lookup_dataset_index:
    _print_header(file_path, header_names)
    raise ValueError(
      f"`{dataset_name}` is an invalid dataset. "
      f"Choose from: {list_utils.cast_to_string(lookup_dataset_index.keys())}, or provide `dataset_index` directly."
    )
  return lookup_dataset_index[dataset_name]

def _extract_data(
  lines         : list[str],
  num_datasets  : int,
  dataset_index : int,
  time_norm     : float,
  raise_error   : bool,
) -> tuple[list[float], list[float]]:
  time_index = 0
  prev_time = numpy.inf
  times, values = [], []
  for line in reversed(lines):
    tokens = line.strip().split()
    if len(tokens) != num_datasets: continue
    if "#" in tokens[time_index] or "#" in tokens[dataset_index]: continue
    try:
      time_val = float(tokens[time_index]) / time_norm
      data_val = float(tokens[dataset_index])
    except ValueError: continue
    if time_val < prev_time:
      if data_val == 0.0 and time_val > 0:
        message = f"field[{dataset_index}] = 0.0 at time = {time_val:.3f}"
        if raise_error: raise ValueError(f"Error: {message}")
        print(f"Warning: {message}")
        continue
      times.append(time_val)
      values.append(data_val)
      prev_time = time_val
  return list(reversed(times)), list(reversed(values))


## END OF MODULE