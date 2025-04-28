import sys
import argparse
from pathlib import Path
from jormi.ww_io import io_manager

def get_user_directory():
  parser = argparse.ArgumentParser(description="Clean & Organise Simulation Directory.")
  parser.add_argument(
    "-d", "--directory",
    type=str, required=False,
    help="Specific simulation path to process. If not provided, all simulations will be processed."
  )
  args = parser.parse_args()
  if args.directory:
    directory = Path(args.directory).resolve()
    is_ssd_sim(directory, raise_error=True)
    return directory
  return None

def get_all_ssd_sim_directories():
  directories = list(Path("/scratch").glob("*/nk7952/Re*/Mach*/Pm*/*"))
  return sorted(
    directory.resolve()
    for directory in directories
    if io_manager.does_directory_exist(directory) and ("anti" not in str(directory))
  )

def is_ssd_sim(directory, raise_error=False):
  all_directories = get_all_ssd_sim_directories()
  result = directory in all_directories
  if not(result) and raise_error:
    raise ValueError(f"`{directory}` is not a valid SSD simulation directory.")
  return result

def do_for_simulations(func):
  directory = get_user_directory()
  all_directories = get_all_ssd_sim_directories()
  if not directory:
    for directory in all_directories:
      func(directory)
  else: func(directory)

def _save_ssd_sim_directories():
  script_directory = io_manager.get_caller_directory()
  file_name = "ssd_sim_directories.txt"
  file_path = io_manager.combine_file_path_parts([script_directory, file_name])
  with open(file_path, "w") as file_pointer:
    for directory in get_all_ssd_sim_directories():
      print(directory)
      file_pointer.write(str(directory) + "\n")
  print("\nSaved:", file_path)

if __name__ == "__main__":
  try:
    _save_ssd_sim_directories()
  except Exception as e:
    print(f"Encountered error while writing to file: {e}")
  sys.exit(0)

## end of script
