## ###############################################################
## DEPENDENCIES
## ###############################################################
import h5py
import time
import numpy
import argparse
from jormi.ww_io import io_manager
from jormi.ww_data import compute_stats
from jormi.ww_plots import plot_manager, add_annotations
from ww_flash_sims.sim_io import read_grid_data


## ###############################################################
## TEST FLASH FIELD REFORMATTING CORRECTNESS
## ###############################################################
def main():
  ## run with: `perf stat -e cache-misses` + `uv run `
  # hdf5_file_path = "/scratch/jh2/nk7952/Re500/Mach0.8/Pm1/144/plt/Turb_hdf5_plt_cnt_0055"
  # hdf5_file_path = "/scratch/ek9/nk7952/Re1500/Mach0.8/Pm1/288/plt/Turb_hdf5_plt_cnt_0069"
  hdf5_file_path = "/scratch/ek9/nk7952/Re1500/Mach0.8/Pm1/576/plt/Turb_hdf5_plt_cnt_0069"
  # hdf5_file_path = "/scratch/ek9/nk7952/Re1500/Mach0.8/Pm1/1152/plt/Turb_hdf5_plt_cnt_0069"
  parser = argparse.ArgumentParser(description="FLASH HDF5 reformatting cache test")
  parser.add_argument("-m", "--method", required=True, type=str, choices=["v1", "v2", "v3"], help="Choose from: reference (v1 or v2) or production (v3) reformatters.")
  parser.add_argument("-r", "--repeat", required=False, type=int, default=5, help="Number of repeats for timing (default: 5)")
  args = parser.parse_args()
  method = args.method
  num_repeats = args.repeat
  grid_props = read_grid_data.read_grid_properties(hdf5_file_path)
  num_blocks = (
    grid_props["num_blocks_x"],
    grid_props["num_blocks_y"],
    grid_props["num_blocks_z"],
  )
  num_cells_per_block = (
    grid_props["num_cells_per_block_x"],
    grid_props["num_cells_per_block_y"],
    grid_props["num_cells_per_block_z"],
  )
  with h5py.File(hdf5_file_path, "r") as hdf5_file:
    sfield_raw = numpy.array(hdf5_file["dens"])
  if method == "v1":
    label = "reference"
    reformatter = read_grid_data._reformat_flash_sfield_v1
  elif method == "v2":
    label = "reference"
    reformatter = read_grid_data._reformat_flash_sfield_v2
  elif method == "v3":
    label = "production"
    reformatter = read_grid_data._reformat_flash_sfield_v3
  elapsed_times = []
  for _ in range(num_repeats):
    start_time = time.time()
    sfield_formatted = reformatter(sfield_raw, num_blocks, num_cells_per_block)
    elapsed_time = time.time() - start_time
    elapsed_times.append(elapsed_time)
  min_time = numpy.min(elapsed_times)
  ave_time = numpy.median(elapsed_times)
  std_time = numpy.std(elapsed_times)
  max_time = numpy.max(elapsed_times)
  print(f"Reformatted using the {label} ({method}) method:")
  print(f"\t- input shape: {sfield_raw.shape}")
  print(f"\t- output shape: {sfield_formatted.shape}")
  print(f"\t- min execution time: {min_time:.3f} seconds")
  print(f"\t- ave execution time: {ave_time:.3f} +/- {std_time:.3f} seconds (after {num_repeats} repeats)")
  print(f"\t- max execution time: {max_time:.3f} seconds")
  print(" ")
  print("Output array properties:")
  print(sfield_formatted.flags)
  bin_centers, estimated_pdf = compute_stats.estimate_pdf(values=elapsed_times, num_bins=10)
  fig, ax = plot_manager.create_figure()
  ax.step(bin_centers, estimated_pdf, where="mid", color="black", marker="o", ms=5, ls="-", lw=1)
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.95,
    y_pos       = 0.95,
    label       = f"after {num_repeats} repeats",
    x_alignment = "right",
    y_alignment = "top",
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.95,
    y_pos       = 0.85,
    label       = f"input shape: {sfield_raw.shape}",
    x_alignment = "right",
    y_alignment = "top",
  )
  add_annotations.add_text(
    ax          = ax,
    x_pos       = 0.95,
    y_pos       = 0.75,
    label       = f"output shape: {sfield_formatted.shape}",
    x_alignment = "right",
    y_alignment = "top",
  )
  ax.set_xlabel(r"execution times [seconds]")
  ax.set_ylabel(r"$p$(times)")
  script_directory = io_manager.get_caller_directory()
  fig_name = f"execution_times_reformator_{method}.png"
  fig_file_path = io_manager.combine_file_path_parts([ script_directory, fig_name ])
  plot_manager.save_figure(fig, fig_file_path)

if __name__ == "__main__":
  main()


## END OF SCRIPT