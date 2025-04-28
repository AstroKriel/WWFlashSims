## ###############################################################
## DEPENDENCIES
## ###############################################################
import sys
import time
import h5py
import numpy
from jormi.ww_io import flash_data, io_manager
from jormi.ww_plots import plot_manager, add_annotations


## ###############################################################
## REFERENCE REFORMAT FUNCTION
## ###############################################################
def reformat_flash_sfield(
    sfield              : numpy.ndarray,
    num_blocks          : tuple[int, int, int],
    num_cells_per_block : tuple[int, int, int],
  ):
  ## initialise dataset with Fortran-style coordinates ordering: [z, y, x]
  sfield_sorted = numpy.zeros(
    shape = (
      num_cells_per_block[2] * num_blocks[2],
      num_cells_per_block[1] * num_blocks[1],
      num_cells_per_block[0] * num_blocks[0],
    ),
    dtype = numpy.float32,
    order = "C" # use C memory layout (row-major) for consistency with python operation
  )
  ## place each block from the Fortran (column-major) output structure in its corresponding portion of the domain
  block_index = 0
  for index_block_z in range(num_blocks[2]):
    for index_block_y in range(num_blocks[1]):
      for index_block_x in range(num_blocks[0]):
        sfield_sorted[
          index_block_z * num_cells_per_block[2] : (index_block_z+1) * num_cells_per_block[2],
          index_block_y * num_cells_per_block[1] : (index_block_y+1) * num_cells_per_block[1],
          index_block_x * num_cells_per_block[0] : (index_block_x+1) * num_cells_per_block[0],
        ] = sfield[block_index, :, :, :]
        block_index += 1
  ## reorder components from [z, y, x] to [x, y, z]
  return numpy.transpose(sfield_sorted, (2, 1, 0))


## ###############################################################
## FLASH FIELD REFORMATTING CORRECTNESS TEST
## ###############################################################
class TestFlashReformat:
  def __init__(self, file_path, num_repeats=10):
      io_manager.does_file_exist(file_path=file_path, raise_error=True)
      self.file_path   = file_path
      self.num_repeats = num_repeats

  def run(self):
    fig, axs = plot_manager.create_figure(num_rows=3, num_cols=2, axis_shape=(5, 5))
    self.grid_properties = flash_data.read_grid_properties(file_path)
    self.num_blocks = (
        self.grid_properties["num_blocks_x"],
        self.grid_properties["num_blocks_y"],
        self.grid_properties["num_blocks_z"],
    )
    self.num_cells_per_block = (
        self.grid_properties["num_cells_per_block_x"],
        self.grid_properties["num_cells_per_block_y"],
        self.grid_properties["num_cells_per_block_z"],
    )
    self.sfield_raw = self._load_field_data()
    print(f"Input has shape: {self.sfield_raw.shape}")
    print(self.num_blocks)
    print(self.num_cells_per_block)
    print(f"Comparing execution times (after {self.num_repeats} repetitions)...")
    self.sfield_reformated_v1, avg_time_v1 = self._benchmark_and_plot("reference", axs[:,0], reformat_flash_sfield)
    self.sfield_reformated_v2, avg_time_v2 = self._benchmark_and_plot("production", axs[:,1], flash_data._read_grid_data._reformat_flash_sfield)
    speedup_percent = avg_time_v1 / avg_time_v2
    improvement_factor = 100 * (avg_time_v1 - avg_time_v2) / avg_time_v1
    print(f"Production version is {improvement_factor:.2f}% faster ({speedup_percent:.2f}x speedup).")
    print(f"Output has shape: {self.sfield_reformated_v2.shape}")
    self._compare_outputs()
    self._adjust_figure(axs)
    script_directory = io_manager.get_caller_directory()
    fig_file_name = "reformatted_flash_sfield_slices.png"
    fig_file_path = io_manager.combine_file_path_parts([ script_directory, fig_file_name ])
    plot_manager.save_figure(fig, fig_file_path)

  def _load_field_data(self):
    with h5py.File(self.file_path, "r") as h5file:
      sfield_raw = numpy.array(h5file["dens"])
    return numpy.log10(sfield_raw)

  def _benchmark_and_plot(self, label, axs, func):
    avg_time, std_time = self._get_average_execution_time(func)
    sfield_formatted = func(self.sfield_raw, self.num_blocks, self.num_cells_per_block)
    self._plot_slices(label, axs, sfield_formatted)
    print(f"average {label}.{func.__name__}() execution time: {avg_time:.6f} +/- {std_time:.6f} seconds")
    return sfield_formatted, avg_time

  def _get_average_execution_time(self, func):
    times = []
    for _ in range(self.num_repeats):
      start = time.time()
      func(self.sfield_raw, self.num_blocks, self.num_cells_per_block)
      times.append(time.time() - start)
    return numpy.median(times), numpy.std(times)

  def _plot_slices(self, label, axs, sfield_formatted):
    slice_index_x = sfield_formatted.shape[0]//2
    slice_index_y = sfield_formatted.shape[1]//2
    slice_index_z = sfield_formatted.shape[2]//2
    axs[0].imshow(sfield_formatted[slice_index_x, :, :], cmap="viridis")
    axs[1].imshow(sfield_formatted[:, slice_index_y, :], cmap="viridis")
    axs[2].imshow(sfield_formatted[:, :, slice_index_z], cmap="viridis")
    add_annotations.add_text(ax=axs[0], x_pos=0.05, y_pos=0.95, label="(x=L/2, y, z) slice")
    add_annotations.add_text(ax=axs[1], x_pos=0.05, y_pos=0.95, label="(x, y=L/2, z) slice")
    add_annotations.add_text(ax=axs[2], x_pos=0.05, y_pos=0.95, label="(x, y, z=L/2) slice")
    add_annotations.add_text(ax=axs[0], x_pos=0.05, y_pos=0.05, label=label, y_alignment="bottom")
    add_annotations.add_text(ax=axs[1], x_pos=0.05, y_pos=0.05, label=label, y_alignment="bottom")
    add_annotations.add_text(ax=axs[2], x_pos=0.05, y_pos=0.05, label=label, y_alignment="bottom")

  def _compare_outputs(self):
    print("\nComparing outputs...")
    if numpy.allclose(self.sfield_reformated_v1, self.sfield_reformated_v2):
      print("Test passed: Both reformated fields are identical.")
    else: print("Error: Something went wrong. The two reformated fields look different!")

  def _adjust_figure(self, axs):
    for row in axs:
      for ax in row:
        ax.set_xticklabels([])
        ax.set_yticklabels([])


## ###############################################################
## TEST ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  file_path = "/scratch/jh2/nk7952/Re500/Mach0.8/Pm1/144/plt/Turb_hdf5_plt_cnt_0055"
  # file_path = "/scratch/ek9/nk7952/Re1500/Mach0.8/Pm1/1152/plt/Turb_hdf5_plt_cnt_0069"
  # file_path = "/scratch/ek9/jh7060/flashv2/NIF_data_analysis/simdata/high_res/NIF_hdf5_plt_cnt_2287"
  test = TestFlashReformat(file_path)
  test.run()
  sys.exit(0)


## END OF TEST