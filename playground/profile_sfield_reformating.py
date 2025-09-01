## { SCRIPT

##
## === DEPENDENCIES ===
##

import time
import h5py
import numpy
import argparse
from functools import partial
from jormi.ww_io import io_manager
from jormi.ww_plots import plot_manager, add_annotations
from ww_flash_sims.sim_io import read_grid_data

##
## === HELPER FUNCTIONS ===
##


def print_elapsed_time_stats(
    reformat_times,
):
    print(f"\t- min: {numpy.min(reformat_times):.3f} seconds")
    print(
        f"\t- ave: {numpy.median(reformat_times):.3f} +/- {numpy.std(reformat_times):.3f} seconds"
    )
    print(f"\t- max: {numpy.max(reformat_times):.3f} seconds")


##
## === ROUTINE ===
##


class FlashReformatProfiler:
    VALID_REFORMATTERS = {
        "v1": partial(read_grid_data._reformat_flash_sfield_v1, force_use=True),
        "v2": partial(read_grid_data._reformat_flash_sfield_v2, force_use=True),
        "v3": read_grid_data._reformat_flash_sfield_v3,
    }

    def __init__(self, *, file_path, version, reformat_repeats, postprocess_repeats, make_plots):
        io_manager.does_file_exist(file_path=file_path, raise_error=True)
        self.file_path = file_path
        self.version = version
        self.reformat_repeats = reformat_repeats
        self.postprocess_repeats = postprocess_repeats
        self.make_plots = make_plots
        if self.version not in self.VALID_REFORMATTERS:
            raise ValueError(f"Invalid version {self.version}")

    def run(self):
        self._load_data()
        self.reformat_func = self.VALID_REFORMATTERS[self.version]
        reformatted_sfield = self._benchmark_reformatter()
        self._print_stats(reformatted_sfield)
        if self.postprocess_repeats > 0: self._benchmark_postprocess(reformatted_sfield)
        if self.make_plots: self._plot_slices(reformatted_sfield)

    def _load_data(self):
        grid_props = read_grid_data.read_grid_properties(self.file_path)
        self.num_blocks = (
            grid_props["num_blocks_x"],
            grid_props["num_blocks_y"],
            grid_props["num_blocks_z"],
        )
        self.num_cells_per_block = (
            grid_props["num_cells_per_block_x"],
            grid_props["num_cells_per_block_y"],
            grid_props["num_cells_per_block_z"],
        )
        with h5py.File(self.file_path, "r") as hdf5_file:
            self.sfield_raw = numpy.array(hdf5_file["dens"])

    def _benchmark_reformatter(self):
        reformat_times = []
        reformatted_sfield = None
        for _ in range(self.reformat_repeats):
            start_time = time.time()
            reformatted_sfield = self.reformat_func(
                self.sfield_raw, self.num_blocks, self.num_cells_per_block
            )
            reformat_times.append(time.time() - start_time)
        print(f"Reformat stats (after {self.reformat_repeats} repeats):")
        print_elapsed_time_stats(reformat_times)
        return reformatted_sfield

    def _print_stats(self, reformatted_sfield):
        print("Array properties:")
        print(f"\t- Input shape: {self.sfield_raw.shape}")
        print(f"\t- Output shape: {reformatted_sfield.shape}")
        print(f"\t- strides: {reformatted_sfield.strides}")
        print(f"\t- dtype: {reformatted_sfield.dtype}")
        for line in str(reformatted_sfield.flags).splitlines():
            formatted_line = line.strip().replace(" :", ":")
            print(f"\t- {formatted_line}")

    def _benchmark_postprocess(self, reformatted_sfield):
        postprocess_times = []
        for _ in range(self.postprocess_repeats):
            start = time.time()
            numpy.gradient(reformatted_sfield)
            postprocess_times.append(time.time() - start)
        print(f"Postprocess stats (after {self.postprocess_repeats} repeats):")
        print_elapsed_time_stats(postprocess_times)

    def _plot_slices(self, reformatted_sfield):
        fig, axs = plot_manager.create_figure(num_rows=3, share_x=True)
        sfield_slices = [
            reformatted_sfield[reformatted_sfield.shape[0] // 2, :, :],
            reformatted_sfield[:, reformatted_sfield.shape[1] // 2, :],
            reformatted_sfield[:, :, reformatted_sfield.shape[2] // 2],
        ]
        labels = [
            r"$(x=L/2, y, z)$",
            r"$(x, y=L/2, z)$",
            r"$(x, y, z=L/2)$",
        ]
        for ax, sfield_slice, label in zip(axs, sfield_slices, labels):
            ax.imshow(sfield_slice, cmap="viridis")
            add_annotations.add_text(
                ax=ax,
                x_pos=0.05,
                y_pos=0.95,
                label=label,
                x_alignment="left",
                y_alignment="top",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        self._save_figure(fig, f"slices_{self.version}.png")

    def _save_figure(self, fig, filename):
        script_dir = io_manager.get_caller_directory()
        output_path = io_manager.combine_file_path_parts([script_dir, filename])
        plot_manager.save_figure(fig, output_path)


##
## === ENTRY POINT ===
##

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile FLASH field reformatting performance and generate diagnostics."
    )
    parser.add_argument(
        "-version",
        required=True,
        choices=["v1", "v2", "v3"],
        help="Choose a version of the reformatter to test (v3 is the most recent)."
    )
    parser.add_argument(
        "-reformat_repeats",
        type=int,
        default=5,
        help="Number of times to run the reformatting function."
    )
    parser.add_argument(
        "-postprocess_repeats",
        type=int,
        default=0,
        help="Number of times to run postprocessing steps like gradients and FFTs."
    )
    parser.add_argument("-plot", action="store_true", help="Plot field slices.")
    args = parser.parse_args()
    # file_path = "/scratch/jh2/nk7952/Re500/Mach0.8/Pm1/144/plt/Turb_hdf5_plt_cnt_0069"
    file_path = "/scratch/ek9/nk7952/Re1500/Mach0.8/Pm1/288/plt/Turb_hdf5_plt_cnt_0069"
    # file_path = "/scratch/ek9/nk7952/Re1500/Mach0.8/Pm1/576/plt/Turb_hdf5_plt_cnt_0069"
    # file_path = "/scratch/ek9/nk7952/Re1500/Mach0.8/Pm1/1152/plt/Turb_hdf5_plt_cnt_0069"
    profiler = FlashReformatProfiler(
        file_path=file_path,
        version=args.version,
        reformat_repeats=args.reformat_repeats,
        postprocess_repeats=args.postprocess_repeats,
        make_plots=args.plot,
    )
    profiler.run()

## } SCRIPT
