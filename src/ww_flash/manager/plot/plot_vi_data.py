## ###############################################################
## DEPENDANCIES
## ###############################################################
import sys
import numpy
from jormi.ww_io import argparse, io_manager, flash_data
from jormi.ww_plots import plot_manager


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
def compute_log10(data):
  log10_data = numpy.full_like(data, numpy.nan, dtype=float)
  valid_mask = data > 0
  log10_data[valid_mask] = numpy.log10(data[valid_mask])
  return log10_data


## ###############################################################
## OPERATOR CLASS
## ###############################################################
class PlotVIData():
  def __init__(self, directory):
    self.directory = directory

  def run(self):
    fig, axs = plot_manager.create_figure(num_rows=3, share_x=True)
    self._load_data()
    self._plot_data(axs)
    self._label_plot(axs)
    script_directory = io_manager.get_caller_directory()
    fig_name = "demo_vi_plot.png"
    fig_file_path = io_manager.combine_file_path_parts([ script_directory, fig_name ])
    plot_manager.save_figure(fig, fig_file_path)

  def _load_data(self):
    flash_data.read_vi_data(directory=self.directory, print_header=True)
    self.time, self.mach_number = flash_data.read_vi_data(directory=self.directory, dataset_name="mach")
    _, self.kinetic_energy  = flash_data.read_vi_data(directory=self.directory, dataset_name="kin")
    _, self.magnetic_energy = flash_data.read_vi_data(directory=self.directory, dataset_name="mag")

  def _plot_data(self, axs):
    plot_params = dict(ls="-", lw=1.5, zorder=3)
    log10_kinetic_energy  = compute_log10(self.kinetic_energy)
    log10_magnetic_energy = compute_log10(self.magnetic_energy)
    axs[0].plot(
      self.time,
      self.mach_number,
      color="black", **plot_params
    )
    axs[1].plot(
      self.time,
      log10_kinetic_energy,
      color="blue", **plot_params, label=r"$E_{\rm kin} = (1/2) \int_\mathcal{V} \rho u^2 {\rm d}V$"
    )
    axs[1].plot(
      self.time,
      log10_magnetic_energy,
      color="red", **plot_params, label=r"$E_{\rm mag} = (1/8\pi) \int_\mathcal{V} b^2 {\rm d}V$"
    )
    axs[2].plot(
      self.time,
      numpy.gradient(log10_magnetic_energy, self.time),
      color="red", **plot_params
    )
  
  def _label_plot(self, axs):
    axs[2].set_xlabel(r"$t$")
    axs[0].set_ylabel(r"$\langle u^2 \rangle^{1/2}_\mathcal{V} / c_s$")
    axs[1].set_ylabel(r"$\log_{10}({\rm Energy})$")
    axs[2].set_ylabel(r"${\rm d} \log_{10}(E_{\rm mag}) / {\rm d} t$")
    axs[1].legend(loc="lower right", fontsize=18)
    axs[2].axhline(y=0.0, color="black", ls="--", lw=1.0, zorder=5)


## ###############################################################
## MAIN PROGRAM
## ###############################################################
def main():
  directory = "/scratch/jh2/nk7952/Re1500/Mach0.5/Pm1/576"
  routine = PlotVIData(directory)
  routine.run()


## ###############################################################
## SCRIPT ENTRY POINT
## ###############################################################
if __name__ == "__main__":
  main()
  sys.exit(0)


## END OF SCRIPT