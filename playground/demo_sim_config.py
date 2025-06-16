from ww_flash_sims.sim_types import ssd_sim


def main():
  tmp_directory = "/home/586/nk7952/asgard/sindri/python/ww_flash_sims/playground/tmp"
  sim_1_obj = ssd_sim.SSDSimulation(
    directory                          = tmp_directory,
    num_cells_per_box_length           = 576,
    init_mach_number                   = 0.5,
    init_energy_ratio                  = 1e-10,
    box_normalised_forcing_wave_number = 2.0,
    num_turnover_times                 = 100,
    init_hydrodynamic_reynolds_number  = 1000,
    init_magnetic_prandtl_number       = 5,
  )
  sim_1_obj.save_to_json_file()
  sim_1_obj.print_sim_params()
  sim_2_obj = ssd_sim.SSDSimulation.read_from_json_file(directory=tmp_directory)
  sim_2_obj.print_sim_params()


if __name__ == "__main__":
  main()

## end of demo script