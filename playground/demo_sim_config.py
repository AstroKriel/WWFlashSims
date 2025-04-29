from ww_flash_sims.sim_types import ssd_sim


def main():
  ssd_sim_obj = ssd_sim.SSDSimulation(
    directory                          = "/bla/blah/bleh/",
    num_cells_per_box_length           = 288,
    init_mach_number                   = 0.5,
    init_energy_ratio                  = 1e-10,
    box_normalised_forcing_wave_number = 2.0,
    num_turnover_times                 = 100,
    init_hydrodynamic_reynolds_number  = 1000,
    init_magnetic_reynolds_number      = 5000,
    init_magnetic_prandtl_number       = 5,
  )
  ssd_sim_obj.compute_missing_params()
  ssd_sim_obj.print_sim_params()

if __name__ == "__main__":
  main()

## end