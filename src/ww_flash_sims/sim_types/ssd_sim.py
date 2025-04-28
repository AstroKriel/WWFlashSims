from pathlib import Path

class SSDSimulation():
  ALIASES = {
    "N_res"         : "num_cells_per_box_length",
    "scaled_k_turb" : "box_normalised_forcing_wave_number",
    "mach_0"        : "init_mach_number",
    "t_turb_0"      : "init_turnover_time",
    "ell_box"       : "box_length",
    "c_s"           : "sound_speed",
    "rho"           : "total_density",
    "Re_0"          : "init_hydrodynamic_reynolds_number",
    "Rm_0"          : "init_magnetic_reynolds_number",
    "Pm_0"          : "init_magnetic_prandtl_number",
    "nu"            : "viscosity",
    "eta"           : "resistivity",
    "cfl"           : "cfl_number"
  }

  def __init__(
      self,
      directory                          : str | Path,
      num_cells_per_box_length           : float,
      box_normalised_forcing_wave_number : float,
      init_energy_ratio                  : float,
      init_mach_number                   : float,
      dumps_per_turnover_time            : int = 2,
      num_turnover_times                 : float = 100.0,
      box_length                         : float = 1.0,
      sound_speed                        : float = 1.0,
      total_density                      : float = 1.0,
      init_hydrodynamic_reynolds_number  : float | None = None,
      init_magnetic_reynolds_number      : float | None = None,
      init_magnetic_prandtl_number       : float | None = None,
      cfl_number                         : float = 0.8,
      run_index                          : int = 0,
      forcing_tuned                      : bool = False,
      num_blocks                         : tuple[int, int, int] | None = None,
      num_cells_per_block                : tuple[int, int, int] | None = None,
    ):
    self.directory                          = directory
    self.num_cells_per_box_length           = num_cells_per_box_length
    self.box_normalised_forcing_wave_number = box_normalised_forcing_wave_number
    self.init_energy_ratio                  = init_energy_ratio
    self.init_kinetic_energy                = None
    self.init_magnetic_energy               = None
    self.init_mach_number                   = init_mach_number
    self.init_turnover_time                 = None
    self.dumps_per_turnover_time            = dumps_per_turnover_time
    self.num_turnover_times                 = num_turnover_times
    self.box_length                         = box_length
    self.sound_speed                        = sound_speed
    self.total_density                      = total_density
    self.init_hydrodynamic_reynolds_number  = init_hydrodynamic_reynolds_number
    self.init_magnetic_reynolds_number      = init_magnetic_reynolds_number
    self.init_magnetic_prandtl_number       = init_magnetic_prandtl_number
    self.viscosity                          = None
    self.resistivity                        = None
    self.cfl_number                         = cfl_number
    self.run_index                          = run_index
    self.forcing_tuned                      = forcing_tuned
    self.num_blocks                         = num_blocks
    self.num_cells_per_block                = num_cells_per_block

  def check_all_params_are_defined(self):
    undefined_vars = []
    for var_name, var_value in self.__dict__.items():
      if var_value is None: undefined_vars.append(var_name)
    if len(undefined_vars) > 0:
      raise ValueError(f"The following parameters have not been defined: {undefined_vars}")

  @classmethod
  def _alias(cls, alias_name, var_name):
    setattr(cls, alias_name, property(
      lambda self: getattr(self, var_name),
      lambda self, value: setattr(self, var_name, value)
    ))

  @classmethod
  def _create_aliases(cls):
    for alias_name, var_name in cls.ALIASES.items():
      cls._alias(alias_name, var_name)

  def _compute_missing_init_conditions(self):
    u_turb_0 = self.mach_0 / self.c_s
    ell_turb = self.ell_box / self.scaled_k_turb
    self.init_kinetic_energy = 0.5 * self.rho * u_turb_0 * u_turb_0
    self.init_magnetic_energy = self.init_kinetic_energy * self.init_energy_ratio
    self.init_turnover_time = ell_turb / u_turb_0

  def _compute_missing_plasma_numbers(self):
    u_turb_0 = self.mach_0 / self.c_s
    ## recall that k = 2 pi / ell, and so scaled_k_turb = k_turb ell_box / (2 pi) = ell_box / ell_turb
    ell_turb = self.ell_box / self.scaled_k_turb
    if (self.Re_0 is not None) and (self.Pm_0 is not None):
      self.Re_0 = float(self.Re_0)
      self.Pm_0 = float(self.Pm_0)
      self.Rm_0 = self.Re_0 * self.Pm_0
      self.nu   = u_turb_0 / (ell_turb * self.Re_0)
      self.eta  = self.nu / self.Pm_0
    elif (self.Rm_0 is not None) and (self.Pm_0 is not None):
      self.Rm_0 = float(self.Rm_0)
      self.Pm_0 = float(self.Pm_0)
      self.Re_0 = self.Rm_0 / self.Pm_0
      self.eta  = u_turb_0 / (ell_turb * self.Rm_0)
      self.nu   = self.eta * self.Pm_0
    else: raise Exception(f"Insufficient plasma numbers provided: Re_0 = {self.Re_0:.2f}, Rm_0 = {self.Rm_0:.2f}, and Pm_0 = {self.Pm_0:.2f}.")

  def _compute_grid_properties(self):
    if   self.N_res in [ 576, 1152 ]: self.num_blocks = (96, 96, 72)
    elif self.N_res in [ 144, 288 ]:  self.num_blocks = (36, 36, 48)
    elif self.N_res in [ 36, 72 ]:    self.num_blocks = (12, 12, 18)
    elif self.N_res in [ 18 ]:        self.num_blocks = (6, 6, 6)
    else: raise ValueError
    self.num_cells_per_block = tuple(
      int(self.N_res / num_blocks_in_dir)
      for num_blocks_in_dir in self.num_blocks
    )

SSDSimulation._create_aliases()
