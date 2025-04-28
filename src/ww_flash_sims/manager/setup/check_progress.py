from sim_utils.ssd_sims import do_for_simulations

from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    bool_debug_mode: bool = False
    bool_safe_mode: bool = True
    bool_ignore_jobs: bool = True
    bool_remove_clutter: bool = True
    bool_organise_plt_files: bool = True
    bool_organise_spect_files: bool = True
    bool_reduce_number_files: bool = True
    bool_aggressive: bool = False
    bool_process_plt_files: bool = False


def main(directory):
  print(directory)


if __name__ == "__main__":
  do_for_simulations(main)

## end