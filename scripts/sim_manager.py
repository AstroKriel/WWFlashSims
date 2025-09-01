## { MODULE

from ww_flash_sims.sim_manager import organise_sim
from ww_flash_sims.sim_types import ssd_sim


def main():
    for sim_directory in ssd_sim.get_all_ssd_sim_directories():
        organise_sim.OrganiseFilesInSimDirectory(sim_directory)


if __name__ == "__main__":
    main()

## } MODULE
