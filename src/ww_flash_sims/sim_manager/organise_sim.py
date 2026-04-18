## { MODULE

##
## === DEPENDENCIES ===
##

from pathlib import Path
from jormi.ww_io import io_manager

##
## === OPERATOR CLASS ===
##


class OrganiseFilesInSimDirectory():

    def __init__(
        self,
        directory: str | Path,
        dry_run: bool = True,
        num_chk_to_keep: int = 1,
    ):
        self.sim_directory = Path(directory).absolute()
        self.plt_directory = io_manager.combine_file_path_parts([self.sim_directory, "plt"])
        self.dry_run = dry_run
        self.num_chk_to_keep = num_chk_to_keep
        io_manager.init_directory(self.plt_directory, verbose=False)

    def run(
        self,
    ):
        self._delete_proj_files()
        self._move_plt_files()
        self._reduce_chk_files()

    def _delete_proj_files(
        self,
    ):
        files = io_manager.filter_files(
            directory=self.sim_directory,
            prefix="Turb_proj_",
        )
        if not files:
            return
        for file_name in files:
            io_manager.delete_file(
                directory=self.sim_directory,
                file_name=file_name,
                dry_run=self.dry_run,
                verbose=False,
            )
        print(f"Deleted {len(files)} projection files.")

    def _move_plt_files(
        self,
    ):
        files = io_manager.filter_files(
            directory=self.sim_directory,
            prefix="Turb_hdf5_plt_cnt_",
        )
        if not files:
            return
        for file_name in files:
            io_manager.move_file(
                directory_from=self.sim_directory,
                directory_to=self.plt_directory,
                file_name=file_name,
                overwrite=True,
                dry_run=self.dry_run,
                verbose=False,
            )
        print(f"Moved {len(files)} plt files to {self.plt_directory}.")

    def _reduce_chk_files(
        self,
    ):
        files = io_manager.filter_files(
            directory=self.sim_directory,
            prefix="Turb_hdf5_chk_",
        )
        if not files:
            return
        if len(files) > self.num_chk_to_keep:
            files_to_delete = files[:-self.num_chk_to_keep]
            for file_name in files_to_delete:
                io_manager.delete_file(
                    directory=self.sim_directory,
                    file_name=file_name,
                    dry_run=self.dry_run,
                    verbose=False,
                )
            print(f"Deleted {len(files_to_delete)} old checkpoint files.")


## } MODULE
