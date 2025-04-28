from pathlib import Path


## ###############################################################
## PROGRAM PARAMETERS
## ###############################################################
DEBUG_MODE = 1
SAFE_MODE  = 1


## ###############################################################
## HELPER FUNCTIONS
## ###############################################################
class DeleteFiles:
    def __init__(self, directory, safe_mode=SAFE_MODE, debug_mode=DEBUG_MODE):
        self.directory  = Path(directory)
        self.safe_mode  = safe_mode
        self.debug_mode = debug_mode
        self.safe_mode_buffer = 5

    def _delete_files(self, file_paths_to_delete: list[Path]):
        num_files_deleted = 0
        for file_path in file_paths_to_delete:
            try:
                if self.debug_mode:
                    print(f"delete (debug mode): {file_path}")
                else: file_path.unlink()
                num_files_deleted += 1
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")
        return num_files_deleted

    def delete_by_pattern(
            self,
            prefix         = "",
            suffix         = "",
            exclude_string = ""
        ):
        file_paths_to_delete = sorted([
            file
            for file in self.directory.glob(f"{prefix}*{suffix}")
            if exclude_string not in file.name
        ])
        if self.safe_mode and (len(file_paths_to_delete) > self.safe_mode_buffer):
            file_paths_to_delete = file_paths_to_delete[:-self.safe_mode_buffer]
        if len(file_paths_to_delete) == 0: return
        num_files_deleted = self._delete_files(file_paths_to_delete)
        print(f"\t> Deleted {num_files_deleted} files in {self.directory}")
    
    def delete_by_number(
            self,
            prefix         = "",
            suffix         = "",
            exclude_string = "",
            range_from     = None,
            range_to       = None,
            range_step     = None,
            num_to_keep    = None,
        ):
        def _get_file_number(file_path):
            return file_path.name.replace(prefix, "").replace(suffix, "")
        file_paths_to_delete = sorted([
            file
            for file in self.directory.glob(f"{prefix}*{suffix}")
            if exclude_string not in file.name
        ])
        if len(file_paths_to_delete) == 0:
            print("No matching files found.")
            return
        if self.safe_mode and (len(file_paths_to_delete) > self.safe_mode_buffer):
            file_paths_to_delete = file_paths_to_delete[:-self.safe_mode_buffer]
        file_numbers = [
            int(_get_file_number(file_path))
            for file_path in file_paths_to_delete
            if _get_file_number(file_path).isdigit()
        ]
        if len(file_numbers) == 0:
            print(f"No valid numbered files found in {self.directory}")
            return
        if range_from is None: range_from = min(file_numbers)
        if range_to is None: range_to = max(file_numbers)
        if num_to_keep is not None:
            total_files_to_delete = len(file_numbers) - num_to_keep
            if total_files_to_delete <= 0:
                print(f"No files will be deleted, as only {num_to_keep} files will be kept.")
                return
            step = total_files_to_delete // (num_to_keep - 1) if (num_to_keep > 1) else 1
            files_to_keep = [
                file_numbers[i] for i in range(0, len(file_numbers), step)
            ]
            if files_to_keep[-1] != file_numbers[-1]:
                files_to_keep.append(file_numbers[-1])
            files_to_delete = [file_number for file_number in file_numbers if file_number not in files_to_keep]
        else:
            if range_step is None: range_step = 1
            files_to_delete = [
                file_number
                for file_number in range(range_from, range_to + 1, range_step)
                if file_number in file_numbers
            ]
        files_to_delete_paths = [
            f"{prefix}{file_number}{suffix}"
            for file_number in files_to_delete
        ]
        num_files_deleted = self._delete_files(files_to_delete_paths)
        print(f"\t> Deleted {num_files_deleted} files (start={range_from}, to={range_to}, step={range_step}, keep={num_to_keep}) in {self.directory}")





