import os
import json

from uuid import uuid4
from typing import Dict, Any, List


class DatasetWriter:
    """Handle outputting dataset items."""

    single_file: bool
    """Whether to save all dataset items in a single file."""
    path: str
    """Path of the output file or directory."""
    dataset_items: List[Dict[str, Any]]
    """Collection of all the items in the current dataset."""

    def __init__(self, path: str = None, single_file: bool = False) -> None:
        """Initialize DatasetWriter."""
        if path == None and single_file:
            path = self.get_unique_filename(os.getcwd())
        elif path == None and not single_file:
            path = self.get_unique_dirname(os.getcwd())
        elif os.path.isdir(path) and single_file:
            path = self.get_unique_filename(path)
        elif os.path.isfile(path) and not single_file:
            raise ValueError(
                "Cannot write to a file with the single_file mode disabled. Try setting --single-file.")

        self.single_file = single_file
        self.path = path
        self.dataset_items = []

    def get_unique_dirname(self, base_path):
        """Get a unique dirname."""
        return os.path.join(base_path, str(uuid4()))

    def get_unique_filename(self, base_path):
        """Get a unique filename."""
        return os.path.join(base_path, f"{uuid4()}.json")

    def save_intermediate_result(self, result: Dict[str, Any]):
        """Either save an item to its own file or concatenate it with all dataset items in a single file."""
        if self.single_file:
            self.dataset_items.append(result)

            current_directory = os.path.dirname(self.path)
            if current_directory != "" and current_directory != ".":
                os.makedirs(current_directory, exist_ok=True)

            with open(self.path, "w") as output_file:
                json.dump(self.dataset_items, output_file)
        else:
            current_filepath = self.get_unique_filename(self.path)

            os.makedirs(self.path, exist_ok=True)
            with open(current_filepath, "w") as output_file:
                json.dump(result, output_file)
