from index_handler import IndexHandler
from file_index import FileIndex
from ..FileSystem import FileManager


class IndexManager:
    def __init__(self, file_manager: FileManager, home_directory: str):
        self._file_manager = file_manager
        self._home_directory = home_directory

