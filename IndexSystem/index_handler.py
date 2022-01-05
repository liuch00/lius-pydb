import numpy as np

from ..FileSystem.BufManager import BufManager
from ..FileSystem import macro


class IndexHandler:
    def __init__(self, file_manager: BufManager, database_name, home_directory):
        self._manager = file_manager
        index_file_name = database_name + macro.INDEX_NAME
        index_file_path = home_directory / database_name / index_file_name
        # todo:
        # if not self._manager.Cre(index_file_path):
        #     self._manager.(index_file_path)
        self._file_id = self._manager.openFile(index_file_path)
        self._is_modified = False

    def get_page(self, page_id):
        res: np.ndarray = self._manager.getPage(self._file_id, page_id)
        return res

    def put_page(self, page_id, data):
        self._is_modified=True
        
