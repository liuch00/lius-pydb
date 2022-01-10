import numpy as np

from FileSystem.BufManager import BufManager
from ManageSystem import macro


class IndexHandler:
    def __init__(self, buf_manager: BufManager, database_name, home_directory):
        self._manager = buf_manager
        index_file_name = database_name + macro.INDEX_NAME
        # index_file_path = home_directory + '/' + database_name + '/' + index_file_name
        index_file_path = home_directory / database_name / index_file_name
        if not self._manager.fileExist(index_file_path):
            self._manager.createFile(index_file_path)
        self._file_id = self._manager.openFile(index_file_path)
        self._is_modified = False

    def get_page(self, page_id):
        res: np.ndarray = self._manager.getPage(fileID=self._file_id, pageID=page_id)
        return res

    def put_page(self, page_id, data):
        self._is_modified = True
        self._manager.fetchPage(fileID=self._file_id, pageID=page_id, buf=data)
        return None

    def new_page(self):
        data = np.zeros(macro.PAGE_SIZE, dtype=np.uint8)
        page_id: int = self._manager.newPage(fileID=self._file_id, buf=data)
        return page_id

    def close_file(self):
        self._manager.closeFile(fileID=self._file_id)
        return None
