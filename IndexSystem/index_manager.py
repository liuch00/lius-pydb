from .index_handler import IndexHandler
from .file_index import FileIndex
from typing import Dict
from FileSystem.BufManager import BufManager
from .FileIndexID import FileIndexID


class IndexManager:
    def __init__(self, buf_manager: BufManager, home_directory: str = '/'):
        self._buf_manager = buf_manager
        self._home_directory = home_directory
        self._started_index_handler: Dict[str, IndexHandler] = {}
        self._started_file_index: Dict[FileIndexID, FileIndex] = {}

    def catch_handler(self, database_name):
        if database_name in self._started_index_handler:
            return self._started_index_handler[database_name]
        else:
            # not exist
            new_handler: IndexHandler = IndexHandler(buf_manager=self._buf_manager, database_name=database_name,
                                                     home_directory=self._home_directory)
            self._started_index_handler[database_name]: IndexHandler = new_handler
            return self._started_index_handler[database_name]

    def shut_handler(self, database_name):
        if database_name in self._started_index_handler:
            index_handler = self._started_index_handler.pop(database_name)
            for key, file_index in tuple(self._started_file_index.items()):
                if file_index.handler is not index_handler:
                    continue
                if (key._table_name, key._file_index_root_id) not in self._started_index_handler:
                    return None
                tmp_file_index = self._started_index_handler.pop((key._table_name, key._file_index_root_id))
                if tmp_file_index.is_modified:
                    tmp_file_index.pour()
            return True
            # for ID in self._started_file_index:
            #     file_index = self._started_file_index.get(ID)
            #     if file_index.handler is index_handler:
            #         # shut index
            #         if ID in self._started_file_index:
            #             tmp_file_index = self._started_file_index.pop(ID)
            #             if tmp_file_index.is_modified:
            #                 tmp_file_index.pour()
        else:
            return False

    def create_index(self, database_name, table_name):
        handler = self.catch_handler(database_name=database_name)
        root_id = handler.new_page()
        ID = FileIndexID(table_name=table_name, file_index_root_id=root_id)
        self._started_file_index[ID] = FileIndex(index_handler=handler, root_id=root_id)
        self._started_file_index[ID].pour()
        return self._started_file_index[ID]

    def start_index(self, database_name, table_name, root_id):
        ID = FileIndexID(table_name=table_name, file_index_root_id=root_id)
        if ID in self._started_file_index:
            file_index = self._started_file_index.get(ID)
            return file_index
        else:
            handler = self.catch_handler(database_name=database_name)
            file_index = FileIndex(index_handler=handler, root_id=root_id)
            # load data
            file_index.take()
            self._started_file_index[ID] = file_index
            return file_index

    def close_manager(self):
        for db_name in self._started_index_handler:
            self.shut_handler(database_name=db_name)
        return None
