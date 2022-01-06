class FileIndexID:
    def __init__(self, table_name, file_index_root_id):
        self._table_name = table_name
        self._file_index_root_id = file_index_root_id

    @property
    def table_name(self):
        return self._table_name

    @property
    def file_index_root_id(self):
        return self._file_index_root_id

    def __str__(self):
        return f'{{table_name: {self.table_name}, file_index_root_id: {self.file_index_root_id}}}'

    def __eq__(self, other):
        return self._file_index_root_id == other.file_index_root_id and self._table_name == other.table_name

    def __hash__(self):
        return hash((self._table_name, self._file_index_root_id))
