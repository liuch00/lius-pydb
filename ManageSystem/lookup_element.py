class Term:
    """term_type:   0 is null
                    1 is compare
                    2 is in
                    3 is like
    """

    def __init__(self, term_type, table_name, col, operator=None, aim_table_name=None, aim_col=None, value=None):
        self._type: int = term_type
        self._table: str = table_name
        self._col: str = col
        self._operator: str = operator
        self._aim_table: str = aim_table_name
        self._aim_col: str = aim_col
        self._value = value

    @property
    def aim_table(self):
        return self._aim_table

    @property
    def table(self):
        return self._table

    @property
    def operator(self):
        return self._operator

    @property
    def col(self):
        return self._col

    @property
    def aim_col(self):
        return self._aim_col


class Reducer:
    """reducer_type:0 is all
                    1 is field
                    2 is aggregation
                    3 is counter
    """

    def __init__(self, reducer_type, table_name=None, col=None, aggregator=None):
        self._reducer_type: int = reducer_type
        self._table_name: str = table_name
        self._col: str = col
        self._aggregator: str = aggregator

    def to_string(self, prefix=True):
        # todo:
        pass

    def select(self, data: tuple):
        # todo:
        pass


class LookupOutput:
    # todo:modified
    def __init__(self, headers=None, data=None, message=None, change_db=None, cost=None):
        if headers and not isinstance(headers, (list, tuple)):
            headers = (headers,)
        if data and not isinstance(data[0], (list, tuple)):
            data = tuple((each,) for each in data)
        self._headers = headers
        self._data = data
        self._header_index = {h: i for i, h in enumerate(headers)} if headers else {}
        self._alias_map = {}
        self._message = message
        self._database = change_db
        self._cost = cost

    def simplify(self):
        pass


class Join:
    def __init__(self):
        pass
