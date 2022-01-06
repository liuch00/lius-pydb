class Term:
    """term_type:   0 is null
                    1 is compare
                    2 is in
                    3 is like
    """

    def __init__(self, term_type, table_name, col, operator, aim_table_name, aim_col, value):
        self._type: int = term_type
        self._table: str = table_name
        self._col: str = col
        self._operator: str = operator
        self._aim_table: str = aim_table_name
        self._aim_col: str = aim_col
        self._value = value


class Reducer:
    """reducer_type:0 is all
                    1 is field
                    2 is aggregation
                    3 is counter
    """

    def __init__(self, reducer_type, table_name, col, aggregator):
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
    def __init__(self):
        pass

    
