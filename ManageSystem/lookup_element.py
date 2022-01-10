from Exceptions.exception import ValueTypeError
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

    @property
    def value(self):
        return self._value

    @property
    def type(self):
        return self._type


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

    def target(self):
        return f'{self._table_name}.{self._col}'

    def to_string(self, prefix=True):
        base = self.target()
        if self._reducer_type == 1:
            return base if prefix else self._col
        if self._reducer_type == 2:
            return f'{self._aggregator}({base})' if prefix else f'{self._aggregator}({self._col})'
        if self._reducer_type == 3:
            return f'COUNT(*)'

    def select(self, data: tuple):
        function_map = {
            'COUNT': lambda x: len(set(x)),
            'MAX': max,
            'MIN': min,
            'SUM': sum,
            'AVG': lambda x: sum(x) / len(x)
        }
        if self._reducer_type == 3:
            return len(data)
        if self._reducer_type == 1:
            return data[0]
        if self._reducer_type == 2:
            try:
                result = function_map[self._aggregator](tuple(filter(lambda x: x is not None, data)))
                return result
            except TypeError:
                raise ValueTypeError("incorrect value type for aggregation")

    @property
    def reducer_type(self):
        return self._reducer_type


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
        """Simplify headers if all headers have same prefix"""
        if not self._headers:
            return
        header: str = self._headers[0]
        if header.find('.') < 0:
            return
        prefix = header[:header.find('.') + 1]  # Prefix contains "."
        for header in self._headers:
            if len(header) <= len(prefix) or not header.startswith(prefix):
                break
        else:
            self._headers = tuple(header[len(prefix):] for header in self._headers)

    def size(self):
        size: int = len(self._data)
        return size

    @property
    def data(self):
        return self._data

    @property
    def headers(self):
        return self._headers

    def header_id(self, header) -> int:
        if header in self._alias_map:
            header = self._alias_map[header]
        if header in self._header_index:
            res = self._header_index[header]
            return res

    def insert_alias(self, alias, header):
        self._alias_map[alias] = header
        return None

    @property
    def alias_map(self):
        return self._alias_map

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self._cost = value