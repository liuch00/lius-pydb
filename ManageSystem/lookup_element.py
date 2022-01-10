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

        func = {
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
                result = func[self._aggregator](tuple(filter(lambda x: x is not None, data)))
                return result
            except TypeError:
                raise ValueTypeError("incorrect value type for aggregation")

    @property
    def reducer_type(self):
        return self._reducer_type


class LookupOutput:
    def __init__(self, headers=None, data=None, message=None, change_db=None, cost=None):
        if headers:
            if not isinstance(headers, (list, tuple)):
                headers = (headers,)
        if data:
            if not isinstance(data[0], (list, tuple)):
                data = tuple((each,) for each in data)
        self._headers = headers
        if headers:
            self._header_index = {h: i for i, h in enumerate(headers)}
        else:
            self._header_index = {}
        self._data = data
        self._alias_map = {}
        self._cost = cost
        self._database = change_db
        self._message = message


    def size(self):
        size: int = len(self._data)
        return size


    def simplify(self):
        if self._headers:
            header: str = self._headers[0]
            num = header.find('.')
            if num >= 0:
                prefix = header[:header.find('.') + 1]
                for header in self._headers:
                    len_h = len(header)
                    len_p = len(prefix)
                    if len_h <= len_p or not header.startswith(prefix):
                        break
                else:
                    len_p = len(prefix)
                    self._headers = tuple(header[len_p:] for header in self._headers)
            else:
                return
        else:
            return


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
