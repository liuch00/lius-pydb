

class FailCreateError(Exception):
    pass

class FailOpenError(Exception):
    pass

class FailReadPageError(Exception):
    pass

class RecordTooLong(Exception):
    pass

class ColumnAlreadyExist(Exception):
    pass

class ColumnNotExist(Exception):
    pass

class ValueNumError(Exception):
    pass

class VarcharTooLong(Exception):
    pass

class ValueTypeError(Exception):
    pass

class TableAlreadyExist(Exception):
    pass

class TableNotExist(Exception):
    pass

class IndexAlreadyExist(Exception):
    pass

class IndexNotExist(Exception):
    pass
