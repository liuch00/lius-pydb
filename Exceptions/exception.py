
class MyException(Exception):
    pass

class FailCreateError(MyException):
    pass

class FailOpenError(MyException):
    pass

class FailReadPageError(MyException):
    pass

class RecordTooLong(MyException):
    pass

class ColumnAlreadyExist(MyException):
    pass

class ColumnNotExist(MyException):
    pass

class ValueNumError(MyException):
    pass

class VarcharTooLong(MyException):
    pass

class ValueTypeError(MyException):
    pass

class TableAlreadyExist(MyException):
    pass

class TableNotExist(MyException):
    pass

class IndexAlreadyExist(MyException):
    pass

class IndexNotExist(MyException):
    pass

class DatabaseAlreadyExist(MyException):
    pass

class DatabaseNotExist(MyException):
    pass

class NoDatabaseInUse(MyException):
    pass