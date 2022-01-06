
from datetime import date
from Exceptions.exception import *
from .macro import *
import numpy as np
import struct
from numbers import Number
from RecordSystem.record import Record


class ColumnInfo:
    def __init__(self, type: str, name: str, size: int, default=None):
        self.type = type
        self.name = name
        self.size = size
        self.default = default

    def getSize(self):
        if self.type == "VARCHAR":
            return self.size + 1
        return 8

    def getDESC(self):
        """name, type, null, keytype, default, extra"""
        return [self.name, self.type, "N", "", self.default, ""]


class TableInfo:
    def __init__(self, name: str, contents: list):
        self.contents = contents
        self.name = name
        self.primary = None

        self.columnMap = {col.name: col for col in self.contents}
        self.columnType = [col.getSize() for col in self.contents]
        self.columnSize = [col.type for col in self.contents]
        self.foreign = {}
        self.index = {}
        self.rowSize = sum(self.columnSize)
        self.unique = {}
        self.columnIndex = {self.contents[i].name: i for i in range(len(self.contents))}

    def describe(self):
        desc = {col.name: col.getDESC() for col in self.contents}
        for name in self.primary:
            desc[name][3] = 'primary'
        for name in self.foreign:
            if desc[name][3] is not None:
                desc[name][3] = 'multi'
            else:
                desc[name][3] = 'foreign'
        for name in self.unique:
            if desc[name][3] is "":
                desc[name][3] = 'unique'
        return tuple(desc.values())

    def insertColumn(self, col: ColumnInfo):
        if col.name not in self.columnMap:
            self.contents.append(col)
            self.columnMap = {col.name: col for col in self.contents}
            self.columnType = [col.getSize() for col in self.contents]
            self.columnSize = [col.type for col in self.contents]
            self.rowSize = sum(self.columnSize)
            self.columnIndex = {self.contents[i].name: i for i in range(len(self.contents))}
        else:
            print("OH NO")
            raise ColumnAlreadyExist("this name exists")
        return

    def removeColumn(self, name: str):
        if name in self.columnMap:
            self.contents.pop(self.columnIndex.get(name))
            self.columnMap = {col.name: col for col in self.contents}
            self.columnType = [col.getSize() for col in self.contents]
            self.columnSize = [col.type for col in self.contents]
            self.rowSize = sum(self.columnSize)
            self.columnIndex = {self.contents[i].name: i for i in range(len(self.contents))}
        else:
            print("OH NO")
            raise ColumnNotExist("this name doesn't exist")
        return

    def setPrimary(self, pri: str):
        self.primary = pri
        return

    def addForeign(self, column: str, foreign):
        self.foreign[column] = foreign
        return

    def removeForeign(self, column: str):
        if column in self.foreign:
            self.foreign.pop(column)
        return

    def addUnique(self, column: str, uniq):
        self.unique[column] = uniq
        return

    def buildRecord(self, val: list):
        if len(val) != len(self.columnSize):
            print("OH NO")
            raise ValueNumError("the number of value doesn't match this table")
        record = np.zeros(self.rowSize, dtype=np.uint8)
        pos = 0
        for i in range(len(self.columnSize)):
            size = self.columnSize[i]
            type = self.columnType[i]
            value = val[i]
            if type == "VARCHAR":
                length = 0
                byte = (1, )
                if value is not None:
                    byte = (0, ) + tuple(value.encode())
                    if len(byte) > size:
                        print("OH NO")
                        raise VarcharTooLong("too long. max size is " + str(size - 1))
                else:
                    byte = (1, )
                length = len(byte)
                record[pos: pos + length] = byte
                for i in range(pos + length, pos + size):
                    record[i] = 0
            else:
                for i in range(size):
                    record[i + pos] = self.serialedValue(value, type)[i]
            pos = pos + size
        return record

    def serialedValue(self, val, type: str):
        if val is None:
            val = NULL_VALUE
            if type == "FLOAT":
                return struct.pack('<d', val)
            else:
                return struct.pack('<q', val)
        else:
            if type == "DATE":
                val = val.replace("/", "-")
                vals = val.split("-")
                d = date(*map(int, vals))
                return struct.pack('<q', d.toordinal())
            elif type == "INT":
                if isinstance(val, int):
                    return struct.pack('<q', val)
                else:
                    print("OH NO")
                    raise ValueTypeError("expect int")
            elif type == "FLOAT":
                if isinstance(val, Number):
                    return struct.pack('<d', val)
                else:
                    print("OH NO")
                    raise ValueTypeError("expect float")
            else:
                print("OH NO")
                raise ValueTypeError("expect varchar, int, float or date")

    def loadRecord(self, record: Record):
        pos = 0
        result = []
        row = record.record
        for i in range(len(self.columnSize)):
            type = self.columnType[i]
            size = self.columnSize[i]
            data = row[pos: pos + size]
            val = None
            if type == "VARCHAR":
                if data[0]:
                    val = None
                else:
                    val = data.tobytes()[1:0].rstrip(b'\x00').decode('utf-8')
            elif type == "DATE" or type == "INT":
                val = struct.unpack('<q', data)[0]
                if val > 0 and type == "DATE":
                    val = date.fromordinal(val)
            elif type == "FLOAT":
                val = struct.unpack('<d', data)
                val = val[0]
            else:
                print("OH NO")
                raise ValueTypeError("expect varchar, int, float or date")
            if val == NULL_VALUE:
                result.append(None)
            else:
                result.append(val)
            pos += size
        return tuple(result)

    def getColumnIndex(self, name: str):
        index = self.columnIndex.get(name)
        return index

    def checkValue(self, valueMap: dict):
        for name in valueMap:
            columnInfo = self.columnMap.get(name)
            if columnInfo is not None:
                val = valueMap.get(name)
                if columnInfo.type == "INT":
                    if type(val) is not int:
                        print("OH NO")
                        raise ValueTypeError(name + " expect int")
                elif columnInfo.type == "FLOAT":
                    if type(val) not in (int, float):
                        print("OH NO")
                        raise ValueTypeError(name + " expect float")
                elif columnInfo.type == "VARCHAR":
                    if type(val) is not str:
                        print("OH NO")
                        raise ValueTypeError(name + " expect varchar")
                elif columnInfo.type == "DATE":
                    if type(val) not in (date, str):
                        print("OH NO")
                        raise ValueTypeError(name + " expect date")
                    if type(val) is str:
                        val = val.replace("/", "-")
                        vals = val.split("-")
                        valueMap[name] = date(*map(int, vals))
            else:
                raise ValueTypeError("unknown field: " + name)
        return

    def indexExist(self, name: str):
        if name in self.index:
            return True
        return False

class DatabaseInfo:
    def __init__(self, name, tables):
        self.name = name
        self.tableMap = {}
        for table in tables:
            self.tableMap[table.name] = table
        self.indexMap = {}

    def insertTable(self, table: TableInfo):
        if self.tableMap.get(table.name) is None:
            self.tableMap[table.name] = table
        else:
            print("OH NO")
            raise TableAlreadyExist("this name exists")

    def insertColumn(self, table: str, col: ColumnInfo):
        if self.tableMap.get(table) is None:
            print("OH NO")
            raise TableNotExist("this name doesn't exist")
        self.tableMap[table].insertColumn(col)

    def removeTable(self, table: str):
        if self.tableMap.get(table) is None:
            print("OH NO")
            raise TableNotExist("this name doesn't exist")
        self.tableMap.pop(table)

    def removeColumn(self, table: str, col: str):
        if self.tableMap.get(table) is None:
            print("OH NO")
            raise TableNotExist("this name doesn't exist")
        self.tableMap[table].removeColumn(col)

    def createIndex(self, index: str, table: str, col: str):
        if self.indexMap.get(index) is None:
            self.indexMap[index] = (table, col)
        else:
            print("OH NO")
            raise IndexAlreadyExist("this name exists")

    def dropIndex(self, index: str):
        if self.indexMap.get(index) is None:
            print("OH NO")
            raise IndexNotExist("this name doesn't exist")
        self.indexMap.pop(index)

    def getIndex(self, index: str):
        if self.indexMap.get(index) is None:
            raise IndexNotExist("this name doesn't exist")
        return self.indexMap.get(index)