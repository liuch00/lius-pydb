from pathlib import Path
from .join import Join
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.Errors import ParseCancellationException
from antlr4.error.ErrorListener import ErrorListener
from copy import deepcopy
from datetime import date
import re
from typing import Tuple
# from .system_visitor import SystemVisitor
from FileSystem.BufManager import BufManager
from RecordSystem.RecordManager import RecordManager
from RecordSystem.FileScan import FileScan
from RecordSystem.FileHandler import FileHandler
from RecordSystem.record import Record
from RecordSystem.rid import RID
from IndexSystem.index_manager import IndexManager
from MetaSystem.MetaHandler import MetaHandler
from Exceptions.exception import *
from .lookup_element import LookupOutput, Term, Reducer
from SQL_Parser.SQLLexer import SQLLexer
from SQL_Parser.SQLParser import SQLParser
from MetaSystem.info import TableInfo, ColumnInfo
from .macro import *
from .printers import TablePrinter


class SystemManger:
    def __init__(self, visitor, syspath: Path, bm: BufManager, rm: RecordManager, im: IndexManager):
        self.visitor = visitor
        self.systemPath = syspath
        self.BM = bm
        self.IM = im
        self.RM = rm
        self.metaHandlers = {}
        self.databaselist = []
        for item in syspath.iterdir():
            self.databaselist.append(item.name)
        self.inUse = None
        self.visitor.system_manager = self

    def shutdown(self):
        self.IM.close_manager()
        self.RM.shutdown()
        self.BM.shutdown()

    def checkInUse(self):
        if self.inUse is None:
            print("OH NO")
            raise NoDatabaseInUse("use a database first")
        return

    def createDatabase(self, dbname: str):
        if dbname not in self.databaselist:
            path: Path = self.systemPath / dbname
            path.mkdir(parents=True)
            self.databaselist.append(dbname)
        else:
            print("OH NO")
            raise DatabaseAlreadyExist("this name exists")
        return

    def removeDatabase(self, dbname: str):
        if dbname in self.databaselist:
            self.IM.shut_handler(dbname)
            if self.metaHandlers.get(dbname) is not None:
                self.metaHandlers.pop(dbname).shutdown()
            path: Path = self.systemPath / dbname
            for table in path.iterdir():
                if path.name.endswith(".table"):
                    self.RM.closeFile(str(table))
                table.unlink()
            self.databaselist.remove(dbname)
            path.rmdir()
            if self.inUse == dbname:
                self.inUse = None
                result = LookupOutput(change_db='None')
                return result
        else:
            print("OH NO")
            raise DatabaseNotExist("this name doesn't exist")

    def useDatabase(self, dbname: str):
        if dbname in self.databaselist:
            self.inUse = dbname
            result = LookupOutput(change_db=dbname)
            return result
        print("OH NO")
        raise DatabaseNotExist("this name doesn't exist")

    def getTablePath(self, table: str):
        self.checkInUse()
        tablePath = self.systemPath / self.inUse / table
        return str(tablePath) + ".table"

    def execute(self, sql):
        class StringErrorListener(ErrorListener):
            def syntaxError(self, recognizer, offending_symbol, line, column, msg, e):
                raise ParseCancellationException("line " + str(line) + ":" + str(column) + " " + msg)

        self.visitor.spend_time()
        input_stream = InputStream(sql)
        lexer = SQLLexer(input_stream)
        lexer.removeErrorListeners()
        lexer.addErrorListener(StringErrorListener())
        tokens = CommonTokenStream(lexer)
        parser = SQLParser(tokens)
        parser.removeErrorListeners()
        parser.addErrorListener(StringErrorListener())
        try:
            tree = parser.program()
        except ParseCancellationException as e:
            ret = LookupOutput(None, None, str(e), cost=self.visitor.spend_time())
            return [ret]
        try:
            ret = self.visitor.visit(tree)
            return ret
        except MyException as e:
            ret = LookupOutput(message=str(e), cost=self.visitor.spend_time())
            return [ret]

    def displayTableNames(self):
        result = []
        self.checkInUse()
        usingDB = self.systemPath / self.inUse
        for file in usingDB.iterdir():
            if file.name.endswith(".table"):
                result.append(file.stem)
        return result

    def fetchMetaHandler(self):
        if self.metaHandlers.get(self.inUse) is None:
            self.metaHandlers[self.inUse] = MetaHandler(self.inUse, str(self.systemPath))
        return self.metaHandlers[self.inUse]

    def createTable(self, table: TableInfo):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        metaHandler.insertTable(table)
        tablePath = self.getTablePath(table.name)
        self.RM.createFile(tablePath, table.rowSize)
        return

    def removeTable(self, table: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        for col in tableInfo.columnMap:
            self.checkRemoveColumn(table, col)
        metaHandler.removeTable(table)
        tablePath = self.getTablePath(table)
        self.RM.destroyFile(tablePath)
        return

    def collectTableinfo(self, table: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        return metaHandler, metaHandler.collectTableInfo(table)

    def descTable(self, table: str):
        head = ('Field', 'Type', 'Null', 'Key', 'Default', 'Extra')
        data = self.collectTableinfo(table)[1].describe()
        return LookupOutput(head, data)

    def renameTable(self, src: str, dst: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        metaHandler.renameTable(src, dst)
        srcFilename = self.getTablePath(src)
        dstFilename = self.getTablePath(dst)
        self.RM.renameFile(srcFilename, dstFilename)
        return

    def createIndex(self, index: str, table: str, col: str):
        metaHandler, tableInfo = self.collectTableinfo(table)
        if index in metaHandler.databaseInfo.indexMap:
            print("OH NO")
            raise IndexAlreadyExist("this name exists")
        if col not in tableInfo.index:
            indexFile = self.IM.create_index(self.inUse, table)
            tableInfo.index[col] = indexFile.root
        else:
            metaHandler.createIndex(index, table, col)
            return
        if tableInfo.getColumnIndex(col) is not None:
            colIndex = tableInfo.getColumnIndex(col)
            for record in FileScan(self.RM.openFile(self.getTablePath(table))):
                recordData = tableInfo.loadRecord(record)
                indexFile.insert(recordData[colIndex], record.rid)
            metaHandler.createIndex(index, table, col)
        else:
            print("OH NO")
            raise ColumnNotExist(col + "doesn't exist")

    def removeIndex(self, index: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        table, col = metaHandler.databaseInfo.getIndex(index)
        metaHandler.collectTableInfo(table).index.pop(col)
        metaHandler.removeIndex(index)
        self.metaHandlers.pop(self.inUse).shutdown()
        return

    def addUnique(self, table: str, col: str, uniq: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        metaHandler.addUnique(table, col, uniq)
        if uniq not in metaHandler.databaseInfo.indexMap:
            self.createIndex(uniq, table, col)
        return

    def addForeign(self, table: str, col: str, foreign, forName=None):
        metaHandler, tableInfo = self.collectTableinfo(table)
        if (table, col) not in metaHandler.databaseInfo.indexMap.values():
            raise AddForeignError("create index on this column first")
        if forName:
            if forName not in metaHandler.databaseInfo.indexMap:
                self.createIndex(forName, foreign[0], foreign[1])
        else:
            indexName = foreign[0] + "." + foreign[1]
            if indexName not in metaHandler.databaseInfo.indexMap:
                self.createIndex(indexName, foreign[0], foreign[1])
        tableInfo.addForeign(col, foreign)
        metaHandler.shutdown()
        return

    def removeForeign(self, table, col, forName=None):
        metaHandler, tableInfo = self.collectTableinfo(table)
        if tableInfo.foreign.get(col) is not None:
            foreign = tableInfo.foreign[col][0] + "." + tableInfo.foreign[col][1]
            reftable: TableInfo = metaHandler.collectTableInfo(tableInfo.foreign[col][0])
            if reftable.primary.count(tableInfo.foreign[col][1]) != 0:
                self.removeIndex(foreign)
            tableInfo.removeForeign(col)
            metaHandler.shutdown()
        return None

    def removePrimary(self, table: str):
        # todo: check foreign
        metaHandler, tableInfo = self.collectTableinfo(table)
        if tableInfo.primary:
            for column in tableInfo.primary:
                indexName = table + "." + column
                if indexName in metaHandler.databaseInfo.indexMap:
                    self.removeIndex(indexName)
            metaHandler.removePrimary(table)
        return

    def setPrimary(self, table: str, pri):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        metaHandler.setPrimary(table, pri)
        if pri:
            for column in pri:
                indexName = table + "." + column
                if indexName not in metaHandler.databaseInfo.indexMap:
                    self.createIndex(indexName, table, column)
        return

    def addColumn(self, table: str, column, pri: bool, foreign: bool):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if pri:
            for co in column:
                if tableInfo.getColumnIndex(co) is None:
                    print("OH NO")
                    raise ColumnNotExist(co + " doesn't exist")
            self.setPrimary(table, column)
        elif foreign:
            co = column[0]
            if tableInfo.getColumnIndex(co) is None:
                print("OH NO")
                raise ColumnNotExist(co + " doesn't exist")
            self.addForeign(table, co, (column[1], column[2]), None)
        else:
            if not isinstance(column, ColumnInfo):
                raise AddError("unsupported add")
            col = column
            if tableInfo.getColumnIndex(col.name):
                print("OH NO")
                raise ColumnNotExist(col.name + " doesn't exist")
            oldTableInfo: TableInfo = deepcopy(tableInfo)
            metaHandler.databaseInfo.insertColumn(table, col)
            metaHandler.shutdown()
            copyTableFile = self.getTablePath(table + ".copy")
            self.RM.createFile(copyTableFile, tableInfo.rowSize)
            newRecordHandle: FileHandler = self.RM.openFile(copyTableFile)
            scan = FileScan(self.RM.openFile(self.getTablePath(table)))
            for record in scan:
                recordVals = oldTableInfo.loadRecord(record)
                valList = list(recordVals)
                valList.append(col.default)
                newRecordHandle.insertRecord(tableInfo.buildRecord(valList))
            self.RM.closeFile(self.getTablePath(table))
            self.RM.closeFile(copyTableFile)
            self.RM.replaceFile(copyTableFile, self.getTablePath(table))
        return

    def removeColumn(self, table: str, col: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        self.checkRemoveColumn(table, col)
        tableInfo = metaHandler.collectTableInfo(table)
        if col not in tableInfo.columnIndex:
            print("OH NO")
            raise ColumnNotExist(col + " doesn't exist")
        oldTableInfo: TableInfo = deepcopy(tableInfo)
        colIndex = tableInfo.getColumnIndex(col)
        metaHandler.removeColumn(table, col)
        copyTableFile = self.getTablePath(table + ".copy")
        self.RM.createFile(copyTableFile, tableInfo.rowSize)
        newRecordHandle: FileHandler = self.RM.openFile(copyTableFile)
        scan = FileScan(self.RM.openFile(self.getTablePath(table)))
        for record in scan:
            recordVals = oldTableInfo.loadRecord(record)
            valList = list(recordVals)
            valList.pop(colIndex)
            newRecordHandle.insertRecord(tableInfo.buildRecord(valList))
        self.RM.closeFile(self.getTablePath(table))
        self.RM.closeFile(copyTableFile)
        self.RM.replaceFile(copyTableFile, self.getTablePath(table))
        return

    def insertRecord(self, table: str, val: list):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)

        info = tableInfo.buildRecord(val)
        tempRecord = Record(RID(0, 0), info)
        valTuple = tableInfo.loadRecord(tempRecord)

        self.checkInsertConstraint(table, valTuple)
        rid = self.RM.openFile(self.getTablePath(table)).insertRecord(info)
        self.handleInsertIndex(table, valTuple, rid)
        return

    def deleteRecords(self, table: str, limits: tuple):
        self.checkInUse()
        fileHandler = self.RM.openFile(self.getTablePath(table))
        metaHandler = self.fetchMetaHandler()
        records, data = self.searchRecordIndex(table, limits)
        for record, valTuple in zip(records, data):
            self.checkRemoveConstraint(table, valTuple)
            fileHandler.deleteRecord(record.rid)
            self.handleRemoveIndex(table, valTuple, record.rid)
        res = LookupOutput('deleted_items', (len(records),))
        return res

    def indexFilter(self, table: str, limits: tuple) -> set:
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        condIndex = {}

        def build(limit: Term):
            if limit.type != 1:
                return None
            if limit.table and limit.table != table:
                return None
            limit_col = limit.col
            colIndex = tableInfo.getColumnIndex(limit_col)
            if colIndex is not None and limit.value is not None and limit.col in tableInfo.index:
                lo, hi = condIndex.get(limit.col, (-1 << 31 + 1, 1 << 31))
                tmp = limit.value
                val = int(tmp)
                if limit.operator == "=":
                    lower = max(lo, val)
                    upper = min(hi, val)
                elif limit.operator == "<":
                    lower = lo
                    upper = min(hi, val - 1)
                elif limit.operator == ">":
                    lower = max(lo, val + 1)
                    upper = hi
                elif limit.operator == "<=":
                    lower = lo
                    upper = min(hi, val)
                elif limit.operator == ">=":
                    lower = max(lo, val)
                    upper = hi
                else:
                    return None
                condIndex[limit.col] = lower, upper

        results = None
        t = tuple(map(build, limits))

        for col in condIndex:
            if results:
                lo, hi = condIndex.get(col)
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                results = results & set(index.range(lo, hi))
            else:
                lo, hi = condIndex.get(col)
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                results = set(index.range(lo, hi))
        return results

    def updateRecords(self, table: str, limits: tuple, valmap: dict):
        self.checkInUse()
        fileHandler = self.RM.openFile(self.getTablePath(table))
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        tableInfo.checkValue(valmap)
        records, data = self.searchRecordIndex(table, limits)
        for record, oldVal in zip(records, data):
            new = list(oldVal)
            for col in valmap:
                new[tableInfo.getColumnIndex(col)] = valmap.get(col)
            self.checkRemoveConstraint(table, oldVal)
            rid = record.rid
            self.checkInsertConstraint(table, new, rid)
            self.handleRemoveIndex(table, oldVal, rid)
            record.record = tableInfo.buildRecord(new)
            fileHandler.updateRecord(record)
            self.handleInsertIndex(table, tuple(new), rid)
        return LookupOutput('updated_items', (len(records),))

    def searchRecordIndex(self, table: str, limits: tuple):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        functions = self.buildConditionsFuncs(table, limits, metaHandler)
        fileHandler: FileHandler = self.RM.openFile(self.getTablePath(table))
        records = []
        data = []
        if self.indexFilter(table, limits):
            iterator = map(fileHandler.getRecord, self.indexFilter(table, limits))
            for record in iterator:
                valTuple = tableInfo.loadRecord(record)
                old_valTuple = valTuple
                if all(map(lambda fun: fun(valTuple), functions)):
                    records.append(record)
                    old_valTuple = valTuple
                    data.append(valTuple)
        else:
            for record in FileScan(fileHandler):
                valTuple = tableInfo.loadRecord(record)
                old_valTuple = valTuple
                if all(map(lambda fun: fun(valTuple), functions)):
                    records.append(record)
                    old_valTuple = valTuple
                    data.append(valTuple)
        return records, data

    def checkAnyUnique(self, table: str, pairs, thisRID: RID = None):
        conds = []
        for col in pairs:
            conds.append(Term(1, table, col, '=', value=pairs.get(col)))
        records, data = self.searchRecordIndex(table, tuple(conds))
        if len(records) <= 1:
            if records and records[0].rid == thisRID:
                return False
            elif records:
                return (tuple(pairs.keys()), tuple(pairs.values()))
            return False
        print("OH NO")
        raise CheckAnyUniqueError("get " + str(len(records)) + " same")

    def condJoin(self, res_map: dict, term):
        if self.inUse is None:
            raise ValueError("No using database!!!")
        else:
            join = Join(res_map=res_map, term=term)
            result: LookupOutput = join.get_output()
            return result

    def checkPrimary(self, table: str, colVals, thisRID: RID = None):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if tableInfo.primary:
            pairs = {}
            for col in tableInfo.primary:
                pairs[col] = colVals[tableInfo.getColumnIndex(col)]
            return self.checkAnyUnique(table, pairs, thisRID)
        return False

    def checkUnique(self, table: str, colVals, thisRID: RID = None):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if tableInfo.unique:
            for col in tableInfo.unique:
                pairs = {col: colVals[tableInfo.getColumnIndex(col)]}
                if self.checkAnyUnique(table, pairs, thisRID):
                    return self.checkAnyUnique(table, pairs, thisRID)
        return False

    def checkForeign(self, table: str, colVals):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if len(tableInfo.foreign) > 0:
            for col in tableInfo.foreign:
                conds = []
                fortable = tableInfo.foreign[col][0]
                forcol = tableInfo.foreign[col][1]
                conds.append(Term(1, fortable, forcol, '=', value=colVals[tableInfo.getColumnIndex(col)]))
                records, data = self.searchRecordIndex(fortable, tuple(conds))
                if len(records) == 0:
                    return tableInfo.name, colVals[tableInfo.getColumnIndex(col)]
                # colVal = colVals[tableInfo.getColumnIndex(col)]
                # foreignTableInfo: TableInfo = metaHandler.collectTableInfo(tableInfo.foreign[col][0])
                # index = self.IM.start_index(self.inUse, tableInfo.foreign[col][0],
                #                             foreignTableInfo.index[tableInfo.foreign[col][1]])
                # if len(set(index.range(colVal, colVal))) == 0:
                #     return col, colVal
        return False

    def checkInsertConstraint(self, table: str, colVals, thisRID: RID = None):
        if self.checkForeign(table, colVals):
            miss = self.checkForeign(table, colVals)
            print("OH NO")
            raise MissForeignKeyError("miss: " + str(miss[0]) + ": " + str(miss[1]))

        if self.checkPrimary(table, colVals, thisRID):
            dup = self.checkPrimary(table, colVals, thisRID)
            print("OH NO")
            raise DuplicatedPrimaryKeyError("duplicated: " + str(dup[0]) + ": " + str(dup[1]))

        if self.checkUnique(table, colVals, thisRID):
            dup = self.checkUnique(table, colVals, thisRID)
            print("OH NO")
            raise DuplicatedUniqueKeyError("duplicated: " + str(dup[0]) + ": " + str(dup[1]))

        return

    def checkRemoveColumn(self, table: str, col: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        for tableInfo in metaHandler.databaseInfo.tableMap.values():
            if tableInfo.name != table and len(tableInfo.foreign) > 0:
                for fromcol, (tab, column) in tableInfo.foreign.items():
                    if tab == table and col == column:
                        raise RemoveError("referenced foreignkey column")
        return False

    def checkRemoveConstraint(self, table: str, colVals):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        thistable = metaHandler.collectTableInfo(table)
        for tableInfo in metaHandler.databaseInfo.tableMap.values():
            if len(tableInfo.foreign) > 0:
                for fromcol, (tab, col) in tableInfo.foreign.items():
                    if tab == table:
                        colval = colVals[thistable.getColumnIndex(col)]
                        index = self.IM.start_index(self.inUse, tableInfo.name, tableInfo.index[fromcol])
                        if len(set(index.range(colval, colval))) != 0:
                            raise RemoveError("referenced foreignkey value")
        return False

    def handleInsertIndex(self, table: str, data: tuple, rid: RID):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        for col in tableInfo.index:
            if data[tableInfo.getColumnIndex(col)]:
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                index.insert(data[tableInfo.getColumnIndex(col)], rid)
            else:
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                index.insert(NULL_VALUE, rid)
        return

    def handleRemoveIndex(self, table: str, data: tuple, rid: RID):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        for col in tableInfo.index:
            if data[tableInfo.getColumnIndex(col)]:
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                index.delete(data[tableInfo.getColumnIndex(col)], rid)
            else:
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                index.delete(NULL_VALUE, rid)
        return

    def buildConditionsFuncs(self, table: str, limits, metahandler):
        tableInfo = metahandler.collectTableInfo(table)

        def build(limit: Term):
            if limit.table is not None and limit.table != table:
                return None
            limit_col = limit.col
            colIndex = tableInfo.getColumnIndex(limit_col)
            if colIndex is not None:
                colType = tableInfo.columnType[colIndex]
                if limit.type == 1:
                    if limit.aim_col:
                        if limit.aim_table == table:
                            return self.compare(colIndex, limit.operator, tableInfo.getColumnIndex(limit.aim_col))
                        return None
                    else:
                        if colType == "DATE":
                            if type(limit.value) not in (str, date):
                                raise ValueTypeError("need str/date here")
                            val = limit.value
                            if type(val) is date:
                                return self.compareV(colIndex, limit.operator, val)
                            valist = val.replace("/", "-").split("-")
                            return self.compareV(colIndex, limit.operator, date(*map(int, valist)))
                        elif colType in ("INT", "FLOAT"):
                            if isinstance(limit.value, (int, float)):
                                return self.compareV(colIndex, limit.operator, limit.value)
                            raise ValueTypeError("need int/float here")
                        elif colType == "VARCHAR":
                            if isinstance(limit.value, str):
                                return self.compareV(colIndex, limit.operator, limit.value)
                            raise ValueTypeError("need varchar here")
                        raise ValueTypeError("limit value error")
                elif limit.type == 2:
                    if colType == "DATE":
                        values = []
                        for val in limit.value:
                            if type(val) is str:
                                valist = val.replace("/", "-").split("-")
                                values.append(date(*map(int, valist)))
                            elif type(val) is date:
                                values.append(val)
                            raise ValueTypeError("need str/date here")
                        return lambda x: x[colIndex] in tuple(values)
                    return lambda x: x[colIndex] in limit.value
                elif limit.type == 3:
                    if colType == "VARCHAR":
                        return lambda x: self.buildPattern(limit.value).match(str(x[colIndex]))
                    raise ValueTypeError("like need varchar here")
                elif limit.type == 0:
                    if isinstance(limit.value, bool):
                        if limit.value:
                            return lambda x: x[colIndex] is None
                        return lambda x: x[colIndex] is not None
                    raise ValueTypeError("limit value need bool here")
                raise ValueTypeError("limit type unknown")
            raise ColumnNotExist("limit column name unknown")

        results = []
        for limit in limits:
            func = build(limit)
            if func is not None:
                results.append(func)
        return results

    def selectRecords(self, reducers: Tuple[Reducer], tables: Tuple[str, ...],
                      limits: Tuple[Term], groupBy: Tuple[str, str]):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()

        def setTableName(object, tableName, colName):
            if getattr(object, colName) is None:
                return
            elif getattr(object, tableName) is None:
                tabs = col2tab.get(getattr(object, colName))
                if not tabs:
                    raise ColumnNotExist(getattr(object, colName) + " unknown")
                elif len(tabs) > 1:
                    raise SameNameError(getattr(object, colName) + " exists in multiple tables")
                setattr(object, tableName, tabs[0])
            return

        def getSelected(col2data):
            col2data['*.*'] = next(iter(col2data.values()))
            return tuple(map(lambda x: x.select(col2data[x.target()]), reducers))

        col2tab = metaHandler.getColumn2Table(tables)
        groupTable, groupCol = groupBy
        for element in limits + reducers:
            if not isinstance(element, Term):
                setTableName(element, '_table_name', '_col')
            else:
                setTableName(element, 'aim_table', 'aim_col')

        groupTableName = groupTable or tables[0]
        groupTable = groupTableName
        groupBy = groupTable + '.' + groupCol
        reducerTypes = []
        for reducer in reducers:
            reducerTypes.append(reducer.reducer_type)
        reducerTypes = set(reducerTypes)
        if not groupCol and 1 in reducerTypes:
            if len(reducerTypes) > 1:
                raise SelectError("no-group select contains both field and aggregations")

        if not reducers and not groupCol and len(tables) == 1 and reducers[0].reducer_type == 3:
            tableInfo = metaHandler.collectTableInfo(tables[0])
            fileHandler = self.RM.openFile(self.getTablePath(tables[0]))
            return LookupOutput((reducers[0].to_string(False),), (fileHandler.head['AllRecord']))
        tab2results = {}
        for table in tables:
            tab2results[table] = self.condScanIndex(table, limits)
        result = None
        if len(tables) == 1:
            result = tab2results[tables[0]]
        else:
            result = self.condJoin(tab2results, limits)

        if not groupCol:
            if reducers[0].reducer_type == 0:
                if len(reducers) == 1:
                    return result
                raise SelectError("reducer num not 1")
            elif 1 in reducerTypes:
                heads = []
                headindexes = []
                for reducer in reducers:
                    heads.append(reducer.target())
                headers = tuple(heads)
                for head in headers:
                    headindexes.append(result.header_id(head))
                indexes = tuple(headindexes)

                def takeCol(row):
                    return tuple(row[ele] for ele in indexes)

                data = tuple(map(takeCol, result.data))
            else:
                if result.data is not None:
                    head2data = {}
                    for head, data in zip(result.headers, zip(*result.data)):
                        head2data[head] = data
                    data = getSelected(head2data)
                else:
                    data = (None,) * len(result.headers)
        else:
            def getRow(group):
                head2data = {}
                for item_head, item_data in zip(result.headers, zip(*group)):
                    head2data[item_head] = item_data
                return getSelected(head2data)

            index = result.header_id(groupBy)
            groups = {}
            for row in result.data:
                if groups.get(row[index]) is None:
                    groups[row[index]] = [row]
                else:
                    groups[row[index]].append(row)
            if reducers[0].reducer_type == 0:
                return LookupOutput(result.headers, tuple(group[0] for group in groups.values()))
            data = tuple(map(getRow, groups.values()))

        headers = []
        for reducer in reducers:
            headers.append(reducer.to_string(len(tables) > 1))
        return LookupOutput(tuple(headers), data)

    def selectRecordsLimit(self, reducers, tables, limits, groupBy, limit: int, off: int):
        result = self.selectRecords(reducers, tables, limits, groupBy)
        if limit is None:
            data = result.data[off:]
        else:
            right = off + limit
            data = result.data[off: right]
        res = LookupOutput(result.headers, data)
        return res

    def condScanIndex(self, table: str, limits: tuple):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        records, data = self.searchRecordIndex(table, limits)
        headers = tuple(tableInfo.name + "." + colName for colName in tableInfo.columnMap.keys())
        return LookupOutput(headers, data)

    @staticmethod
    def resultToValue(result: LookupOutput, is_in):
        if len(result.headers) <= 1:
            val = sum(result.data, ())
            if not is_in:
                if len(result.data) == 1:
                    val, = val
                    return val
                raise ValueError("expect one value, get " + str(len(result.data)))
            return val
        raise SelectError("expect one column, get " + str(len(result.headers)))

    @staticmethod
    def printResults(result: LookupOutput):
        TablePrinter().print([result])

    @staticmethod
    def compare(this, operator, other):
        if operator == "<":
            return lambda x: x[this] < x[other]
        elif operator == "<=":
            return lambda x: x[this] <= x[other]
        elif operator == ">":
            return lambda x: x[this] > x[other]
        elif operator == ">=":
            return lambda x: x[this] >= x[other]
        elif operator == "<>":
            return lambda x: x[this] != x[other]
        elif operator == "=":
            return lambda x: x[this] == x[other]

    @staticmethod
    def compareV(this, operator, val):
        if operator == '<':
            return lambda x: x is not None and x[this] < val
        elif operator == '<=':
            return lambda x: x is not None and x[this] <= val
        elif operator == '>':
            return lambda x: x is not None and x[this] > val
        elif operator == '>=':
            return lambda x: x is not None and x[this] >= val
        elif operator == '<>':
            return lambda x: x[this] != val
        elif operator == '=':
            return lambda x: x[this] == val

    @staticmethod
    def buildPattern(pat: str):
        pat = pat.replace('%%', '\r')
        pat = pat.replace('%?', '\n')
        pat = pat.replace('%_', '\0')
        pat = re.escape(pat)
        pat = pat.replace('%', '.*')
        pat = pat.replace(r'\?', '.')
        pat = pat.replace('_', '.')
        pat = pat.replace('\r', '%')
        pat = pat.replace('\n', r'\?')
        pat = pat.replace('\0', '_')
        pat = re.compile('^' + pat + '$')
        return pat
