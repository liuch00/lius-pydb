from pathlib import Path
from .join import Join
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.Errors import ParseCancellationException
from antlr4.error.ErrorListener import ErrorListener
from copy import deepcopy
from .system_visitor import SystemVisitor
from FileSystem.FileManager import FileManager
from FileSystem.BufManager import BufManager
from RecordSystem.RecordManager import RecordManager
from RecordSystem.FileScan import FileScan
from RecordSystem.FileHandler import FileHandler
from RecordSystem.record import Record
from RecordSystem.rid import RID
from IndexSystem.index_manager import IndexManager
from MetaSystem.MetaHandler import MetaHandler
from Exceptions.exception import *
from .lookup_element import LookupOutput, Term, Join
from SQL_Parser.SQLLexer import SQLLexer
from SQL_Parser.SQLParser import SQLParser
from MetaSystem.info import TableInfo, ColumnInfo
from .macro import *


class SystemManger:
    def __init__(self, visitor: SystemVisitor, syspath: Path, bm: BufManager, rm: RecordManager, im: IndexManager):
        self.visitor = visitor
        self.systemPath = syspath
        self.BM = bm
        self.IM = im
        self.RM = rm
        self.metaHandlers = {}
        self.databaselist = []
        for dir in syspath.iterdir():
            self.databaselist.append(dir.name)
        self.inUse = None
        self.visitor.manager = self

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
            return [LookupOutput(None, None, str(e), cost=self.visitor.spend_time())]
        try:
            return self.visitor.visit(tree)
        except MyException as e:
            return [LookupOutput(message=str(e), cost=self.visitor.spend_time())]

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
        if col in tableInfo.index:
            metaHandler.createIndex(index, table, col)
            return
        indexFile = self.IM.create_index(self.inUse, table)
        tableInfo.index[col] = indexFile._root

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
        metaHandler.collectTableInfo(table).indexes.pop(col)
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
        tableInfo.addForeign(col, foreign)
        metaHandler.shutdown()
        if forName:
            if forName not in metaHandler.databaseInfo.indexMap:
                self.createIndex(forName, foreign[0], foreign[1])
        else:
            indexName = foreign[0] + "." + foreign[1]
            if indexName not in metaHandler.databaseInfo.indexMap:
                self.createIndex(indexName, foreign[0], foreign[1])
        return

    def removeForeign(self, table: str, col: str, forName=None):
        metaHandler, tableInfo = self.collectTableinfo(table)
        if forName:
            if metaHandler.databaseInfo.indexMap.get(forName):
                self.removeIndex(forName)
        else:
            if tableInfo.foreign.get(col) is not None:
                foreign = tableInfo.foreign[col][0] + "." + tableInfo.foreign[col][1]
                if metaHandler.databaseInfo.indexMap.get(foreign):
                    self.removeIndex(foreign)
            tableInfo.removeForeign(col)
            metaHandler.shutdown()

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

    def removePrimary(self, table: str):
        metaHandler, tableInfo = self.collectTableinfo(table)
        if tableInfo.primary:
            for column in tableInfo.primary:
                indexName = table + "." + column
                if indexName in metaHandler.databaseInfo.indexMap:
                    self.removeIndex(indexName)
        return

    def addColumn(self, table: str, col: ColumnInfo):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if tableInfo.getColumnIndex(col.name):
            print("OH NO")
            raise ColumnAlreadyExist(col.name + " exists")
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
        return LookupOutput('deleted_items', (len(records),))

    def updateRecords(self, table: str, limits: tuple, valmap: dict):
        self.checkInUse()
        fileHandler = self.RM.openFile(self.getTablePath(table))
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        tableInfo.checkValue(valmap)
        records, data = self.searchRecordIndex(table, limits)
        for record, oldVal in zip(records, data):

    def indexFilter(self, table: str, limits: tuple) -> set:

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
                if all(map(lambda fun: fun(valTuple), functions)):
                    records.append(record)
                    data.append(valTuple)
        else:
            for record in FileScan(fileHandler):
                valTuple = tableInfo.loadRecord(record)
                if all(map(lambda fun: fun(valTuple), functions)):
                    records.append(record)
                    data.append(valTuple)
        return records, data

    def condJoin(self, res_map: dict, term):
        if self.inUse is None:
            raise ValueError("No using database!!!")
        else:
            join = Join(res_map=res_map, term=term)
            result: LookupOutput = join.get_output()
            return result

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
                colVal = colVals[tableInfo.getColumnIndex(col)]
                foreignTableInfo: TableInfo = metaHandler.collectTableInfo(tableInfo.foreign[col][0])
                index = self.IM.start_index(self.inUse, tableInfo.foreign[col][0],
                                            foreignTableInfo.index[tableInfo.foreign[col][1]])
                if len(set(index.range(colVal, colVal))) == 0:
                    return col, colVal
        return False

    def checkInsertConstraint(self, table: str, colVals, thisRID: RID = None):
        if self.checkPrimary(table, colVals, thisRID):
            dup = self.checkPrimary(table, colVals, thisRID)
            print("OH NO")
            raise DuplicatedPrimaryKeyError("duplicated: " + str(dup[0]) + ": " + str(dup[1]))

        if self.checkUnique(table, colVals, thisRID):
            dup = self.checkUnique(table, colVals, thisRID)
            print("OH NO")
            raise DuplicatedUniqueKeyError("duplicated: " + str(dup[0]) + ": " + str(dup[1]))

        if self.checkForeign(table, colVals):
            miss = self.checkForeign(table, colVals)
            print("OH NO")
            raise MissForeignKeyError("miss: " + str(miss[0]) + ": " + str(miss[1]))
        return

    def checkRemoveConstraint(self, table: str, colVals):

        return

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
