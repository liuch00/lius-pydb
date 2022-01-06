
from .info import *
import pickle as pic
import os


class MetaHandler:
    def __init__(self, database: str, syspath: str):

        self.databaseName = database
        self.systemPath = syspath
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.databaseInfo = None

        if not os.path.exists(self.metaPath):
            self.databaseInfo = DatabaseInfo(self.databaseName, [])
            self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
            self.toPickle(self.metaPath)
        else:
            self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
            metaInfo = open(self.metaPath, 'rb')
            self.databaseInfo = pic.load(metaInfo)
            metaInfo.close()

    def toPickle(self, path: str):
        metaInfo = open(path, 'wb')
        pic.dump(self.databaseInfo, metaInfo)
        metaInfo.close()
        return

    def insertTable(self, table: TableInfo):
        self.databaseInfo.insertTable(table)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def removeTable(self, table: str):
        self.databaseInfo.removeTable(table)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def getTableInfo(self, table: str):
        if self.databaseInfo.tableMap.get(table) is None:
            print("OH NO")
            raise TableNotExist("this name doesn't exist")
        tableInfo = self.databaseInfo.tableMap[table]
        return tableInfo

    def insertColumn(self, table: str, col: ColumnInfo):
        self.databaseInfo.insertColumn(table, col)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def removeColumn(self, table: str, column: str):
        self.databaseInfo.removeColumn(table, column)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def createIndex(self, index: str, table: str, column: str):
        self.databaseInfo.createIndex(index, table, column)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def removeIndex(self, index: str):
        self.databaseInfo.removeIndex(index)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def setPrimary(self, table: str, pri: str):
        self.getTableInfo(table).primary = pri
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def removePrimary(self, table: str):
        self.getTableInfo(table).primary = None
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def addUnique(self, table: str, column: str, uniq: str):
        self.getTableInfo(table).addUnique(column, uniq)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def renameTable(self, src: str, dst: str):
        if self.databaseInfo.tableMap.get(src) is None:
            print("OH NO")
            raise TableNotExist("this name doesn't exist")
        srcInfo = self.databaseInfo.tableMap.pop(src)
        self.databaseInfo.tableMap[dst] = srcInfo
        indexMap = self.databaseInfo.indexMap
        for index in indexMap.keys():
            if indexMap.get(index)[0] == src:
                columnName = indexMap.get(index)[1]
                self.databaseInfo.indexMap[index] = (dst, columnName)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def renameIndex(self, src: str, dst: str):
        if self.databaseInfo.indexMap.get(src) is None:
            print("OH NO")
            raise IndexNotExist("this name doesn't exist")
        info = self.databaseInfo.indexMap.pop(src)
        self.databaseInfo.indexMap[dst] = info
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def shutdown(self):
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def getColumn2Table(self, tables: list):
        result = {}
        for table in tables:
            tableInfo = self.getTableInfo(table)
            for col in tableInfo.columnMap.keys():
                colInfo = tableInfo.columnMap.get(col)
                if result.get(col) is None:
                    result[col] = [colInfo]
                else:
                    result[col].append(colInfo)
        return result
