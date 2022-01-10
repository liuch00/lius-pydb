
from Exceptions.exception import *
from FileSystem.FileManager import FileManager
from FileSystem.BufManager import BufManager
from .FileHandler import FileHandler
from ManageSystem.macro import *
from json import dumps, loads

import numpy as np

class RecordManager:

    def __init__(self, bm: BufManager):
        self.BM = bm
        self.opened = {}

    def createFile(self, name: str, recordLen: int):
        self.BM.FM.createFile(name)
        fileID = self.BM.openFile(name)

        self.BM.FM.newPage(fileID, self.toSerial(self.getHead(recordLen, name)))
        self.BM.closeFile(fileID)
        return

    def getHead(self, recordLen: int, name: str):
        recordNum = self.getRecordNum(recordLen)
        bitmapLen = self.getBitmapLen(recordNum)
        return {'RecordLen': recordLen, 'RecordNum': recordNum, 'PageNum': 1,
                'AllRecord': 0, 'NextAvai': 0, 'BitmapLen': bitmapLen, 'filename': str(name)}

    def openFile(self, name: str):
        if name in self.opened:
            handler = self.opened[name]
            return handler
        fID = self.BM.openFile(name)
        self.opened[name] = FileHandler(self, fID, name)
        return self.opened[name]

    def destroyFile(self, name: str):
        self.BM.FM.destroyFile(name)
        return

    def renameFile(self, src: str, dst: str):
        if self.opened.get(src) is None:
            self.BM.FM.renameFile(src, dst)
        else:
            self.closeFile(src)
            self.BM.FM.renameFile(src, dst)
        return

    def closeFile(self, name: str):
        if self.opened.get(name) is None:
            return
        handler = self.opened.get(name)
        if handler.headChanged:
            handler.changeHead()
        self.BM.closeFile(handler.fileID)
        self.opened.pop(name)
        handler.open = False
        return

    @staticmethod
    def getRecordNum(recordLen: int):
        nohead = PAGE_SIZE - RECORD_PAGE_FIXED_HEADER
        num = (nohead * 8) // (1 + (recordLen * 8)) + 1
        total = ((num + 7) / 8) + num * recordLen
        while total > nohead:
            num = num - 1
            total = ((num + 7) / 8) + num * recordLen
        if num <= 0:
            print("OH NO")
            raise RecordTooLong("record too long")
        return num

    @staticmethod
    def getBitmapLen(recordNum: int) -> int:
        length = (recordNum + 7) / 8
        return int(length)


    def replaceFile(self, src: str, dst: str):
        if self.opened.get(src) is not None:
            self.closeFile(src)
        if self.opened.get(dst) is not None:
            self.closeFile(dst)
        self.destroyFile(dst)
        self.BM.FM.renameFile(src, dst)
        return

    def shutdown(self):
        for name in tuple(self.opened.keys()):
            self.closeFile(name)

    @staticmethod
    def toSerial(d: dict):
        serial = dumps(d, ensure_ascii=False).encode('utf-8')
        empty = np.zeros(PAGE_SIZE, dtype=np.uint8)
        for i in range(len(serial)):
            empty[i] = list(serial)[i]
        return empty