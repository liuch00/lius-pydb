import numpy as np

from .macro import *
from .FileManager import FileManager
from .FindReplace import FindReplace


class BufManager:
    def __init__(self, fm: FileManager):
        self.FM = fm
        self.c = CAP
        self.m = MOD
        self.last = -1
        self.dirty = np.zeros(CAP, dtype=np.bool)
        self.addr = np.zeros((CAP, PAGE_SIZE), dtype=np.uint8)
        self.replace = FindReplace(CAP)
        self.index2FPID = np.zeros(CAP, dtype=np.int64)
        for i in range(CAP):
            self.index2FPID[i] = -1
        self.FPID2index = {}
        self.index_in_file = {}

    def combine_FPID(self, fileID, pageID):
        return fileID | (pageID << 16)

    def split_FPID(self, FPID):
        """return fileID, pageID"""
        return FPID & ((1 << 16) - 1), FPID >> 16

    def access(self, index):
        if index == self.last:
            return

        self.last = index
        self.replace.access(index)
        return

    def markDirty(self, index):
        self.dirty[index] = True
        self.access(index)
        return

    def release(self, index):
        self.dirty[index] = False
        self.replace.free(index)
        self.index2FPID[index] = -1
        fpID = self.index2FPID[index]
        self.FPID2index.pop(fpID)
        fID = self.split_FPID(fpID)[0]
        self.index_in_file[fID].remove(index)
        return

    def writeBack(self, index):
        if self.dirty[index]:
            fpID = self.index2FPID[index]
            fID = self.split_FPID(fpID)[0]
            pID = self.split_FPID(fpID)[1]
            self.FM.writePage(fID, pID, self.addr[index])
            self.dirty[index] = False
        self.replace.free(index)
        self.index2FPID[index] = -1
        fpID = self.index2FPID[index]
        self.FPID2index.pop(fpID)
        fID = self.split_FPID(fpID)[0]
        self.index_in_file[fID].remove(index)
        return

    def openFile(self, name: str):
        fID = self.FM.openFile(name)
        self.index_in_file[fID] = set()
        return fID

    def closeFile(self, fileID: int):
        for index in self.index_in_file.pop(fileID, {}):
            self.replace.free(index)

            fpID = self.index2FPID[index]
            self.index2FPID[index] = -1
            self.FPID2index.pop(fpID)
            if self.dirty[index]:
                # fID, pID = self.split_FPID(fpID)
                self.FM.writePage(*(self.split_FPID(fpID)), self.addr[index])
                self.dirty[index] = False
        self.FM.closeFile(fileID)
        return

    def fetchPage(self, fileID: int, pageID: int, buf: np.ndarray):
        fpID = self.combine_FPID(fileID, pageID)
        index = self.FPID2index.get(fpID)
        fID = self.split_FPID(fpID)[0]
        pID = self.split_FPID(fpID)[1]
        if index is None:
            self.getPage(fID, pID)
            index = self.FPID2index[fpID]
        self.dirty[index] = True
        self.addr[index] = buf
        self.replace.access(index)
        return

    def getPage(self, fileID: int, pageID: int):
        fpID = self.combine_FPID(fileID, pageID)
        index = self.FPID2index.get(fpID)
        if index is None:
            index = self.replace.find()
            foundFP = self.index2FPID[index]
            if foundFP == -1:
                self.FPID2index[fpID] = index
                self.index2FPID[index] = fpID
                fID = self.split_FPID(fpID)[0]
                pID = self.split_FPID(fpID)[1]
                buf = self.FM.readPage(fID, pID)
                buf = np.frombuffer(buf, np.uint8, PAGE_SIZE)
                self.index_in_file[fileID].add(index)
                self.addr[index] = buf
            else:
                self.writeBack(index)
        else:
            self.access(index)

        return self.addr[index].copy()

    def shutdown(self):
        for i in np.where(self.dirty)[0]:
            self.writeBack(i)
        # no need to clear dirty
        self.addr = np.zeros((CAP, PAGE_SIZE), dtype=np.uint8)
        self.index2FPID = np.zeros(CAP)
        for i in range(CAP):
            self.index2FPID[i] = -1
        self.FPID2index = {}
        self.last = -1
        while self.index_in_file:
            fID = self.index_in_file.popitem()[0]
            self.closeFile(fID)

    def createFile(self, name: str):
        self.FM.createFile(name)
        return

    def fileExist(self, name: str):
        return self.FM.fileExist(name)

    def destroyFile(self, name: str):
        return self.FM.destroyFile(name)

    def renameFile(self, src: str, dst: str):
        self.FM.renameFile(src, dst)
        return

    def writePage(self, fileID: int, pageID: int, buf: np.ndarray):
        self.FM.writePage(fileID, pageID, buf)
        return

    def readPage(self, fileID: int, pageID: int):
        return self.FM.readPage(fileID, pageID)

    def newPage(self, fileID: int, buf: np.ndarray):
        return self.FM.newPage(fileID, buf)