
from FileSystem.MyBitMap import MyBitMap
from .macro import *
from Exceptions.exception import *

import numpy as np
import os

class FileManager:

    def __init__(self):
        self._fd = np.zeros(MAX_FILE_NUM)
        self._fm = MyBitMap(MAX_FILE_NUM, 1)

    def createFile(self, name: str):
        f = open(name, 'w')
        if f is None:
            print("fail to create " + name)
            raise FailCreateError
        f.close()

    def openFile(self, name: str):
        fileID = os.open(name, os.O_RDWR)
        if fileID == -1:
            print("fail to open " + name)
            raise FailOpenError
        return fileID

    def closeFile(self, fileID: int):
        os.close(fileID)

    def writePage(self, fileID: int, pageID: int, buf: np.ndarray):
        offset = pageID
        offset = offset << PAGE_SIZE_IDX
        error = os.lseek(fileID, offset, os.SEEK_SET)
        os.write(fileID, buf.tobytes())

    def readPage(self, fileID: int, pageID: int):
        offset = pageID
        offset = offset << PAGE_SIZE_IDX
        error = os.lseek(fileID, offset, os.SEEK_SET)
        error = os.read(fileID, PAGE_SIZE)
        if error is None:
            print("fail to read pid: " + str(pageID) + ", fid: " + str(fileID))
            raise FailReadPageError
        return error

