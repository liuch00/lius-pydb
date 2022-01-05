
from .macro import *
from Exceptions.exception import *

import numpy as np
import os

class FileManager:

    def __init__(self):
        self._fd = np.zeros(MAX_FILE_NUM)

    def createFile(self, name: str):
        f = open(name, 'w')
        if f is None:
            print("fail to create " + name)
            raise FailCreateError
        f.close()
        return

    def createExist(self, name: str):
        f = open(name, 'a')
        f.close()
        return

    def destroyFile(self, name: str):
        os.remove(name)
        return

    def fileExist(self, name: str):
        if os.path.exists(name):
            return True
        return False

    def renameFile(self, src: str, dst: str):
        os.rename(src, dst)
        return

    def openFile(self, name: str):
        fileID = os.open(name, os.O_RDWR)
        if fileID == -1:
            print("fail to open " + name)
            raise FailOpenError
        return fileID

    def closeFile(self, fileID: int):
        os.close(fileID)
        return

    def writePage(self, fileID: int, pageID: int, buf: np.ndarray):
        offset = pageID
        offset = offset << PAGE_SIZE_IDX
        error = os.lseek(fileID, offset, os.SEEK_SET)
        os.write(fileID, buf.tobytes())
        return

    def readPage(self, fileID: int, pageID: int):
        offset = pageID
        offset = offset << PAGE_SIZE_IDX
        error = os.lseek(fileID, offset, os.SEEK_SET)
        error = os.read(fileID, PAGE_SIZE)
        if error is None:
            print("fail to read pid: " + str(pageID) + ", fid: " + str(fileID))
            raise FailReadPageError
        return error

    def newPage(self, fileID: int, buf: np.ndarray):
        offset = os.lseek(fileID, 0, os.SEEK_END)
        os.write(fileID, buf.tobytes())
        pID = offset >> PAGE_SIZE_IDX
        return pID
