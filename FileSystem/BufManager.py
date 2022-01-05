
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
        self.index2FPID = np.zeros(CAP)
        for i in range(CAP):
            self.index2FPID[i] = -1
        self.FPID2index = {}


    def combine_PFID(self, fileID, pageID):
        return fileID | (pageID << 16)

    def split_PFID(self, PFID):
        """return fileID, pageID"""
        return PFID & ((1 << 16) - 1), PFID >> 16

    def access(self, index):
        if index == self.last:
            return
        self.replace.access(index)
        self.last = index

    def markdirty(self, index):
        self.dirty[index] = True
        self.access(index)

    def release(self, index):
        self.dirty[index] = False
        self.replace.free(index)

    def
