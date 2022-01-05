
import numpy as np
from json import loads

from .RecordManager import RecordManager


class FileHandler:

    def __init__(self, rm: RecordManager, fid: int, name: str):
        self.RM = rm
        self.fileID = fid
        self.name = name

        self.headChanged = False
        self.open = True
        self.headpage = self.RM.BM.getPage(self.fileID, )
        self.head = loads(self.headpage.tobytes().decode('utf-8').rstrip('\0'))



    def changeHead(self):

