
from ManageSystem.macro import *
from .FileHandler import FileHandler
from .rid import RID

import numpy as np

class FileScan:
    def __init__(self, handler: FileHandler):
        self.handler = handler

    def __iter__(self):
        pageNum = self.handler.head['PageNum']
        for pID in range(1, pageNum):
            page = self.handler.RM.BM.getPage(self.handler.fileID, pID)
            if page[PAGE_FLAG_OFFSET] != RECORD_PAGE_FLAG:
                continue
            else:
                bitmap = self.handler.getBitmap(page)
                for slot in range(len(bitmap)):
                    if bitmap[slot] == 0:
                        rid = RID(pID, slot)
                        yield self.handler.getRecord(rid, page)
