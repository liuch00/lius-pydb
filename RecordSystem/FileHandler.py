
import numpy as np
from json import loads

from .RecordManager import RecordManager
from .macro import *
from .rid import RID
from .record import Record

class FileHandler:

    def __init__(self, rm: RecordManager, fid: int, name: str):
        self.RM = rm
        self.fileID = fid
        self.name = name

        self.headChanged = False
        self.open = True # not opened
        self.headpage = self.RM.BM.getPage(self.fileID, 0)
        self.head = loads(self.headpage.tobytes().decode('utf-8').rstrip('\0'))

    def changeHead(self):
        serialized = self.RM.toSerial(self.head)
        self.RM.BM.fetchPage(self.fileID, 0, serialized)
        return

    def getBitmap(self, pageBuf: np.ndarray):
        num = self.head['RecordNum']
        bml = self.head['BitmapLen']
        bitinfo = pageBuf[RECORD_PAGE_FIXED_HEADER: RECORD_PAGE_FIXED_HEADER + bml]
        bitmap = np.unpackbits(bitinfo)[:num]
        return bitmap

    def getNextAvai(self, pageBuf: np.ndarray):
        byte = pageBuf[RECORD_PAGE_NEXT_OFFSET: RECORD_PAGE_NEXT_OFFSET + 4].tobytes()
        return int.from_bytes(byte, 'big')

    def setNextAvai(self, pageBuf: np.ndarray, pageID: int):
        data = np.frombuffer(pageID.to_bytes(4, 'big'), dtype=np.uint8)
        pageBuf[RECORD_PAGE_NEXT_OFFSET: RECORD_PAGE_NEXT_OFFSET + 4] = data
        return

    def getPage(self, pageID: int):
        return self.RM.BM.getPage(self.fileID, pageID)

    def fetchPage(self, pageID: int, buf: np.ndarray):
        self.RM.BM.fetchPage(self.fileID, pageID, buf)
        return

    def newPage(self):
        pID = self.RM.BM.FM.newPage(self.fileID, np.zeros(PAGE_SIZE, dtype=np.uint8))
        return pID

    def getRecord(self, rid: RID, record, buf=None):
        slotID = rid.slot
        if buf is None:
            pID = rid.page
            buf = self.RM.BM.getPage(self.fileID, pID)
        recordLen = self.head['RecordLen']
        bitmapLen = self.head['BitmapLen']
        recordOff = RECORD_PAGE_FIXED_HEADER + bitmapLen + slotID * recordLen
        record = Record(rid, buf[recordOff: recordOff + recordLen])
        return record

    def Nextvalid(self):
        if self.head['NextAvai'] == 0:
            return False
        return True

    def insertRecord(self, record: np.ndarray):
        if not self.Nextvalid():
            self.appendPage()
        nextAvai = self.head['NextAvai']
        page = self.RM.BM.getPage(self.fileID, nextAvai)
        bitmap = self.getBitmap(page)

        slotID = np.where(bitmap)[0]
        if len(np.where(bitmap)) == 1:
            self.head['NextAvai'] = self.getNextAvai(page)
            self.setNextAvai(page, nextAvai)

        recordLen = self.head['RecordLen']
        bitmapLen = self.head['BitmapLen']
        recordOff = RECORD_PAGE_FIXED_HEADER + bitmapLen + slotID * recordLen

        bitmap[slotID] = False
        self.headChanged = True
        page[recordOff: recordOff + recordLen] = record
        bitmap = np.packbits(bitmap)
        page[RECORD_PAGE_FIXED_HEADER: RECORD_PAGE_FIXED_HEADER + bitmapLen] = bitmap
        self.head['AllRecord'] += 1

        self.RM.BM.fetchPage(self.fileID, nextAvai, page)
        rid = RID(nextAvai, slotID)
        return rid

    def appendPage(self):
        buf = np.full(PAGE_SIZE, -1, dtype=np.uint8)
        buf[PAGE_FLAG_OFFSET] = RECORD_PAGE_FLAG
        self.setNextAvai(buf, self.head['NextAvai'])
        pID = self.RM.BM.FM.newPage(self.fileID, buf)
        self.head['NextAvai'] = pID
        self.headChanged = True
        self.head['PageNum'] += 1
        return

    def deleteRecord(self, rid: RID):
        page = self.RM.BM.getPage(self.fileID, rid.page)
        bitmap = self.getBitmap(page)
        bitmapLen = self.head['BitmapLen']
        self.head['AllRecord'] -= 1
        if bitmap[rid.slot] is False:
            bitmap[rid.slot] = True
        self.headChanged = True

        bitmap = np.packbits(bitmap)
        page[RECORD_PAGE_FIXED_HEADER: RECORD_PAGE_FIXED_HEADER + bitmapLen] = bitmap
        if self.getNextAvai(page) == rid.page:
            self.setNextAvai(page, self.head['NextAvai'])
            self.head['NextAvai'] = rid.page
        self.RM.BM.fetchPage(self.fileID, rid.page, page)
        return

    def updateRecord(self, record: Record):
        page = self.RM.BM.getPage(self.fileID, record.rid.page)

        recordLen = self.head['RecordLen']
        bitmapLen = self.head['BitmapLen']
        recordOff = RECORD_PAGE_FIXED_HEADER + bitmapLen + record.rid.slot * recordLen
        page[recordOff: recordOff + recordLen] = record.record
        self.RM.BM.fetchPage(self.fileID, record.rid.page, page)

    def __del__(self):
        if self.open:
            self.RM.closeFile(self.name)
