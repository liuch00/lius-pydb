

from .system_visitor import SystemVisitor
from FileSystem.FileManager import FileManager
from FileSystem.BufManager import BufManager
from IndexSystem.index_manager import IndexManager
from MetaSystem.MetaManager import MetaManager

from pathlib import Path

class SystemManger:
    def __init__(self, visitor: SystemVisitor, syspath: Path, bm: BufManager, im: IndexManager, mm: MetaManager):
        self.visitor = visitor
        self.systemPath = syspath
        self.BM = bm
        self.IM = im
        self.MM = mm
        self.databaselist = []
        for dbname in syspath.iterdir():
            self.databaselist.append(dbname)
        self.inUse = None
        self.visitor.manager = self

