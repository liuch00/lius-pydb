
from FileSystem.BufManager import BufManager

class MetaManager:
    def __init__(self, bm: BufManager, syspath: str):
        self.BM = bm
        self.systemPath = syspath
        self.handlers = {}

    def shutHandler(self, database: str):
        if self.handlers.get(database) is not None:
            self.handlers.pop(database).shutdown()
        return

    def shutdown(self):
        namelist = self.handlers.keys()
        for name in namelist:
            self.shutHandler(name)
        return