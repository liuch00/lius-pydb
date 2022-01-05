

from .MyLinkList import MyLinkList


class FindReplace:
    def __init__(self, cap: int):
        self.CAP = cap
        self.list = MyLinkList(cap, 1)
        for i in range(cap):
            self.list.insert(0, i)

    def find(self):
        index = self.list.getFirst(0)
        self.list.delete(index)
        self.list.insert(0, index)
        return index

    def access(self, index: int):
        self.list.insert(0, index)

    def free(self, index: int):
        self.list.insertFirst(0, index)
