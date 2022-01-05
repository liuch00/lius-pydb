
import numpy as np


class MyLinkList:
    def __init__(self, c: int, n: int):
        self.cap = c
        self.list_num = n
        self.next = np.arange(c + n)
        self.prev = np.arange(c + n)

    def link(self, prev: int, next: int):
        self.next[prev] = next
        self.prev[next] = prev

    def delete(self, index: int):
        if self.prev[index] == index:
            return
        self.link(self.prev[index], self.next[index])
        self.prev[index] = index
        self.next[index] = index

    def insert(self, listID: int, ele: int):
        self.delete(ele)
        node = listID + self.cap
        prev = self.prev[node]
        self.link(prev, ele)
        self.link(ele, node)

    def insertFirst(self, listID: int, ele: int):
        self.delete(ele)
        node = listID + self.cap
        next = self.next[node]
        self.link(node, ele)
        self.link(ele, next)

    def getFirst(self, listID: int):
        return self.next[listID + self.cap]

    def isHead(self, index: int):
        return index >= self.cap

    def isAlone(self, index: int):
        return self.next[index] == index
