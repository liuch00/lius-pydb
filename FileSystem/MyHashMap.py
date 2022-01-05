
import numpy as np

from .MyLinkList import MyLinkList


class MyHashMap:

    def __init__(self, c, m):
        self.A = 1
        self.B = 1
        self.cap = c
        self.mod = m
        self.list = MyLinkList(c, m)
        self.nodes = np.full((2, c), (-1, -1))

    def hash(self, k1: int, k2: int):
        return (k1 * self.A + k2 * self.B) % self.mod

    def findIndex(self, ):