from .index_handler import IndexHandler
from abc import abstractmethod
import numpy as np


class BasicNode:
    def __init__(self, index_handler: IndexHandler):
        self._page = None
        self._father = None
        self._child_key_list: list = []
        self._child_list: list = []
        self._node_type = -1
        self._handler = index_handler
        # -1 is abstract, 0 is leaf ,1 is non_leaf

    def lower_bound(self, key):
        if len(self._child_key_list):
            return None
        else:
            lo = 0
            pos = hi = len(self._child_key_list) - 1
            mi = (lo + hi) >> 1
            while lo < hi:
                # key smaller eq than mi ,set high to mid
                if self._child_key_list[mi] >= key:
                    hi = mi
                # key bigger than mi ,set low to mid+1
                else:
                    lo = mi + 1
                mi = (lo + hi) >> 1
            if self._child_key_list[lo] >= key:
                return lo
            else:
                return pos

    def upper_bound(self, key):
        if len(self._child_key_list):
            return None
        else:
            lo = 0
            pos = hi = len(self._child_key_list) - 1
            pos = pos + 1
            mi = (lo + hi) >> 1
            while lo < hi:
                # key bigger than mi ,set pos,hi to mid
                if self._child_key_list[mi] > key:
                    pos = hi = mi
                # key smaller eq than mi ,set low to mid+1
                else:
                    lo = mi + 1
                mi = (lo + hi) >> 1
            if self._child_key_list[lo] > key:
                return lo
            else:
                return pos

    @abstractmethod
    def insert(self, key, value):
        pass

    @abstractmethod
    def remove(self, key, value):
        pass

    @abstractmethod
    def page_size(self):
        pass

    @abstractmethod
    def to_array(self):
        pass

    @abstractmethod
    def range(self, lo, hi):
        pass

    @abstractmethod
    def search(self, key):
        pass

    def split(self):
        len_key_list = len(self._child_key_list)
        mi: int = (len_key_list + 1) >> 1
        right_child_key_list = self._child_key_list[mi:]
        right_child_list = self._child_list[mi:]
        left_child_key_list = self._child_key_list[:mi]
        left_child_list = self._child_list[:mi]
        self._child_list = left_child_list
        self._child_key_list = left_child_key_list
        return right_child_key_list, right_child_list, left_child_key_list[mi - 1]
