from .basic_node import BasicNode
from .index_handler import IndexHandler
from ..RecordSystem.rid import RID
from ..FileSystem import macro
import numpy as np


class LeafNode(BasicNode):
    def __init__(self, page, father, left, right, child_key_list, child_rid_list, index_handler: IndexHandler):
        super(LeafNode, self).__init__(index_handler)
        self._child_rid_list = child_rid_list
        self._node_type = 0

        self._page = page
        self._father = father
        self._left = left
        self._right = right
        self._child_key_list = child_key_list

    def insert(self, key, rid: RID):
        upper = self.upper_bound(key)
        if upper is None:
            self._child_key_list.insert(0, key)
            self._child_rid_list.insert(0, rid)
        else:
            self._child_key_list.insert(upper, key)
            self._child_rid_list.insert(upper, rid)
        return None

    def remove(self, key, rid: RID):
        lower = self.lower_bound(key)
        cursor = upper = self.upper_bound(key)
        len_key_list = len(self._child_key_list)
        if upper < len_key_list:
            upper = upper + 1
            cursor = cursor + 1
        for index in range(lower, upper):
            if self._child_rid_list == rid:
                cursor = index
                break
        if cursor != upper:
            self._child_key_list.pop(cursor)
            self._child_rid_list.pop(cursor)
            if len_key_list > 0:
                if cursor == 0:
                    return self._child_rid_list[0]
        else:
            return None

    def page_size(self) -> int:
        # todo:modify
        len_key_list: int = len(self._child_key_list)
        res = 64 + 24 * len_key_list
        return res

    def to_array(self) -> np.ndarray:
        # todo:modify
        num: int = int(macro.PAGE_SIZE >> 3)
        array = np.zeros(num, np.int64)
        array[0] = [1]
        array[1] = [self._father]
        array[2] = [self._left]
        array[3] = [self._right]
        len_key_list = len(self._child_key_list)
        array[4] = [len_key_list]
        for i in range(len_key_list):
            rid: RID = self._child_rid_list[i]
            array[3 * i + 5] = [self._child_key_list[i]]
            array[3 * i + 6] = [rid.page]
            array[3 * i + 7] = [rid.slot]
        array.dtype = np.uint8
        return array

    def search(self, key):
        index = self.lower_bound(key)
        len_key_list = len(self._child_key_list)
        if len_key_list == 0:
            return None
        else:
            if self._child_key_list[index] == key:
                return self._child_rid_list[index]
            else:
                return None

    def range(self, lo, hi):
        lower = self.lower_bound(lo)
        upper = self.upper_bound(hi)
        if lower > upper:
            return None
        else:
            return self._child_rid_list[lower:upper]
