from .basic_node import BasicNode
from .index_handler import IndexHandler
from ..RecordSystem.rid import RID
from  ..FileSystem import macro
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
        len_key_list = len(self._child_key_list)
        return 32 + len_key_list * (8 + 16) + 32

    def to_array(self) -> np.ndarray:
        # todo:modify
        len_key_list = len(self._child_key_list)
        arr = np.zeros(int(macro.PAGE_SIZE / 8), np.int64)
        arr[0:5] = [1, self._father, self._left, self._right, len_key_list]
        for i in range(len_key_list):
            rid: RID = self._child_rid_list[i]
            arr[5 + 3 * i: 8 + 3 * i] = [self._child_key_list[i], rid.page, rid.slot]
        arr.dtype = np.uint8
        assert arr.size == macro.PAGE_SIZE
        return arr
