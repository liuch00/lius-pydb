from .basic_node import BasicNode
from .leaf_node import LeafNode
from .index_handler import IndexHandler
from ..RecordSystem.rid import RID
from ..FileSystem import macro
import numpy as np


class NoneLeafNode(BasicNode):
    def __init__(self, page, father, child_key_list, child_list, index_handler: IndexHandler):
        super(NoneLeafNode, self).__init__(index_handler)
        self._node_type = 1

        self._child_key_list = child_key_list
        self._child_list = child_list

        self._page = page
        self._father = father

    def insert(self, key, value):
        cursor = self.lower_bound(key)
        if cursor is not None:
            node: BasicNode = self._child_list[cursor]

        else:
            new_page = self._handler.new_page()
            node = LeafNode(new_page, self._page, 0, 0, [], [], self._handler)
            self._child_val.append(node)
            self._child_key_list.append(key)
            node.insert(key, value)
            return None

    def remove(self, key, rid: RID):
        # lower = self.lower_bound(key)
        # cursor = upper = self.upper_bound(key)
        # len_key_list = len(self._child_key_list)
        # if upper < len_key_list:
        #     upper = upper + 1
        #     cursor = cursor + 1
        # for index in range(lower, upper):
        #     if self._child_rid_list == rid:
        #         cursor = index
        #         break
        # if cursor != upper:
        #     self._child_key_list.pop(cursor)
        #     self._child_rid_list.pop(cursor)
        #     if len_key_list > 0:
        #         if cursor == 0:
        #             return self._child_rid_list[0]
        # else:
        #     return None
        pass

    def page_size(self) -> int:
        # todo:modify
        len_key_list: int = len(self._child_key_list)
        res = 48 + 16 * len_key_list
        return res

    def to_array(self) -> np.ndarray:
        # todo:modify
        num: int = int(macro.PAGE_SIZE >> 3)
        array = np.zeros(num, np.int64)
        len_key_list = len(self._child_key_list)
        array[0] = [1]
        array[1] = [self._father]
        array[2] = [len_key_list]
        for i in range(len_key_list):
            array[2 * i + 3] = [self._child_key_list[i]]
            node: BasicNode = self._child_leaf_list[i]
            array[2 * i + 4] = [node._page]
        array.dtype = np.uint8
        return array

    def search(self, key):
        index = self.lower_bound(key=key)
        len_child_list = len(self._child_list)
        if len_child_list == index:
            index = index - 1
        # search in child
        return self._child_list[index].search(key)

    def range(self, lo, hi):
        res = []
        lower = self.lower_bound(key=lo)
        upper = self.upper_bound(key=hi)
        len_child_list = len(self._child_list)
        if lower is None:
            return res
        if upper is not None:
            if upper + 1 < len_child_list:
                upper = upper + 1
        for index in range(lower, upper):
            node = self._child_list[index]
            node_range = node.range(lo=lower, hi=upper)
            if node_range is not None:
                res = res + node_range
        return res
