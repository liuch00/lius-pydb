from .basic_node import BasicNode
from .leaf_node import LeafNode
from .index_handler import IndexHandler
from FileSystem import macro
# from ..FileSystem import macro
import numpy as np


class NoneLeafNode(BasicNode):
    def __init__(self, page, father, child_key_list, child_list, index_handler: IndexHandler):
        super(NoneLeafNode, self).__init__(index_handler)
        self._node_type = 0

        self._child_key_list = child_key_list
        self._child_list = child_list

        self._page = page
        self._father = father

    def insert(self, key, value):
        cursor = self.lower_bound(key)
        # cursor_new = self.upper_bound(key)
        if cursor is not None:
            if key > self._child_key_list[cursor]:
                self._child_key_list[cursor] = key
            node: BasicNode = self._child_list[cursor]
            node.insert(key=key, value=value)
            node_page_size = node.page_size()
            if node_page_size <= macro.PAGE_SIZE:
                return None
            else:
                right_child_key_list, right_child_list, origin_mi_key = node.split()
                old_key = self._child_key_list[cursor]
                self._child_key_list[cursor] = origin_mi_key
                cursor = cursor + 1
                self._child_key_list.insert(cursor, old_key)
                new_page_id = self._handler.new_page()
                if node._node_type == 0:
                    new_node = NoneLeafNode(page=new_page_id, father=self._page, child_key_list=right_child_key_list,
                                            child_list=right_child_list, index_handler=self._handler)
                    self._child_list.insert(cursor, new_node)
                elif node._node_type == 1:
                    new_node = LeafNode(page=new_page_id, father=self._page, left=node._page, right=new_page_id,
                                        child_key_list=right_child_key_list,
                                        child_list=right_child_list, index_handler=self._handler)
                    self._child_list.insert(cursor, new_node)
                else:
                    raise ValueError('node_type error!')
                return None
        else:
            new_page = self._handler.new_page()
            node = LeafNode(page=new_page, father=self._page, left=0, right=0, child_key_list=[], child_list=[],
                            index_handler=self._handler)
            self._child_list.append(node)
            self._child_key_list.append(key)
            node.insert(key=key, value=value)
            return None

    def remove(self, key, value):
        lower = self.lower_bound(key=key)
        upper = self.upper_bound(key=key)
        delta: int = 0
        res = None
        len_key_list = len(self._child_key_list)
        if upper < len_key_list:
            upper = upper + 1
        for index in range(lower, upper):
            index = index - delta
            node: BasicNode = self._child_list[index]
            temp = node.remove(key=key, value=value)
            if temp is not None:
                self._child_key_list[index] = temp
                if index == 0:
                    res = temp
                else:
                    res = res
            len_node_key_list = len(node._child_key_list)
            if len_node_key_list != 0:
                res = res
            else:
                delta = delta + 1
                self._child_key_list.pop(index)
                self._child_list.pop(index)
                if index == 0:
                    len_key_list = len(self._child_key_list)
                    if len_key_list > 0:
                        res = self._child_key_list[0]
        return res

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
        array[0] = 0
        array[1] = self._father
        array[2] = len_key_list
        for i in range(len_key_list):
            array[2 * i + 3] = self._child_key_list[i]
            node: BasicNode = self._child_list[i]
            array[2 * i + 4] = node._page
        array.dtype = np.uint8
        return array

    def search(self, key):
        index = self.lower_bound(key=key)
        len_child_list = len(self._child_list)
        if len_child_list == index:
            index = index - 1
        # search in child
        return self._child_list[index].search(key=key)

    def range(self, lo, hi):
        res = []
        lower = 0
        upper = self.upper_bound(key=hi)
        if lower is None:
            return res
        else:
            len_child_key_list = len(self._child_key_list)
            if upper is not None:
                if upper + 1 < len_child_key_list:
                    upper = upper + 1
            for index in range(lower, upper):
                node = self._child_list[index]
                node_range = node.range(lo=lo, hi=hi)
                if node_range is not None:
                    res = res + node_range
            return res
