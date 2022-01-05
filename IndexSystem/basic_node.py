from .index_handler import IndexHandler


class BasicNode:
    def __init__(self, index_handler: IndexHandler):
        self._child_key_list: list = []
        self._node_type = -1
        self._handler = index_handler
        # -1 is abstract, 0 is leaf ,1 is non_leaf_1 ,2 is non_leaf_2

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
