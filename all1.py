
class MyException(Exception):
    pass

class FailCreateError(MyException):
    pass

class FailOpenError(MyException):
    pass

class FailReadPageError(MyException):
    pass

class RecordTooLong(MyException):
    pass

class ColumnAlreadyExist(MyException):
    pass

class ColumnNotExist(MyException):
    pass

class ValueNumError(MyException):
    pass

class VarcharTooLong(MyException):
    pass

class ValueTypeError(MyException):
    pass

class TableAlreadyExist(MyException):
    pass

class TableNotExist(MyException):
    pass

class IndexAlreadyExist(MyException):
    pass

class IndexNotExist(MyException):
    pass

class DatabaseAlreadyExist(MyException):
    pass

class DatabaseNotExist(MyException):
    pass

class NoDatabaseInUse(MyException):
    pass

class CheckAnyUniqueError(MyException):
    pass

class DuplicatedPrimaryKeyError(MyException):
    pass

class DuplicatedUniqueKeyError(MyException):
    pass

class MissForeignKeyError(MyException):
    pass

class SameNameError(MyException):
    pass

class SelectError(MyException):
    pass

class DateValueError(MyException):
    pass

class JoinError(MyException):
    pass

class AddForeignError(MyException):
    pass

class RemoveError(MyException):
    pass

class AddError(MyException):
    passimport numpy as np

from .macro import *
from .FileManager import FileManager
from .FindReplace import FindReplace


class BufManager:
    def __init__(self, fm: FileManager):
        self.FM = fm
        self.c = CAP
        self.m = MOD
        self.last = -1
        self.dirty = np.zeros(CAP, dtype=np.bool)
        self.addr = np.zeros((CAP, PAGE_SIZE), dtype=np.uint8)
        self.replace = FindReplace(CAP)
        self.index2FPID = np.zeros(CAP, dtype=np.int64)
        for i in range(CAP):
            self.index2FPID[i] = -1
        self.FPID2index = {}
        self.index_in_file = {}

    def combine_FPID(self, fileID, pageID):
        return fileID | (pageID << 16)

    def split_FPID(self, FPID):
        """return fileID, pageID"""
        return FPID & ((1 << 16) - 1), FPID >> 16

    def access(self, index):
        if index == self.last:
            return

        self.last = index
        self.replace.access(index)
        return

    def markDirty(self, index):
        self.dirty[index] = True
        self.access(index)
        return

    def release(self, index):
        self.dirty[index] = False
        self.replace.free(index)
        self.index2FPID[index] = -1
        fpID = self.index2FPID[index]
        self.FPID2index.pop(fpID)
        fID = self.split_FPID(fpID)[0]
        self.index_in_file[fID].remove(index)
        return

    def writeBack(self, index):
        if self.dirty[index]:
            fpID = self.index2FPID[index]
            fID = self.split_FPID(fpID)[0]
            pID = self.split_FPID(fpID)[1]
            self.FM.writePage(fID, pID, self.addr[index])
            self.dirty[index] = False
        self.replace.free(index)
        fpID = self.index2FPID[index]
        self.index2FPID[index] = -1
        self.FPID2index.pop(fpID)
        fID = self.split_FPID(fpID)[0]
        self.index_in_file[fID].remove(index)
        return

    def openFile(self, name: str):
        fID = self.FM.openFile(name)
        self.index_in_file[fID] = set()
        return fID

    def closeFile(self, fileID: int):
        for index in self.index_in_file.pop(fileID, {}):
            self.replace.free(index)

            fpID = self.index2FPID[index]
            self.index2FPID[index] = -1
            self.FPID2index.pop(fpID)
            if self.dirty[index]:
                # fID, pID = self.split_FPID(fpID)
                self.FM.writePage(*(self.split_FPID(fpID)), self.addr[index])
                self.dirty[index] = False
        self.FM.closeFile(fileID)
        return

    def fetchPage(self, fileID: int, pageID: int, buf: np.ndarray):
        fpID = self.combine_FPID(fileID, pageID)
        index = self.FPID2index.get(fpID)
        fID = self.split_FPID(fpID)[0]
        pID = self.split_FPID(fpID)[1]
        if index is None:
            self.getPage(fID, pID)
            index = self.FPID2index[fpID]
        self.dirty[index] = True
        self.addr[index] = buf
        self.replace.access(index)
        return

    def getPage(self, fileID: int, pageID: int):
        fpID = self.combine_FPID(fileID, pageID)
        index = self.FPID2index.get(fpID)
        if index is None:
            index = self.replace.find()
            foundFP = self.index2FPID[index]
            if foundFP == -1:
                self.FPID2index[fpID] = index
                self.index2FPID[index] = fpID
                fID = self.split_FPID(fpID)[0]
                pID = self.split_FPID(fpID)[1]
                buf = self.FM.readPage(fID, pID)
                buf = np.frombuffer(buf, np.uint8, PAGE_SIZE)
                self.index_in_file[fileID].add(index)
                self.addr[index] = buf
            else:
                self.writeBack(index)
        else:
            self.access(index)

        return self.addr[index].copy()

    def shutdown(self):
        for i in np.where(self.dirty)[0]:
            self.writeBack(i)
        # no need to clear dirty
        self.addr = np.zeros((CAP, PAGE_SIZE), dtype=np.uint8)
        self.index2FPID = np.zeros(CAP)
        for i in range(CAP):
            self.index2FPID[i] = -1
        self.FPID2index = {}
        self.last = -1
        while self.index_in_file:
            fID = self.index_in_file.popitem()[0]
            self.closeFile(fID)

    def createFile(self, name: str):
        self.FM.createFile(name)
        return

    def fileExist(self, name: str):
        return self.FM.fileExist(name)

    def destroyFile(self, name: str):
        return self.FM.destroyFile(name)

    def renameFile(self, src: str, dst: str):
        self.FM.renameFile(src, dst)
        return

    def writePage(self, fileID: int, pageID: int, buf: np.ndarray):
        self.FM.writePage(fileID, pageID, buf)
        return

    def readPage(self, fileID: int, pageID: int):
        return self.FM.readPage(fileID, pageID)

    def newPage(self, fileID: int, buf: np.ndarray):
        return self.FM.newPage(fileID, buf)

LEAF_BIT = 32
MAX_LEVEL = 5
MAX_INNER_NUM = 67
BIAS = 5

class MyBitMap:
    def __init__(self, cap, k):
        self.size = (cap >> BIAS)

    def getMask(self, k):
        s = 0


from .MyLinkList import MyLinkList


class FindReplace:
    def __init__(self, cap: int):
        self.CAP = cap
        self.list = MyLinkList(cap, 1)
        for i in range(cap - 1, 0, -1):
            self.list.insertFirst(0, i)

    def find(self):
        index = self.list.getFirst(0)
        self.list.delete(index)
        self.list.insert(0, index)
        return index

    def access(self, index: int):
        self.list.insert(0, index)

    def free(self, index: int):
        self.list.insertFirst(0, index)
from .macro import *
from Exceptions.exception import *

import numpy as np
import os


class FileManager:

    def __init__(self):
        self._fd = np.zeros(MAX_FILE_NUM)

    def createFile(self, name: str):
        f = open(name, 'w')
        if f is None:
            print("OH NO")
            raise FailCreateError("fail to create " + name)
        f.close()
        return

    def destroyFile(self, name: str):
        os.remove(name)
        return

    def fileExist(self, name: str):
        if os.path.exists(name):
            return True
        return False

    def renameFile(self, src: str, dst: str):
        os.rename(src, dst)
        return

    def openFile(self, name: str):
        fileID = os.open(name, os.O_RDWR)
        if fileID == -1:
            print("OH NO")
            raise FailOpenError("fail to open " + name)
        return fileID

    def closeFile(self, fileID: int):
        os.close(fileID)
        return

    def writePage(self, fileID: int, pageID: int, buf: np.ndarray):
        offset = pageID
        offset = offset << PAGE_SIZE_IDX
        error = os.lseek(fileID, offset, os.SEEK_SET)
        os.write(fileID, buf.tobytes())
        return

    def readPage(self, fileID: int, pageID: int):
        offset = pageID
        offset = offset << PAGE_SIZE_IDX
        error = os.lseek(fileID, offset, os.SEEK_SET)
        error = os.read(fileID, PAGE_SIZE)
        if error is None:
            print("OH NO")
            raise FailReadPageError("fail to read pid: " + str(pageID) + ", fid: " + str(fileID))
        return error

    def newPage(self, fileID: int, buf: np.ndarray):
        offset = os.lseek(fileID, 0, os.SEEK_END)
        os.write(fileID, buf.tobytes())
        pID = offset >> PAGE_SIZE_IDX
        return pID
PAGE_SIZE = 8192

PAGE_INT_NUM = 2048

PAGE_SIZE_IDX = 13
MAX_FMT_INT_NUM = 128
BUF_PAGE_NUM = 65536
MAX_FILE_NUM = 128
MAX_TYPE_NUM = 256

CAP = 60000

MOD = 60000
IN_DEBUG = 0
DEBUG_DELETE = 0
DEBUG_ERASE = 1
DEBUG_NEXT = 1

MAX_COL_NUM = 31

MAX_TB_NUM = 31
RELEASE = 1

# file name:
INDEX_NAME = '.id'
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
from .leaf_node import LeafNode
from .nonleaf_node import NoneLeafNode
from .index_handler import IndexHandler
from RecordSystem.rid import RID
# from ..RecordSystem.rid import RID
import numpy as np
from RecordSystem import macro


# from ..RecordSystem import macro


class FileIndex:
    def __init__(self, index_handler: IndexHandler, root_id):
        self._root = root_id
        self._handler = index_handler
        self._root_node = NoneLeafNode(page=root_id, father=root_id, child_key_list=[], child_list=[],
                                       index_handler=self._handler)
        self._is_modified = False

    @property
    def is_modified(self):
        return self._is_modified

    def insert(self, key, value: RID):
        self._is_modified = True
        self._root_node.insert(key=key, value=value)
        node_page_size = self._root_node.page_size()
        if node_page_size <= macro.PAGE_SIZE:
            return None
        else:
            new_page_id_father = self._handler.new_page()
            new_root_father = NoneLeafNode(page=new_page_id_father, father=new_page_id_father, child_key_list=[],
                                           child_list=[],
                                           index_handler=self._handler)
            self._root_node._father = new_page_id_father
            right_key_index = len(self._root_node.child_key_list) - 1
            right_key = self._root_node.child_key_list[right_key_index]
            right_child_key_list, right_child_list, origin_mi_key = self._root_node.split()
            new_root_son = NoneLeafNode(page=self._handler.new_page(), father=new_page_id_father,
                                        child_key_list=right_child_key_list, child_list=right_child_list,
                                        index_handler=self._handler)
            old_root_node = self._root_node
            self._root_node = new_root_father
            self._root = new_page_id_father
            father_child_key_list = [origin_mi_key, right_key]
            father_child_node_list = [old_root_node, new_root_son]
            self._root_node._child_key_list = father_child_key_list
            self._root_node._child_list = father_child_node_list

    def delete(self, key, value: RID):
        self._is_modified = True
        self._root_node.remove(key=key, value=value)
        return None

    def search(self, key):
        return self._root_node.search(key=key)

    def range(self, lo, hi):
        return self._root_node.range(lo=lo, hi=hi)

    def pour(self):
        temp_node_list = []
        temp_node = None
        temp_node_list.append(self._root_node)
        while len(temp_node_list) > 0:
            temp_node = temp_node_list.pop(0)
            page_id = temp_node.page
            data = temp_node.to_array()
            if isinstance(temp_node, NoneLeafNode):
                for item in temp_node.child_list:
                    temp_node_list.append(item)
            self._handler.put_page(page_id=page_id, data=data)
        return None

    def build_node(self, page_id):
        self._is_modified = True
        page_data: np.ndarray = self._handler.get_page(page_id=page_id)
        page_data.dtype = np.int64
        node_type = page_data[0]
        if node_type == 0:
            res: NoneLeafNode = self._build_node_type_0(page_id=page_id, data=page_data)
        elif node_type == 1:
            res: LeafNode = self._build_node_type_1(page_id=page_id, data=page_data)
        else:
            raise ValueError('node_type error!')
        return res

    def _build_node_type_1(self, page_id, data: np.ndarray):
        data.dtype = np.int64
        child_num = data[4]
        child_key_list = []
        child_rid_list = []
        for i in range(child_num):
            child_key_list.append(data[5 + 3 * i])
            rid = RID(int(data[6 + 3 * i]), int(data[7 + 3 * i]))
            child_rid_list.append(rid)
        leaf_node = LeafNode(page=page_id, father=data[1], left=data[2], right=data[3], child_key_list=child_key_list,
                             child_list=child_rid_list, index_handler=self._handler)
        return leaf_node

    def _build_node_type_0(self, page_id, data: np.ndarray):
        data.dtype = np.int64
        child_num = data[2]
        child_key_list = []
        child_node_list = []
        for i in range(child_num):
            child_key_list.append(data[3 + 2 * i])
            child_node_list.append(self.build_node(data[4 + 2 * i]))
        nonleaf_node = NoneLeafNode(page=page_id, father=data[1], child_key_list=child_key_list,
                                    child_list=child_node_list, index_handler=self._handler)
        return nonleaf_node

    def take(self):
        page_data: np.ndarray = self._handler.get_page(page_id=self._root)
        page_data.dtype = np.int64
        assert (page_data[1] == self._root), 'page take error!'
        self._root_node = self.build_node(page_id=self._root)
        return None

    @property
    def handler(self):
        return self._handler

    @property
    def root(self):
        return self._root
class FileIndexID:
    def __init__(self, table_name, file_index_root_id):
        self._table_name = table_name
        self._file_index_root_id = file_index_root_id

    @property
    def table_name(self):
        return self._table_name

    @property
    def file_index_root_id(self):
        return self._file_index_root_id

    def __str__(self):
        return f'{{table_name: {self.table_name}, file_index_root_id: {self.file_index_root_id}}}'

    def __eq__(self, other):
        return self._file_index_root_id == other.file_index_root_id and self._table_name == other.table_name

    def __hash__(self):
        return hash((self._table_name, self._file_index_root_id))
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
        # -1 is abstract, 1 is leaf ,0 is non_leaf

    def lower_bound(self, key):
        if len(self._child_key_list):
            lo = 0
            pos = hi = len(self._child_key_list) - 1
            # todo:modified
            while lo < hi:
                mi = (lo + hi) >> 1
                # key smaller eq than mi ,set high to mid
                if self._child_key_list[mi] >= key:
                    hi = mi
                # key bigger than mi ,set low to mid+1
                else:
                    lo = mi + 1
            if self._child_key_list[lo] >= key:
                return lo
            else:
                return pos
        else:
            return None

    def upper_bound(self, key):
        if len(self._child_key_list):
            lo = 0
            pos = hi = len(self._child_key_list) - 1
            pos = pos + 1
            while lo < hi:
                mi = (lo + hi) >> 1
                # key bigger than mi ,set pos,hi to mid
                if self._child_key_list[mi] > key:
                    pos = hi = mi
                # key smaller eq than mi ,set low to mid+1
                else:
                    lo = mi + 1
            if self._child_key_list[lo] > key:
                return lo
            else:
                return pos
        else:
            return 0


    @abstractmethod
    def insert(self, key, value):
        raise NotImplemented

    @abstractmethod
    def remove(self, key, value):
        raise NotImplemented

    @abstractmethod
    def child_list(self):
        raise NotImplemented

    @property
    def page(self):
        res = self._page
        return res

    @property
    def child_key_list(self):
        return self._child_key_list

    @property
    def child_list(self):
        return self._child_list

    @abstractmethod
    def page_size(self):
        raise NotImplemented

    @abstractmethod
    def to_array(self):
        raise NotImplemented

    @abstractmethod
    def range(self, lo, hi):
        raise NotImplemented

    @abstractmethod
    def search(self, key):
        raise NotImplemented

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
import numpy as np

from FileSystem.BufManager import BufManager
from FileSystem import macro


class IndexHandler:
    def __init__(self, buf_manager: BufManager, database_name, home_directory):
        self._manager = buf_manager
        index_file_name = database_name + macro.INDEX_NAME
        # index_file_path = home_directory + '/' + database_name + '/' + index_file_name
        index_file_path = home_directory / database_name / index_file_name
        if not self._manager.fileExist(index_file_path):
            self._manager.createFile(index_file_path)
        self._file_id = self._manager.openFile(index_file_path)
        self._is_modified = False

    def get_page(self, page_id):
        res: np.ndarray = self._manager.getPage(fileID=self._file_id, pageID=page_id)
        return res

    def put_page(self, page_id, data):
        self._is_modified = True
        self._manager.fetchPage(fileID=self._file_id, pageID=page_id, buf=data)
        return None

    def new_page(self):
        data = np.zeros(macro.PAGE_SIZE, dtype=np.uint8)
        page_id: int = self._manager.newPage(fileID=self._file_id, buf=data)
        return page_id

    def close_file(self):
        self._manager.closeFile(fileID=self._file_id)
        return None
from .index_handler import IndexHandler
from .file_index import FileIndex
from typing import Dict
from FileSystem.BufManager import BufManager
from .FileIndexID import FileIndexID


class IndexManager:
    def __init__(self, buf_manager: BufManager, home_directory: str = '/'):
        self._buf_manager = buf_manager
        self._home_directory = home_directory
        self._started_index_handler: Dict[str, IndexHandler] = {}
        self._started_file_index: Dict[FileIndexID, FileIndex] = {}

    def catch_handler(self, database_name):
        if database_name in self._started_index_handler:
            return self._started_index_handler[database_name]
        else:
            # not exist
            new_handler: IndexHandler = IndexHandler(buf_manager=self._buf_manager, database_name=database_name,
                                                     home_directory=self._home_directory)
            self._started_index_handler[database_name]: IndexHandler = new_handler
            return self._started_index_handler[database_name]

    def shut_handler(self, database_name):
        if database_name in self._started_index_handler:
            index_handler = self._started_index_handler.pop(database_name)
            for key, file_index in tuple(self._started_file_index.items()):
                if file_index.handler is not index_handler:
                    continue
                if (key._table_name, key._file_index_root_id) not in self._started_index_handler:
                    return None
                tmp_file_index = self._started_index_handler.pop((key._table_name, key._file_index_root_id))
                if tmp_file_index.is_modified:
                    tmp_file_index.pour()
            return True
            # for ID in self._started_file_index:
            #     file_index = self._started_file_index.get(ID)
            #     if file_index.handler is index_handler:
            #         # shut index
            #         if ID in self._started_file_index:
            #             tmp_file_index = self._started_file_index.pop(ID)
            #             if tmp_file_index.is_modified:
            #                 tmp_file_index.pour()
        else:
            return False

    def create_index(self, database_name, table_name):
        handler = self.catch_handler(database_name=database_name)
        root_id = handler.new_page()
        ID = FileIndexID(table_name=table_name, file_index_root_id=root_id)
        self._started_file_index[ID] = FileIndex(index_handler=handler, root_id=root_id)
        self._started_file_index[ID].pour()
        return self._started_file_index[ID]

    def start_index(self, database_name, table_name, root_id):
        ID = FileIndexID(table_name=table_name, file_index_root_id=root_id)
        if ID in self._started_file_index:
            file_index = self._started_file_index.get(ID)
            return file_index
        else:
            handler = self.catch_handler(database_name=database_name)
            file_index = FileIndex(index_handler=handler, root_id=root_id)
            # load data
            file_index.take()
            self._started_file_index[ID] = file_index
            return file_index

    def close_manager(self):
        for db_name in tuple(self._started_index_handler):
            self.shut_handler(database_name=db_name)
        return None
from .basic_node import BasicNode
from .index_handler import IndexHandler
from RecordSystem.rid import RID
# from ..RecordSystem.rid import RID
from FileSystem import macro
# from ..FileSystem import macro
import numpy as np


class LeafNode(BasicNode):
    def __init__(self, page, father, left, right, child_key_list, child_list, index_handler: IndexHandler):
        super(LeafNode, self).__init__(index_handler)
        self._node_type = 1

        self._child_key_list = child_key_list
        self._child_list = child_list

        self._page = page
        self._father = father
        self._left = left
        self._right = right

    def insert(self, key, value: RID):
        upper = self.upper_bound(key=key)
        if upper is None:
            self._child_key_list.insert(0, key)
            self._child_list.insert(0, value)
        else:
            self._child_key_list.insert(upper, key)
            self._child_list.insert(upper, value)
        return None

    def remove(self, key, value: RID):
        lower = self.lower_bound(key=key)
        cursor = upper = self.upper_bound(key=key)
        len_key_list = len(self._child_key_list)
        if upper < len_key_list:
            upper = upper + 1
            cursor = cursor + 1
        for index in range(lower, upper):
            if self._child_list[index] == value:
                cursor = index
                break
        if cursor != upper:
            self._child_key_list.pop(cursor)
            self._child_list.pop(cursor)
            len_key_list = len(self._child_key_list)
            if len_key_list > 0:
                if cursor == 0:
                    return self._child_key_list[0]
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
        array[0] = 1
        array[1] = self._father
        array[2] = self._left
        array[3] = self._right
        len_key_list = len(self._child_key_list)
        array[4] = len_key_list
        for i in range(len_key_list):
            rid: RID = self._child_list[i]
            array[3 * i + 5] = self._child_key_list[i]
            array[3 * i + 6] = rid.page
            array[3 * i + 7] = rid.slot
        array.dtype = np.uint8
        return array

    def search(self, key):
        index = self.lower_bound(key=key)
        len_key_list = len(self._child_key_list)
        if len_key_list == 0:
            return None
        else:
            if self._child_key_list[index] == key:
                return self._child_list[index]
            else:
                return None

    def range(self, lo, hi):
        lower = self.lower_bound(key=lo)
        upper = self.upper_bound(key=hi)
        if lower > upper:
            return None
        else:
            return self._child_list[lower:upper]

from .lookup_element import Term, LookupOutput
from Exceptions.exception import JoinError

class Join:
    def __init__(self, res_map: dict, term, union=None):
        self.res_map = res_map
        self._term = term
        self._join_map = {}
        self._union = union

    def create_pair(self, term: Term):
        if term.aim_table is not None:
            if term.table != term.aim_table:
                if term.operator != '=':
                    raise TypeError('Join create_pair error!')
                pairs = (term.table, term.col), (term.aim_table, term.aim_col)
                sorted_pairs = zip(*sorted(pairs))
                res = tuple(sorted_pairs)
                return res
            else:
                return None, None
        else:
            return None, None

    def union_create(self):
        for key, element in map(self.create_pair, self._term):
            if element is None:
                continue
            else:
                if key in self._join_map:
                    element_0 = element[0]
                    element_1 = element[1]
                    self._join_map[key][0].append(element_0)
                    self._join_map[key][1].append(element_1)
                else:
                    element_0 = [element[0]]
                    element_1 = [element[1]]
                    self._join_map[key] = (element_0, element_1)
        if not self._join_map:
            raise JoinError('Join tables errors!!!!')
        self._union = {key: key for key in self.res_map.keys()}

    def union_search(self, element):
        if element != self._union[element]:
            self._union[element] = self.union_search(self._union[element])
        return self._union[element]

    def union_merge(self, element_1, element_2):
        father_1 = self.union_search(element=element_1)
        father_2 = self.union_search(element=element_2)
        self._union[father_1] = father_2

    def get_output(self):
        res = None
        self.union_create()
        for each_pair in self._join_map:
            each_pair_0 = each_pair[0]
            outside: LookupOutput = self.res_map[each_pair_0]
            each_pair_1 = each_pair[1]
            inside: LookupOutput = self.res_map[each_pair_1]
            # outside_joined = tuple(each_pair_0 + ".")
            outside_joined = tuple(each_pair_0 + "." + col for col in self._join_map[each_pair][0])
            inside_joined = tuple(each_pair_1 + "." + col for col in self._join_map[each_pair][1])
            # for each_0 in self._join_map[each_pair][0]:
            #     outside_joined += tuple(each_0)
            # inside_joined = tuple(each_pair_1 + ".")
            # for each_1 in self._join_map[each_pair][1]:
            #     inside_joined += tuple(each_1)
            new_res = self.loop_join(outside, inside, outside_joined, inside_joined)
            self.union_merge(each_pair_0, each_pair_1)
            new_key = self.union_search(each_pair_0)
            self.res_map[new_key] = new_res
            res = new_res
        return res

    @staticmethod
    def get_values(value: tuple, block: tuple):
        # res = ()
        # for item in block:
        #     res = res + (value[item])
        res = tuple(value[i] for i in block)
        return res

    def create_join_value(self, outside: tuple, outside_joined: tuple):
        row_index = 0
        join_value_map = {}
        for item in outside:
            val = self.get_values(item, outside_joined)
            if val in join_value_map:
                join_value_map[val].append(row_index)
            else:
                join_value_map[val] = [row_index]
            row_index = row_index + 1
        return join_value_map

    @staticmethod
    def check_join(len_outside_joined, len_inside_joined):
        if len_outside_joined != len_inside_joined:
            raise ValueError("join error!")

    def loop_join_data(self, outside: tuple, inside: tuple, outside_joined: tuple, inside_joined: tuple):
        len_outside = len(outside)
        len_inside = len(inside)
        len_outside_joined = len(outside_joined)
        len_inside_joined = len(inside_joined)
        self.check_join(len_outside_joined=len_outside_joined, len_inside_joined=len_inside_joined)
        if len_outside == 0 or len_inside == 0:
            return None, None, None
        len_outside_0 = len(outside[0])
        len_inside_0 = len(inside[0])
        outside_left = tuple(i for i in range(len_outside_0) if i not in outside_joined)
        inside_left = tuple(i for i in range(len_inside_0) if i not in inside_joined)
        res = []
        join_value = self.create_join_value(outside=outside, outside_joined=outside_joined)
        for inside_index in range(len_inside):
            tmp_value = self.get_values(inside[inside_index], inside_joined)
            if tmp_value in join_value:
                in_tmp_value = self.get_values(inside[inside_index], inside_joined)
                outside_list = join_value[in_tmp_value]
                for outside_index in outside_list:
                    t1 = self.get_values(outside[outside_index], outside_left)
                    t2 = self.get_values(inside[inside_index], inside_left)
                    t3 = self.get_values(outside[outside_index], outside_joined)
                    res.append(t1 + t2 + t3)
        return res, outside_left, inside_left

    def loop_join(self, outside: LookupOutput, inside: LookupOutput, outside_joined: tuple, inside_joined: tuple):
        len_out_data = outside.size()
        len_in_data = inside.size()
        if len_out_data > len_in_data:
            # swap in out
            tmp = outside
            outside = inside
            inside = tmp
            tmp_joined = outside_joined
            outside_joined = inside_joined
            inside_joined = tmp_joined
        outside_data = outside.data
        outside_head = outside.headers
        outside_joined_id = tuple(outside.header_id(item) for item in outside_joined)
        inside_data = inside.data
        inside_head = inside.headers
        inside_joined_id = tuple(inside.header_id(item) for item in inside_joined)

        joined_data, outside_left, inside_left = self.loop_join_data(outside=outside_data, inside=inside_data,
                                                                     outside_joined=outside_joined_id,
                                                                     inside_joined=inside_joined_id)
        if joined_data is None:
            res = LookupOutput(headers=[], data=[])
        else:
            h1 = self.get_values(outside_head, outside_left)
            h2 = self.get_values(inside_head, inside_left)
            head = h1 + h2 + outside_joined
            res = LookupOutput(headers=head, data=joined_data)
        for outside_h, inside_h in zip(outside_joined, inside_joined):
            res.insert_alias(inside_h, outside_h)
        for alias in outside.alias_map:
            out_t = outside.alias_map[alias]
            res.insert_alias(alias, out_t)
        for alias in inside.alias_map:
            in_t = inside.alias_map[alias]
            res.insert_alias(alias, in_t)
        return res
from MetaSystem.info import TableInfo, ColumnInfo
from SQL_Parser.SQLVisitor import SQLVisitor
from SQL_Parser.SQLParser import SQLParser
from antlr4 import ParserRuleContext
import time
from .system_manager import SystemManger
from .lookup_element import Reducer, Term, LookupOutput


# todo:move to SQL_parser
class SystemVisitor(SQLVisitor):
    def __init__(self, system_manager=None):
        super(SQLVisitor, self).__init__()
        self.system_manager: SystemManger = system_manager
        self.time_begin = None

    @staticmethod
    def to_str(context):
        if isinstance(context, ParserRuleContext):
            context = context.getText()
            res = str(context)
            return res
        else:
            res = str(context)
            return res

    def to_int(self, context):
        str_context = self.to_str(context)
        int_context = int(str_context)
        return int_context

    def to_float(self, context):
        str_context = self.to_str(context)
        float_context = float(str_context)
        return float_context

    def spend_time(self):
        if self.time_begin is None:
            self.time_begin = time.time()
            return None
        else:
            time_end = time.time()
            time_begin = self.time_begin
            self.time_begin = time.time()
            return time_end - time_begin


    def aggregateResult(self, aggregate, next_result):
        return aggregate if next_result is None else next_result

    # Visit a parse tree produced by SQLParser#program.
    def visitProgram(self, ctx: SQLParser.ProgramContext):
        # todo:add
        res = []
        for item in ctx.statement():
            output: LookupOutput = item.accept(self)
            if output is not None:
                output._cost = self.spend_time()
                output.simplify()
                res.append(output)
        return res

    # Visit a parse tree produced by SQLParser#system_statement.
    def visitSystem_statement(self, ctx: SQLParser.System_statementContext):
        return LookupOutput('databases', tuple(self.system_manager.databaselist))

    # Visit a parse tree produced by SQLParser#create_db.
    def visitCreate_db(self, ctx: SQLParser.Create_dbContext):
        return self.system_manager.createDatabase(self.to_str(ctx.Identifier()))

    # Visit a parse tree produced by SQLParser#drop_db.
    def visitDrop_db(self, ctx: SQLParser.Drop_dbContext):
        return self.system_manager.removeDatabase(self.to_str(ctx.Identifier()))

    # Visit a parse tree produced by SQLParser#use_db.
    def visitUse_db(self, ctx: SQLParser.Use_dbContext):
        return self.system_manager.useDatabase(self.to_str(ctx.Identifier()))

    # Visit a parse tree produced by SQLParser#show_tables.
    def visitShow_tables(self, ctx: SQLParser.Show_tablesContext):
        return LookupOutput('tables', self.system_manager.displayTableNames())

    # Visit a parse tree produced by SQLParser#create_table.
    def visitCreate_table(self, ctx: SQLParser.Create_tableContext):
        # todo:fix
        columns, foreign_keys, primary = ctx.field_list().accept(self)
        table_name = self.to_str(ctx.Identifier())
        res = self.system_manager.createTable(TableInfo(table_name, columns))
        for col in foreign_keys:
            self.system_manager.addForeign(table_name, col, foreign_keys[col])
        self.system_manager.setPrimary(table_name, primary)
        return res

    # Visit a parse tree produced by SQLParser#drop_table.
    def visitDrop_table(self, ctx: SQLParser.Drop_tableContext):
        return self.system_manager.removeTable(self.to_str(ctx.Identifier()))

    # Visit a parse tree produced by SQLParser#describe_table.
    def visitDescribe_table(self, ctx: SQLParser.Describe_tableContext):
        return self.system_manager.descTable(self.to_str(ctx.Identifier()))

    # Visit a parse tree produced by SQLParser#insert_into_table.
    def visitInsert_into_table(self, ctx: SQLParser.Insert_into_tableContext):
        data = ctx.value_lists().accept(self)
        for item in data:
            self.system_manager.insertRecord(self.to_str(ctx.getChild(2)), item)
        return LookupOutput('inserted_items', (len(data),))

    # Visit a parse tree produced by SQLParser#delete_from_table.
    def visitDelete_from_table(self, ctx: SQLParser.Delete_from_tableContext):
        return self.system_manager.deleteRecords(self.to_str(ctx.Identifier()), ctx.where_and_clause().accept(self))

    # Visit a parse tree produced by SQLParser#update_table.
    def visitUpdate_table(self, ctx: SQLParser.Update_tableContext):
        return self.system_manager.updateRecords(self.to_str(ctx.Identifier()), ctx.where_and_clause().accept(self),
                                                 ctx.set_clause().accept(self))

    # Visit a parse tree produced by SQLParser#select_table.
    def visitSelect_table(self, ctx: SQLParser.Select_tableContext):
        term = ctx.where_and_clause().accept(self) if ctx.where_and_clause() else ()
        group_by = ctx.column().accept(self) if ctx.column() else (None, '')
        limit = self.to_int(ctx.Integer(0)) if ctx.Integer() else None
        offset = self.to_int(ctx.Integer(1)) if ctx.Integer(1) else 0
        return self.system_manager.selectRecordsLimit(ctx.selectors().accept(self), ctx.identifiers().accept(self),
                                                      term, group_by,
                                                      limit, offset)

    # Visit a parse tree produced by SQLParser#create_index.
    def visitCreate_index(self, ctx: SQLParser.Create_indexContext):
        for item_col in ctx.identifiers().accept(self):
            self.system_manager.createIndex(self.to_str(ctx.getChild(2)), self.to_str(ctx.getChild(4)), item_col)

    # Visit a parse tree produced by SQLParser#drop_index.
    def visitDrop_index(self, ctx: SQLParser.Drop_indexContext):
        return self.system_manager.removeIndex(self.to_str(ctx.Identifier()))

    # Visit a parse tree produced by SQLParser#alter_add_index.
    def visitAlter_add_index(self, ctx: SQLParser.Alter_add_indexContext):
        for item in ctx.identifiers().accept(self):
            self.system_manager.createIndex(self.to_str(ctx.Identifier(1)), self.to_str(ctx.Identifier(0)), item)

    # Visit a parse tree produced by SQLParser#alter_drop_index.
    def visitAlter_drop_index(self, ctx: SQLParser.Alter_drop_indexContext):
        return self.system_manager.removeIndex(self.to_str(ctx.Identifier(1)))

    # Visit a parse tree produced by SQLParser#alter_table_add.
    def visitAlter_table_add(self, ctx: SQLParser.Alter_table_addContext):
        col: ColumnInfo = ctx.field().accept(self)
        pri = isinstance(ctx.field(), SQLParser.Primary_key_fieldContext)
        foreign = ctx.field().getChild(0).getText() == 'FOREIGN'
        self.system_manager.addColumn(self.to_str(ctx.Identifier()), col, pri, foreign)

    # Visit a parse tree produced by SQLParser#alter_table_drop.
    def visitAlter_table_drop(self, ctx: SQLParser.Alter_table_dropContext):
        self.system_manager.removeColumn(self.to_str(ctx.Identifier(0)), self.to_str(ctx.Identifier(1)))

    # Visit a parse tree produced by SQLParser#alter_table_rename.
    def visitAlter_table_rename(self, ctx: SQLParser.Alter_table_renameContext):
        self.system_manager.renameTable(self.to_str(ctx.Identifier(0)), self.to_str(ctx.Identifier(1)))

    # Visit a parse tree produced by SQLParser#alter_table_drop_pk.
    def visitAlter_table_drop_pk(self, ctx: SQLParser.Alter_table_drop_pkContext):
        self.system_manager.removePrimary(self.to_str(ctx.Identifier(0)))

    # Visit a parse tree produced by SQLParser#alter_table_drop_foreign_key.
    def visitAlter_table_drop_foreign_key(self, ctx: SQLParser.Alter_table_drop_foreign_keyContext):
        self.system_manager.removeForeign(self.to_str(ctx.Identifier(0)), self.to_str(ctx.Identifier(1)), None)

        # self.system_manager.removeForeign(None, None, self.to_str(ctx.Identifier(1)))

    # Visit a parse tree produced by SQLParser#alter_table_add_pk.
    def visitAlter_table_add_pk(self, ctx: SQLParser.Alter_table_add_pkContext):
        self.system_manager.setPrimary(self.to_str(ctx.Identifier(0)), ctx.identifiers().accept(self))

    # Visit a parse tree produced by SQLParser#alter_table_add_foreign_key.
    def visitAlter_table_add_foreign_key(self, ctx: SQLParser.Alter_table_add_foreign_keyContext):
        for (item1, item2) in zip(ctx.identifiers(0).accept(self), ctx.identifiers(1).accept(self)):
            self.system_manager.addForeign(self.to_str(ctx.Identifier(0)), item1,
                                           (self.to_str(ctx.Identifier(2)), item2), self.to_str(ctx.Identifier(1)))

    # Visit a parse tree produced by SQLParser#alter_table_add_unique.
    def visitAlter_table_add_unique(self, ctx: SQLParser.Alter_table_add_uniqueContext):
        table, name, column = tuple(map(self.to_str, ctx.Identifier()))
        return self.system_manager.addUnique(table, column, name)

    # Visit a parse tree produced by SQLParser#field_list.
    def visitField_list(self, ctx: SQLParser.Field_listContext):
        name_to_column = {}
        foreign_keys = {}
        primary_key = None
        for field in ctx.field():
            if isinstance(field, SQLParser.Normal_fieldContext):
                name = self.to_str(field.Identifier())
                type_, size = field.type_().accept(self)
                null_permit = True
                if len(field.children) > 2 and 'NOT' in (field.getChild(1).getText(), field.getChild(2).getText()):
                    null_permit = False
                name_to_column[name] = ColumnInfo(type_, name, size, null_permit)
            elif isinstance(field, SQLParser.Foreign_key_fieldContext):
                field_name, table_name, refer_name = field.accept(self)
                if field_name in foreign_keys:
                    raise NameError(f'Foreign key named {field_name} is duplicated')
                foreign_keys[field_name] = table_name, refer_name
            else:
                assert isinstance(field, SQLParser.Primary_key_fieldContext)
                names = field.accept(self)
                for name in names:
                    if name not in name_to_column:
                        raise NameError(f'Unknown field {name} field list')
                if primary_key:
                    raise NameError('Only one primary key supported')
                primary_key = names
        return list(name_to_column.values()), foreign_keys, primary_key

    # Visit a parse tree produced by SQLParser#normal_field.
    def visitNormal_field(self, ctx: SQLParser.Normal_fieldContext):
        item1, item2 = ctx.type_().accept(self)
        return ColumnInfo(item1, self.to_str(ctx.Identifier()), item2)

    # Visit a parse tree produced by SQLParser#primary_key_field.
    def visitPrimary_key_field(self, ctx: SQLParser.Primary_key_fieldContext):
        return ctx.identifiers().accept(self)

    # Visit a parse tree produced by SQLParser#foreign_key_field.
    def visitForeign_key_field(self, ctx: SQLParser.Foreign_key_fieldContext):
        return tuple(self.to_str(item) for item in ctx.Identifier())

    # Visit a parse tree produced by SQLParser#type_.
    def visitType_(self, ctx: SQLParser.Type_Context):
        if ctx.Integer():
            size = self.to_int(ctx.Integer())
        else:
            size = 0
        return self.to_str(ctx.getChild(0)), size

    # Visit a parse tree produced by SQLParser#value_lists.
    def visitValue_lists(self, ctx: SQLParser.Value_listsContext):
        return tuple(item.accept(self) for item in ctx.value_list())

    # Visit a parse tree produced by SQLParser#value_list.
    def visitValue_list(self, ctx: SQLParser.Value_listContext):
        return tuple(item.accept(self) for item in ctx.value())

    # Visit a parse tree produced by SQLParser#value.
    def visitValue(self, ctx: SQLParser.ValueContext):
        if ctx.Integer():
            return self.to_int(ctx)
        if ctx.Float():
            return self.to_float(ctx)
        if ctx.String():
            return self.to_str(ctx)[1:-1]
        if ctx.Null():
            return None

    # Visit a parse tree produced by SQLParser#where_and_clause.
    def visitWhere_and_clause(self, ctx: SQLParser.Where_and_clauseContext):
        return tuple(item.accept(self) for item in ctx.where_clause())

    # Visit a parse tree produced by SQLParser#where_operator_expression.
    def visitWhere_operator_expression(self, ctx: SQLParser.Where_operator_expressionContext):
        operator = self.to_str(ctx.operator())
        table_name, col_name = ctx.column().accept(self)
        value = ctx.expression().accept(self)
        if isinstance(value, tuple):
            return Term(1, table_name, col_name, operator,
                        aim_table_name=value[0], aim_col=value[1])
        else:
            return Term(1, table_name, col_name, operator, value=value)

    # Visit a parse tree produced by SQLParser#where_operator_select.
    def visitWhere_operator_select(self, ctx: SQLParser.Where_operator_selectContext):
        table_name, column_name = ctx.column().accept(self)
        operator = self.to_str(ctx.operator())
        result: LookupOutput = ctx.select_table().accept(self)
        value = self.system_manager.resultToValue(result=result, is_in=False)
        return Term(1, table_name, column_name, operator, value=value)

    # Visit a parse tree produced by SQLParser#where_null.
    def visitWhere_null(self, ctx: SQLParser.Where_nullContext):
        table_name = ctx.parentCtx.parentCtx.identifiers().accept(self)[0]
        _, col_name = ctx.column().accept(self)
        is_null = ctx.getChild(2).getText() != "NOT"
        return Term(0, table_name, col_name, value=is_null)

    # Visit a parse tree produced by SQLParser#where_in_list.
    def visitWhere_in_list(self, ctx: SQLParser.Where_in_listContext):
        table_name, col_name = ctx.column().accept(self)
        value_list = ctx.value_list().accept(self)
        return Term(2, table_name, col_name, value=value_list)

    # Visit a parse tree produced by SQLParser#where_in_select.
    def visitWhere_in_select(self, ctx: SQLParser.Where_in_selectContext):
        table_name, col_name = ctx.column().accept(self)
        res: LookupOutput = ctx.select_table().accept(self)
        value = self.system_manager.resultToValue(res, True)
        return Term(2, table_name, col_name, value=value)

    # Visit a parse tree produced by SQLParser#where_like_string.
    def visitWhere_like_string(self, ctx: SQLParser.Where_like_stringContext):
        table_name, col_name = ctx.column().accept(self)
        return Term(3, table_name, col_name, value=self.to_str(ctx.String())[1:-1])

    # Visit a parse tree produced by SQLParser#column.
    def visitColumn(self, ctx: SQLParser.ColumnContext):
        if len(ctx.Identifier()) != 1:
            return self.to_str(ctx.Identifier(0)), self.to_str(ctx.Identifier(1))
        else:
            return None, self.to_str(ctx.Identifier(0))

    # Visit a parse tree produced by SQLParser#set_clause.
    def visitSet_clause(self, ctx: SQLParser.Set_clauseContext):
        tmp_map = {}
        for identifier, value in zip(ctx.Identifier(), ctx.value()):
            tmp_map[self.to_str(identifier)] = value.accept(self)
        return tmp_map

    # Visit a parse tree produced by SQLParser#selectors.
    def visitSelectors(self, ctx: SQLParser.SelectorsContext):
        if self.to_str(ctx.getChild(0)) == '*':
            return Reducer(0, '*', '*'),
        return tuple(item.accept(self) for item in ctx.selector())

    # Visit a parse tree produced by SQLParser#selector.
    def visitSelector(self, ctx: SQLParser.SelectorContext):
        if ctx.Count():
            return Reducer(3, '*', '*')
        table_name, column_name = ctx.column().accept(self)
        if ctx.aggregator():
            return Reducer(2, table_name, column_name, self.to_str(ctx.aggregator()))
        return Reducer(1, table_name, column_name)

    # Visit a parse tree produced by SQLParser#identifiers.
    def visitIdentifiers(self, ctx: SQLParser.IdentifiersContext):
        return tuple(self.to_str(item) for item in ctx.Identifier())

import csv
from prettytable import PrettyTable
from sys import stderr, stdout
from typing import List
from .lookup_element import LookupOutput
from datetime import timedelta

class TablePrinter:
    def __init__(self):
        self.inUse = None

    def myprint(self, result: LookupOutput):
        table = self.MyPT()
        table.field_names = result.headers
        table.add_rows(result.data)
        if not len(result.data):
            print("Empty set in " + f'{(timedelta(result.cost).total_seconds() / 10 ** 5):.3f}' + "s")
        else:
            print(table.get_string())
            print(f'{len(result.data)}' + ' results in ' + f'{(timedelta(result.cost).total_seconds() / 10 ** 5):.3f}s')
        print()

    def messageReport(self, msg):
        print(msg, file=stderr)

    def databaseChanged(self):
        print('Database changed to', self.inUse)
        print()

    def print(self, results: List[LookupOutput]):
        for result in results:
            if result:
                if result._database:
                    self.inUse = result._database
                    self.databaseChanged()
                if result.headers:
                    self.myprint(result)
                if result._message:
                    self.messageReport(result._message)
            else:
                return



    class MyPT(PrettyTable):
        def _format_value(self, field, value):
            if value is not None:
                return super()._format_value(field, value)
            return 'NULL'


class CSVPrinter:
    def messageReport(self, msg):
        print(msg, file=stderr)

    def myprint(self, result: LookupOutput):
        csv.writer(stdout).writerow(result.headers)
        csv.writer(stdout).writerows(result.data)

    def databaseChanged(self):
        pass

    def print(self, results: List[LookupOutput]):
        for result in results:
            if result:
                if result._database:
                    self.inUse = result._database
                    self.databaseChanged()
                if result.headers:
                    self.myprint(result)
                if result._message:
                    self.messageReport(result._message)
            else:
                returnfrom pathlib import Path
from .join import Join
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.Errors import ParseCancellationException
from antlr4.error.ErrorListener import ErrorListener
from copy import deepcopy
from datetime import date
import re
from typing import Tuple
# from .system_visitor import SystemVisitor
from FileSystem.BufManager import BufManager
from RecordSystem.RecordManager import RecordManager
from RecordSystem.FileScan import FileScan
from RecordSystem.FileHandler import FileHandler
from RecordSystem.record import Record
from RecordSystem.rid import RID
from IndexSystem.index_manager import IndexManager
from MetaSystem.MetaHandler import MetaHandler
from Exceptions.exception import *
from .lookup_element import LookupOutput, Term, Reducer
from SQL_Parser.SQLLexer import SQLLexer
from SQL_Parser.SQLParser import SQLParser
from MetaSystem.info import TableInfo, ColumnInfo
from .macro import *
from .printers import TablePrinter


class SystemManger:
    def __init__(self, visitor, syspath: Path, bm: BufManager, rm: RecordManager, im: IndexManager):
        self.visitor = visitor
        self.systemPath = syspath
        self.BM = bm
        self.IM = im
        self.RM = rm
        self.metaHandlers = {}
        self.databaselist = []
        for item in syspath.iterdir():
            self.databaselist.append(item.name)
        self.inUse = None
        self.visitor.system_manager = self

    def shutdown(self):
        self.IM.close_manager()
        self.RM.shutdown()
        self.BM.shutdown()

    def checkInUse(self):
        if self.inUse is None:
            print("OH NO")
            raise NoDatabaseInUse("use a database first")
        return

    def createDatabase(self, dbname: str):
        if dbname not in self.databaselist:
            path: Path = self.systemPath / dbname
            path.mkdir(parents=True)
            self.databaselist.append(dbname)
        else:
            print("OH NO")
            raise DatabaseAlreadyExist("this name exists")
        return

    def removeDatabase(self, dbname: str):
        if dbname in self.databaselist:
            self.IM.shut_handler(dbname)
            if self.metaHandlers.get(dbname) is not None:
                self.metaHandlers.pop(dbname).shutdown()
            path: Path = self.systemPath / dbname
            for table in path.iterdir():
                if path.name.endswith(".table"):
                    self.RM.closeFile(str(table))
                table.unlink()
            self.databaselist.remove(dbname)
            path.rmdir()
            if self.inUse == dbname:
                self.inUse = None
                result = LookupOutput(change_db='None')
                return result
        else:
            print("OH NO")
            raise DatabaseNotExist("this name doesn't exist")

    def useDatabase(self, dbname: str):
        if dbname in self.databaselist:
            self.inUse = dbname
            result = LookupOutput(change_db=dbname)
            return result
        print("OH NO")
        raise DatabaseNotExist("this name doesn't exist")

    def getTablePath(self, table: str):
        self.checkInUse()
        tablePath = self.systemPath / self.inUse / table
        return str(tablePath) + ".table"

    def execute(self, sql):
        class StringErrorListener(ErrorListener):
            def syntaxError(self, recognizer, offending_symbol, line, column, msg, e):
                raise ParseCancellationException("line " + str(line) + ":" + str(column) + " " + msg)

        self.visitor.spend_time()
        input_stream = InputStream(sql)
        lexer = SQLLexer(input_stream)
        lexer.removeErrorListeners()
        lexer.addErrorListener(StringErrorListener())
        tokens = CommonTokenStream(lexer)
        parser = SQLParser(tokens)
        parser.removeErrorListeners()
        parser.addErrorListener(StringErrorListener())
        try:
            tree = parser.program()
        except ParseCancellationException as e:
            return [LookupOutput(None, None, str(e), cost=self.visitor.spend_time())]
        try:
            return self.visitor.visit(tree)
        except MyException as e:
            return [LookupOutput(message=str(e), cost=self.visitor.spend_time())]

    def displayTableNames(self):
        result = []
        self.checkInUse()
        usingDB = self.systemPath / self.inUse
        for file in usingDB.iterdir():
            if file.name.endswith(".table"):
                result.append(file.stem)
        return result

    def fetchMetaHandler(self):
        if self.metaHandlers.get(self.inUse) is None:
            self.metaHandlers[self.inUse] = MetaHandler(self.inUse, str(self.systemPath))
        return self.metaHandlers[self.inUse]

    def createTable(self, table: TableInfo):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        metaHandler.insertTable(table)
        tablePath = self.getTablePath(table.name)
        self.RM.createFile(tablePath, table.rowSize)
        return

    def removeTable(self, table: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        for col in tableInfo.columnMap:
            self.checkRemoveColumn(table, col)
        metaHandler.removeTable(table)
        tablePath = self.getTablePath(table)
        self.RM.destroyFile(tablePath)
        return

    def collectTableinfo(self, table: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        return metaHandler, metaHandler.collectTableInfo(table)

    def descTable(self, table: str):
        head = ('Field', 'Type', 'Null', 'Key', 'Default', 'Extra')
        data = self.collectTableinfo(table)[1].describe()
        return LookupOutput(head, data)

    def renameTable(self, src: str, dst: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        metaHandler.renameTable(src, dst)
        srcFilename = self.getTablePath(src)
        dstFilename = self.getTablePath(dst)
        self.RM.renameFile(srcFilename, dstFilename)
        return

    def createIndex(self, index: str, table: str, col: str):
        metaHandler, tableInfo = self.collectTableinfo(table)
        if index in metaHandler.databaseInfo.indexMap:
            print("OH NO")
            raise IndexAlreadyExist("this name exists")
        if col in tableInfo.index:
            metaHandler.createIndex(index, table, col)
            return
        indexFile = self.IM.create_index(self.inUse, table)
        tableInfo.index[col] = indexFile.root

        if tableInfo.getColumnIndex(col) is not None:
            colIndex = tableInfo.getColumnIndex(col)
            for record in FileScan(self.RM.openFile(self.getTablePath(table))):
                recordData = tableInfo.loadRecord(record)
                indexFile.insert(recordData[colIndex], record.rid)
            metaHandler.createIndex(index, table, col)
        else:
            print("OH NO")
            raise ColumnNotExist(col + "doesn't exist")

    def removeIndex(self, index: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        table, col = metaHandler.databaseInfo.getIndex(index)
        metaHandler.collectTableInfo(table).index.pop(col)
        metaHandler.removeIndex(index)
        self.metaHandlers.pop(self.inUse).shutdown()
        return

    def addUnique(self, table: str, col: str, uniq: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        metaHandler.addUnique(table, col, uniq)
        if uniq not in metaHandler.databaseInfo.indexMap:
            self.createIndex(uniq, table, col)
        return

    def addForeign(self, table: str, col: str, foreign, forName=None):
        metaHandler, tableInfo = self.collectTableinfo(table)
        if (table, col) not in metaHandler.databaseInfo.indexMap.values():
            raise AddForeignError("create index on this column first")
        if forName:
            if forName not in metaHandler.databaseInfo.indexMap:
                self.createIndex(forName, foreign[0], foreign[1])
        else:
            indexName = foreign[0] + "." + foreign[1]
            if indexName not in metaHandler.databaseInfo.indexMap:
                self.createIndex(indexName, foreign[0], foreign[1])
        tableInfo.addForeign(col, foreign)
        metaHandler.shutdown()
        return

    def removeForeign(self, table, col, forName=None):
        metaHandler, tableInfo = self.collectTableinfo(table)
        if tableInfo.foreign.get(col) is not None:
            foreign = tableInfo.foreign[col][0] + "." + tableInfo.foreign[col][1]
            reftable: TableInfo = metaHandler.collectTableInfo(tableInfo.foreign[col][0])
            if reftable.primary.count(tableInfo.foreign[col][1]) != 0:
                self.removeIndex(foreign)
            tableInfo.removeForeign(col)
            metaHandler.shutdown()

    def setPrimary(self, table: str, pri):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        metaHandler.setPrimary(table, pri)
        if pri:
            for column in pri:
                indexName = table + "." + column
                if indexName not in metaHandler.databaseInfo.indexMap:
                    self.createIndex(indexName, table, column)
        return

    def removePrimary(self, table: str):
        # todo: check foreign
        metaHandler, tableInfo = self.collectTableinfo(table)
        if tableInfo.primary:
            for column in tableInfo.primary:
                indexName = table + "." + column
                if indexName in metaHandler.databaseInfo.indexMap:
                    self.removeIndex(indexName)
            metaHandler.removePrimary(table)
        return

    def addColumn(self, table: str, column, pri: bool, foreign: bool):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if pri:
            for co in column:
                if tableInfo.getColumnIndex(co) is None:
                    print("OH NO")
                    raise ColumnNotExist(co + " doesn't exist")
            self.setPrimary(table, column)
        elif foreign:
            co = column[0]
            if tableInfo.getColumnIndex(co) is None:
                print("OH NO")
                raise ColumnNotExist(co + " doesn't exist")
            self.addForeign(table, co, (column[1], column[2]), None)
        else:
            if not isinstance(column, ColumnInfo):
                raise AddError("unsupported add")
            col = column
            if tableInfo.getColumnIndex(col.name):
                print("OH NO")
                raise ColumnNotExist(col.name + " doesn't exist")
            oldTableInfo: TableInfo = deepcopy(tableInfo)
            metaHandler.databaseInfo.insertColumn(table, col)
            metaHandler.shutdown()
            copyTableFile = self.getTablePath(table + ".copy")
            self.RM.createFile(copyTableFile, tableInfo.rowSize)
            newRecordHandle: FileHandler = self.RM.openFile(copyTableFile)
            scan = FileScan(self.RM.openFile(self.getTablePath(table)))
            for record in scan:
                recordVals = oldTableInfo.loadRecord(record)
                valList = list(recordVals)
                valList.append(col.default)
                newRecordHandle.insertRecord(tableInfo.buildRecord(valList))
            self.RM.closeFile(self.getTablePath(table))
            self.RM.closeFile(copyTableFile)
            self.RM.replaceFile(copyTableFile, self.getTablePath(table))
        return

    def removeColumn(self, table: str, col: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        self.checkRemoveColumn(table, col)
        tableInfo = metaHandler.collectTableInfo(table)
        if col not in tableInfo.columnIndex:
            print("OH NO")
            raise ColumnNotExist(col + " doesn't exist")
        oldTableInfo: TableInfo = deepcopy(tableInfo)
        colIndex = tableInfo.getColumnIndex(col)
        metaHandler.removeColumn(table, col)
        copyTableFile = self.getTablePath(table + ".copy")
        self.RM.createFile(copyTableFile, tableInfo.rowSize)
        newRecordHandle: FileHandler = self.RM.openFile(copyTableFile)
        scan = FileScan(self.RM.openFile(self.getTablePath(table)))
        for record in scan:
            recordVals = oldTableInfo.loadRecord(record)
            valList = list(recordVals)
            valList.pop(colIndex)
            newRecordHandle.insertRecord(tableInfo.buildRecord(valList))
        self.RM.closeFile(self.getTablePath(table))
        self.RM.closeFile(copyTableFile)
        self.RM.replaceFile(copyTableFile, self.getTablePath(table))
        return

    def insertRecord(self, table: str, val: list):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)

        info = tableInfo.buildRecord(val)
        tempRecord = Record(RID(0, 0), info)
        valTuple = tableInfo.loadRecord(tempRecord)

        self.checkInsertConstraint(table, valTuple)
        rid = self.RM.openFile(self.getTablePath(table)).insertRecord(info)
        self.handleInsertIndex(table, valTuple, rid)
        return

    def deleteRecords(self, table: str, limits: tuple):
        self.checkInUse()
        fileHandler = self.RM.openFile(self.getTablePath(table))
        metaHandler = self.fetchMetaHandler()
        records, data = self.searchRecordIndex(table, limits)
        for record, valTuple in zip(records, data):
            self.checkRemoveConstraint(table, valTuple)
            fileHandler.deleteRecord(record.rid)
            self.handleRemoveIndex(table, valTuple, record.rid)
        return LookupOutput('deleted_items', (len(records),))

    def updateRecords(self, table: str, limits: tuple, valmap: dict):
        self.checkInUse()
        fileHandler = self.RM.openFile(self.getTablePath(table))
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        tableInfo.checkValue(valmap)
        records, data = self.searchRecordIndex(table, limits)
        for record, oldVal in zip(records, data):
            new = list(oldVal)
            for col in valmap:
                new[tableInfo.getColumnIndex(col)] = valmap.get(col)
            self.checkRemoveConstraint(table, oldVal)
            rid = record.rid
            self.checkInsertConstraint(table, new, rid)
            self.handleRemoveIndex(table, oldVal, rid)
            record.record = tableInfo.buildRecord(new)
            fileHandler.updateRecord(record)
            self.handleInsertIndex(table, tuple(new), rid)
        return LookupOutput('updated_items', (len(records),))

    def indexFilter(self, table: str, limits: tuple) -> set:
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        condIndex = {}

        def build(limit: Term):
            if limit.type != 1 or (limit.table and limit.table != table):
                return None
            colIndex = tableInfo.getColumnIndex(limit.col)
            if colIndex is not None and limit.value is not None and limit.col in tableInfo.index:
                lo, hi = condIndex.get(limit.col, (-1 << 31 + 1, 1 << 31))
                val = int(limit.value)
                if limit.operator == "=":
                    lower = max(lo, val)
                    upper = min(hi, val)
                elif limit.operator == "<":
                    lower = lo
                    upper = min(hi, val - 1)
                elif limit.operator == ">":
                    lower = max(lo, val + 1)
                    upper = hi
                elif limit.operator == "<=":
                    lower = lo
                    upper = min(hi, val)
                elif limit.operator == ">=":
                    lower = max(lo, val)
                    upper = hi
                else:
                    return None
                condIndex[limit.col] = lower, upper

        results = None
        t = tuple(map(build, limits))

        for col in condIndex:
            if results:
                lo, hi = condIndex.get(col)
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                results = results & set(index.range(lo, hi))
            else:
                lo, hi = condIndex.get(col)
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                results = set(index.range(lo, hi))
        return results

    def searchRecordIndex(self, table: str, limits: tuple):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        functions = self.buildConditionsFuncs(table, limits, metaHandler)
        fileHandler: FileHandler = self.RM.openFile(self.getTablePath(table))
        records = []
        data = []
        if self.indexFilter(table, limits):
            iterator = map(fileHandler.getRecord, self.indexFilter(table, limits))
            for record in iterator:
                valTuple = tableInfo.loadRecord(record)
                if all(map(lambda fun: fun(valTuple), functions)):
                    records.append(record)
                    data.append(valTuple)
        else:
            for record in FileScan(fileHandler):
                valTuple = tableInfo.loadRecord(record)
                if all(map(lambda fun: fun(valTuple), functions)):
                    records.append(record)
                    data.append(valTuple)
        return records, data

    def condJoin(self, res_map: dict, term):
        if self.inUse is None:
            raise ValueError("No using database!!!")
        else:
            join = Join(res_map=res_map, term=term)
            result: LookupOutput = join.get_output()
            return result

    def checkAnyUnique(self, table: str, pairs, thisRID: RID = None):
        conds = []
        for col in pairs:
            conds.append(Term(1, table, col, '=', value=pairs.get(col)))
        records, data = self.searchRecordIndex(table, tuple(conds))
        if len(records) <= 1:
            if records and records[0].rid == thisRID:
                return False
            elif records:
                return (tuple(pairs.keys()), tuple(pairs.values()))
            return False
        print("OH NO")
        raise CheckAnyUniqueError("get " + str(len(records)) + " same")

    def checkPrimary(self, table: str, colVals, thisRID: RID = None):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if tableInfo.primary:
            pairs = {}
            for col in tableInfo.primary:
                pairs[col] = colVals[tableInfo.getColumnIndex(col)]
            return self.checkAnyUnique(table, pairs, thisRID)
        return False

    def checkUnique(self, table: str, colVals, thisRID: RID = None):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if tableInfo.unique:
            for col in tableInfo.unique:
                pairs = {col: colVals[tableInfo.getColumnIndex(col)]}
                if self.checkAnyUnique(table, pairs, thisRID):
                    return self.checkAnyUnique(table, pairs, thisRID)
        return False

    def checkForeign(self, table: str, colVals):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if len(tableInfo.foreign) > 0:
            for col in tableInfo.foreign:
                conds = []
                fortable = tableInfo.foreign[col][0]
                forcol = tableInfo.foreign[col][1]
                conds.append(Term(1, fortable, forcol, '=', value=colVals[tableInfo.getColumnIndex(col)]))
                records, data = self.searchRecordIndex(fortable, tuple(conds))
                if len(records) == 0:
                    return tableInfo.name, colVals[tableInfo.getColumnIndex(col)]
                # colVal = colVals[tableInfo.getColumnIndex(col)]
                # foreignTableInfo: TableInfo = metaHandler.collectTableInfo(tableInfo.foreign[col][0])
                # index = self.IM.start_index(self.inUse, tableInfo.foreign[col][0],
                #                             foreignTableInfo.index[tableInfo.foreign[col][1]])
                # if len(set(index.range(colVal, colVal))) == 0:
                #     return col, colVal
        return False

    def checkInsertConstraint(self, table: str, colVals, thisRID: RID = None):
        if self.checkForeign(table, colVals):
            miss = self.checkForeign(table, colVals)
            print("OH NO")
            raise MissForeignKeyError("miss: " + str(miss[0]) + ": " + str(miss[1]))

        if self.checkPrimary(table, colVals, thisRID):
            dup = self.checkPrimary(table, colVals, thisRID)
            print("OH NO")
            raise DuplicatedPrimaryKeyError("duplicated: " + str(dup[0]) + ": " + str(dup[1]))

        if self.checkUnique(table, colVals, thisRID):
            dup = self.checkUnique(table, colVals, thisRID)
            print("OH NO")
            raise DuplicatedUniqueKeyError("duplicated: " + str(dup[0]) + ": " + str(dup[1]))

        return

    def checkRemoveColumn(self, table: str, col: str):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        for tableInfo in metaHandler.databaseInfo.tableMap.values():
            if tableInfo.name != table and len(tableInfo.foreign) > 0:
                for fromcol, (tab, column) in tableInfo.foreign.items():
                    if tab == table and col == column:
                        raise RemoveError("referenced foreignkey column")
        return False

    def checkRemoveConstraint(self, table: str, colVals):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        thistable = metaHandler.collectTableInfo(table)
        for tableInfo in metaHandler.databaseInfo.tableMap.values():
            if len(tableInfo.foreign) > 0:
                for fromcol, (tab, col) in tableInfo.foreign.items():
                    if tab == table:
                        colval = colVals[thistable.getColumnIndex(col)]
                        index = self.IM.start_index(self.inUse, tableInfo.name, tableInfo.index[fromcol])
                        if len(set(index.range(colval, colval))) != 0:
                            raise RemoveError("referenced foreignkey value")
        return False

    def handleInsertIndex(self, table: str, data: tuple, rid: RID):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        for col in tableInfo.index:
            if data[tableInfo.getColumnIndex(col)]:
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                index.insert(data[tableInfo.getColumnIndex(col)], rid)
            else:
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                index.insert(NULL_VALUE, rid)
        return

    def handleRemoveIndex(self, table: str, data: tuple, rid: RID):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        for col in tableInfo.index:
            if data[tableInfo.getColumnIndex(col)]:
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                index.delete(data[tableInfo.getColumnIndex(col)], rid)
            else:
                index = self.IM.start_index(self.inUse, table, tableInfo.index[col])
                index.delete(NULL_VALUE, rid)
        return

    def buildConditionsFuncs(self, table: str, limits, metahandler):
        tableInfo = metahandler.collectTableInfo(table)

        def build(limit: Term):
            if limit.table is not None and limit.table != table:
                return None
            colIndex = tableInfo.getColumnIndex(limit.col)
            if colIndex is not None:
                colType = tableInfo.columnType[colIndex]
                if limit.type == 1:
                    if limit.aim_col:
                        if limit.aim_table == table:
                            return self.compare(colIndex, limit.operator, tableInfo.getColumnIndex(limit.aim_col))
                        return None
                    else:
                        if colType == "DATE":
                            if type(limit.value) not in (str, date):
                                raise ValueTypeError("need str/date here")
                            val = limit.value
                            if type(val) is date:
                                return self.compareV(colIndex, limit.operator, val)
                            valist = val.replace("/", "-").split("-")
                            return self.compareV(colIndex, limit.operator, date(*map(int, valist)))
                        elif colType in ("INT", "FLOAT"):
                            if isinstance(limit.value, (int, float)):
                                return self.compareV(colIndex, limit.operator, limit.value)
                            raise ValueTypeError("need int/float here")
                        elif colType == "VARCHAR":
                            if isinstance(limit.value, str):
                                return self.compareV(colIndex, limit.operator, limit.value)
                            raise ValueTypeError("need varchar here")
                        raise ValueTypeError("limit value error")
                elif limit.type == 2:
                    if colType == "DATE":
                        values = []
                        for val in limit.value:
                            if type(val) is str:
                                valist = val.replace("/", "-").split("-")
                                values.append(date(*map(int, valist)))
                            elif type(val) is date:
                                values.append(val)
                            raise ValueTypeError("need str/date here")
                        return lambda x: x[colIndex] in tuple(values)
                    return lambda x: x[colIndex] in limit.value
                elif limit.type == 3:
                    if colType == "VARCHAR":
                        return lambda x: self.buildPattern(limit.value).match(str(x[colIndex]))
                    raise ValueTypeError("like need varchar here")
                elif limit.type == 0:
                    if isinstance(limit.value, bool):
                        if limit.value:
                            return lambda x: x[colIndex] is None
                        return lambda x: x[colIndex] is not None
                    raise ValueTypeError("limit value need bool here")
                raise ValueTypeError("limit type unknown")
            raise ColumnNotExist("limit column name unknown")

        results = []
        for limit in limits:
            func = build(limit)
            if func is not None:
                results.append(func)
        return results

    def selectRecords(self, reducers: Tuple[Reducer], tables: Tuple[str, ...],
                      limits: Tuple[Term], groupBy: Tuple[str, str]):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()

        def getSelected(col2data):
            col2data['*.*'] = next(iter(col2data.values()))
            return tuple(map(lambda x: x.select(col2data[x.target()]), reducers))

        def setTableName(object, tableName, colName):
            if getattr(object, colName) is None:
                return
            elif getattr(object, tableName) is None:
                tabs = col2tab.get(getattr(object, colName))
                if not tabs:
                    raise ColumnNotExist(getattr(object, colName) + " unknown")
                elif len(tabs) > 1:
                    raise SameNameError(getattr(object, colName) + " exists in multiple tables")
                setattr(object, tableName, tabs[0])
            return

        col2tab = metaHandler.getColumn2Table(tables)
        groupTable, groupCol = groupBy
        for element in limits + reducers:
            if not isinstance(element, Term):
                setTableName(element, '_table_name', '_col')
            else:
                setTableName(element, 'aim_table', 'aim_col')

        groupTableName = groupTable or tables[0]
        groupTable = groupTableName
        groupBy = groupTable + '.' + groupCol
        reducerTypes = []
        for reducer in reducers:
            reducerTypes.append(reducer.reducer_type)
        reducerTypes = set(reducerTypes)
        if not groupCol and 1 in reducerTypes and len(reducerTypes) > 1:
            raise SelectError("no-group select contains both field and aggregations")

        if not reducers and not groupCol and len(tables) == 1 and reducers[0].reducer_type == 3:
            tableInfo = metaHandler.collectTableInfo(tables[0])
            fileHandler = self.RM.openFile(self.getTablePath(tables[0]))
            return LookupOutput((reducers[0].to_string(False),), (fileHandler.head['AllRecord']))
        tab2results = {}
        for table in tables:
            tab2results[table] = self.condScanIndex(table, limits)
        result = None
        if len(tables) == 1:
            result = tab2results[tables[0]]
        else:
            result = self.condJoin(tab2results, limits)

        if not groupCol:
            if reducers[0].reducer_type == 0:
                if len(reducers) == 1:
                    return result
                raise SelectError("reducer num not 1")
            elif 1 in reducerTypes:
                heads = []
                headindexes = []
                for reducer in reducers:
                    heads.append(reducer.target())
                headers = tuple(heads)
                for head in headers:
                    headindexes.append(result.header_id(head))
                indexes = tuple(headindexes)

                def takeCol(row):
                    return tuple(row[ele] for ele in indexes)

                data = tuple(map(takeCol, result.data))
            else:
                if result.data is not None:
                    head2data = {}
                    for head, data in zip(result.headers, zip(*result.data)):
                        head2data[head] = data
                    data = getSelected(head2data)
                else:
                    data = (None,) * len(result.headers)
        else:
            def getRow(group):
                head2data = {}
                for item_head, item_data in zip(result.headers, zip(*group)):
                    head2data[item_head] = item_data
                return getSelected(head2data)

            index = result.header_id(groupBy)
            groups = {}
            for row in result.data:
                if groups.get(row[index]) is None:
                    groups[row[index]] = [row]
                else:
                    groups[row[index]].append(row)
            if reducers[0].reducer_type == 0:
                return LookupOutput(result.headers, tuple(group[0] for group in groups.values()))
            data = tuple(map(getRow, groups.values()))

        headers = []
        for reducer in reducers:
            headers.append(reducer.to_string(len(tables) > 1))
        return LookupOutput(tuple(headers), data)

    def selectRecordsLimit(self, reducers, tables, limits, groupBy, limit: int, off: int):
        result = self.selectRecords(reducers, tables, limits, groupBy)
        if limit is None:
            data = result.data[off:]
        else:
            data = result.data[off: off + limit]
        return LookupOutput(result.headers, data)

    def condScanIndex(self, table: str, limits: tuple):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        records, data = self.searchRecordIndex(table, limits)
        headers = tuple(tableInfo.name + "." + colName for colName in tableInfo.columnMap.keys())
        return LookupOutput(headers, data)

    @staticmethod
    def resultToValue(result: LookupOutput, is_in):
        if len(result.headers) <= 1:
            val = sum(result.data, ())
            if not is_in:
                if len(result.data) == 1:
                    val, = val
                    return val
                raise ValueError("expect one value, get " + str(len(result.data)))
            return val
        raise SelectError("expect one column, get " + str(len(result.headers)))

    @staticmethod
    def printResults(result: LookupOutput):
        TablePrinter().print([result])

    @staticmethod
    def compare(this, operator, other):
        if operator == "<":
            return lambda x: x[this] < x[other]
        elif operator == "<=":
            return lambda x: x[this] <= x[other]
        elif operator == ">":
            return lambda x: x[this] > x[other]
        elif operator == ">=":
            return lambda x: x[this] >= x[other]
        elif operator == "<>":
            return lambda x: x[this] != x[other]
        elif operator == "=":
            return lambda x: x[this] == x[other]

    @staticmethod
    def compareV(this, operator, val):
        if operator == '<':
            return lambda x: x is not None and x[this] < val
        elif operator == '<=':
            return lambda x: x is not None and x[this] <= val
        elif operator == '>':
            return lambda x: x is not None and x[this] > val
        elif operator == '>=':
            return lambda x: x is not None and x[this] >= val
        elif operator == '<>':
            return lambda x: x[this] != val
        elif operator == '=':
            return lambda x: x[this] == val

    @staticmethod
    def buildPattern(pat: str):
        pat = pat.replace('%%', '\r')
        pat = pat.replace('%?', '\n')
        pat = pat.replace('%_', '\0')
        pat = re.escape(pat)
        pat = pat.replace('%', '.*')
        pat = pat.replace(r'\?', '.')
        pat = pat.replace('_', '.')
        pat = pat.replace('\r', '%')
        pat = pat.replace('\n', r'\?')
        pat = pat.replace('\0', '_')
        pat = re.compile('^' + pat + '$')
        return pat
from Exceptions.exception import ValueTypeError
class Term:
    """term_type:   0 is null
                    1 is compare
                    2 is in
                    3 is like
    """

    def __init__(self, term_type, table_name, col, operator=None, aim_table_name=None, aim_col=None, value=None):
        self._type: int = term_type
        self._table: str = table_name
        self._col: str = col
        self._operator: str = operator
        self._aim_table: str = aim_table_name
        self._aim_col: str = aim_col
        self._value = value

    @property
    def aim_table(self):
        return self._aim_table

    @property
    def table(self):
        return self._table

    @property
    def operator(self):
        return self._operator

    @property
    def col(self):
        return self._col

    @property
    def aim_col(self):
        return self._aim_col

    @property
    def value(self):
        return self._value

    @property
    def type(self):
        return self._type


class Reducer:
    """reducer_type:0 is all
                    1 is field
                    2 is aggregation
                    3 is counter
    """

    def __init__(self, reducer_type, table_name=None, col=None, aggregator=None):
        self._reducer_type: int = reducer_type
        self._table_name: str = table_name
        self._col: str = col
        self._aggregator: str = aggregator

    def target(self):
        return f'{self._table_name}.{self._col}'

    def to_string(self, prefix=True):
        base = self.target()
        if self._reducer_type == 1:
            return base if prefix else self._col
        if self._reducer_type == 2:
            return f'{self._aggregator}({base})' if prefix else f'{self._aggregator}({self._col})'
        if self._reducer_type == 3:
            return f'COUNT(*)'

    def select(self, data: tuple):
        function_map = {
            'COUNT': lambda x: len(set(x)),
            'MAX': max,
            'MIN': min,
            'SUM': sum,
            'AVG': lambda x: sum(x) / len(x)
        }
        if self._reducer_type == 3:
            return len(data)
        if self._reducer_type == 1:
            return data[0]
        if self._reducer_type == 2:
            try:
                result = function_map[self._aggregator](tuple(filter(lambda x: x is not None, data)))
                return result
            except TypeError:
                raise ValueTypeError("incorrect value type for aggregation")

    @property
    def reducer_type(self):
        return self._reducer_type


class LookupOutput:
    # todo:modified
    def __init__(self, headers=None, data=None, message=None, change_db=None, cost=None):
        if headers and not isinstance(headers, (list, tuple)):
            headers = (headers,)
        if data and not isinstance(data[0], (list, tuple)):
            data = tuple((each,) for each in data)
        self._headers = headers
        self._data = data
        self._header_index = {h: i for i, h in enumerate(headers)} if headers else {}
        self._alias_map = {}
        self._message = message
        self._database = change_db
        self._cost = cost

    def simplify(self):
        """Simplify headers if all headers have same prefix"""
        if not self._headers:
            return
        header: str = self._headers[0]
        if header.find('.') < 0:
            return
        prefix = header[:header.find('.') + 1]  # Prefix contains "."
        for header in self._headers:
            if len(header) <= len(prefix) or not header.startswith(prefix):
                break
        else:
            self._headers = tuple(header[len(prefix):] for header in self._headers)

    def size(self):
        size: int = len(self._data)
        return size

    @property
    def data(self):
        return self._data

    @property
    def headers(self):
        return self._headers

    def header_id(self, header) -> int:
        if header in self._alias_map:
            header = self._alias_map[header]
        if header in self._header_index:
            res = self._header_index[header]
            return res

    def insert_alias(self, alias, header):
        self._alias_map[alias] = header
        return None

    @property
    def alias_map(self):
        return self._alias_map

    @property
    def cost(self):
        return self._cost

    @cost.setter
    def cost(self, value):
        self._cost = value
PAGE_SIZE = 8192

PAGE_INT_NUM = 2048

PAGE_SIZE_IDX = 13
MAX_FMT_INT_NUM = 128
BUF_PAGE_NUM = 65536
MAX_FILE_NUM = 128
MAX_TYPE_NUM = 256


RECORD_PAGE_NEXT_OFFSET = 1

RECORD_PAGE_FIXED_HEADER = RECORD_PAGE_NEXT_OFFSET + 4

NULL_VALUE = -1 << 32

PAGE_FLAG_OFFSET = 0

RECORD_PAGE_FLAG = 0

VARCHAR_PAGE_FLAG = 1

IN_DEBUG = 0
DEBUG_DELETE = 0
DEBUG_ERASE = 1
DEBUG_NEXT = 1

MAX_COL_NUM = 31

MAX_TB_NUM = 31
RELEASE = 1

from .system_manager import SystemManger
from pathlib import Path
from Exceptions.exception import MyException
from .lookup_element import LookupOutput



class Executor:

    def exec_csv(self, manager: SystemManger, path: Path, dbname: str, tbname: str):
        def load(iterator):
            m, tableInfo = manager.collectTableinfo(tbname)
            def parse(valtypePair):
                val, type = valtypePair
                if type == "INT":
                    return int(val) if val else None
                elif type == "FLOAT":
                    return float(val) if val else None
                elif type == "VARCHAR":
                    return val.rstrip()
                elif type == "DATE":
                    return val if val else None
            inserted = 0
            for row in iterator:
                if row[-1] == '':
                    row = row[:-1]
                row = row.split(',')
                result = tuple(map(parse, zip(row, tableInfo.columnType)))
                # try:
                manager.insertRecord(tbname, list(result))
                inserted += 1
            return inserted

        if not tbname:
            tbname = path.stem
        manager.useDatabase(dbname)
        inserted = load(open(path, encoding='utf-8'))
        timeCost = manager.visitor.spend_time()
        return [LookupOutput('inserted_items', (inserted,), cost=timeCost)]

    def exec_sql(self, manager: SystemManger, path: Path, dbname: str, tbname: str):
        if not dbname:
            return manager.execute(open(path, encoding='utf-8').read())
        manager.useDatabase(dbname)
        return manager.execute(open(path, encoding='utf-8').read())

    def execute(self, manager: SystemManger, path: Path, dbname: str, tbname: str):
        manager.visitor.spend_time()
        if getattr(self, 'exec_' + path.suffix.lstrip('.')):
            try:
                func = getattr(self, 'exec_' + path.suffix.lstrip('.'))
                return func(manager, path, dbname, tbname)
            except MyException as e:
                timeCost = manager.visitor.spend_time()
                return [LookupOutput(message=str(e), cost=timeCost)]
        timeCost = manager.visitor.spend_time()
        return [LookupOutput(message="Unsupported format " + path.suffix.lstrip('.'), cost=timeCost)]
from .info import *
import pickle as pic
import os


class MetaHandler:
    def __init__(self, database: str, syspath: str):

        self.databaseName = database
        self.systemPath = syspath
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.databaseInfo = None

        if not os.path.exists(self.metaPath):
            self.databaseInfo = DatabaseInfo(self.databaseName, [])
            self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
            self.toPickle(self.metaPath)
        else:
            self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
            metaInfo = open(self.metaPath, 'rb')
            self.databaseInfo = pic.load(metaInfo)
            metaInfo.close()

    def toPickle(self, path: str):
        metaInfo = open(path, 'wb')
        pic.dump(self.databaseInfo, metaInfo)
        metaInfo.close()
        return

    def insertTable(self, table: TableInfo):
        self.databaseInfo.insertTable(table)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def removeTable(self, table: str):
        self.databaseInfo.removeTable(table)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def collectTableInfo(self, table: str):
        if self.databaseInfo.tableMap.get(table) is None:
            print("OH NO")
            raise TableNotExist("this name doesn't exist")
        tableInfo = self.databaseInfo.tableMap[table]
        return tableInfo

    def insertColumn(self, table: str, col: ColumnInfo):
        self.databaseInfo.insertColumn(table, col)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def removeColumn(self, table: str, column: str):
        self.databaseInfo.removeColumn(table, column)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def createIndex(self, index: str, table: str, column: str):
        self.databaseInfo.createIndex(index, table, column)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def removeIndex(self, index: str):
        self.databaseInfo.removeIndex(index)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def setPrimary(self, table: str, pri):
        self.collectTableInfo(table).primary = pri
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def removePrimary(self, table: str):
        self.collectTableInfo(table).primary = None
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def addUnique(self, table: str, column: str, uniq: str):
        self.collectTableInfo(table).addUnique(column, uniq)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def renameTable(self, src: str, dst: str):
        if self.databaseInfo.tableMap.get(src) is None:
            print("OH NO")
            raise TableNotExist("this name doesn't exist")
        srcInfo = self.databaseInfo.tableMap.pop(src)
        self.databaseInfo.tableMap[dst] = srcInfo
        indexMap = self.databaseInfo.indexMap
        for index in indexMap.keys():
            if indexMap.get(index)[0] == src:
                columnName = indexMap.get(index)[1]
                self.databaseInfo.indexMap[index] = (dst, columnName)
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def renameIndex(self, src: str, dst: str):
        if self.databaseInfo.indexMap.get(src) is None:
            print("OH NO")
            raise IndexNotExist("this name doesn't exist")
        info = self.databaseInfo.indexMap.pop(src)
        self.databaseInfo.indexMap[dst] = info
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def shutdown(self):
        self.metaPath = os.path.join(self.systemPath, self.databaseName, self.databaseName + ".me")
        self.toPickle(self.metaPath)
        return

    def getColumn2Table(self, tables: list):
        result = {}
        for table in tables:
            tableInfo = self.collectTableInfo(table)
            for col in tableInfo.columnMap.keys():
                if result.get(col) is None:
                    result[col] = [table]
                else:
                    result[col].append(table)
        return result
   Bud1           	                                                           i t _ _ . p                                                                                                                                                                                                                                                                                                                                                                                                                                           _ _ i n i t _ _ . p yIlocblob      A   .      _ _ p y c a c h e _ _Ilocblob         .      i n f o . p yIlocblob        .      m a c r o . p yIlocblob        .      M e t a H a n d l e r . p yIlocblob        .                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              @                                              @                                                @                                                @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   E  	                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       DSDB                                 `                                                   @                                                @                                                @                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
from datetime import date
from Exceptions.exception import *
from .macro import *
import numpy as np
import struct
from numbers import Number
from RecordSystem.record import Record


class ColumnInfo:
    def __init__(self, type: str, name: str, size: int, null_permit: bool = True, default=None):
        self.type = type
        self.name = name
        self.size = size
        self.null_permit = null_permit
        self.default = default

    def getSize(self):
        if self.type == "VARCHAR":
            return self.size + 1
        return 8

    def getDESC(self):
        """name, type, null, keytype, default, extra"""
        return [self.name, self.type, "OK" if self.null_permit else "NO", "", self.default, ""]


class TableInfo:
    def __init__(self, name: str, contents: list):
        self.contents = contents
        self.name = name
        self.primary = None

        self.columnMap = {col.name: col for col in self.contents}
        self.columnSize = [col.getSize() for col in self.contents]
        self.columnType = [col.type for col in self.contents]
        self.foreign = {}
        self.index = {}
        self.rowSize = sum(self.columnSize)
        self.unique = {}
        self.columnIndex = {self.contents[i].name: i for i in range(len(self.contents))}

    def describe(self):
        desc = {col.name: col.getDESC() for col in self.contents}
        if self.primary:
            for name in self.primary:
                desc[name][3] = 'primary'
        for name in self.foreign:
            if desc[name][3]:
                desc[name][3] = 'multi'
            else:
                desc[name][3] = 'foreign'
        for name in self.unique:
            if desc[name][3] == "":
                desc[name][3] = 'unique'
        return tuple(desc.values())

    def insertColumn(self, col: ColumnInfo):
        if col.name not in self.columnMap:
            self.contents.append(col)
            self.columnMap = {col.name: col for col in self.contents}
            self.columnSize = [col.getSize() for col in self.contents]
            self.columnType = [col.type for col in self.contents]
            self.rowSize = sum(self.columnSize)
            self.columnIndex = {self.contents[i].name: i for i in range(len(self.contents))}
        else:
            print("OH NO")
            raise ColumnAlreadyExist("this name exists")
        return

    def removeColumn(self, name: str):
        if name in self.columnMap:
            self.contents.pop(self.columnIndex.get(name))
            self.columnMap = {col.name: col for col in self.contents}
            self.columnSize = [col.getSize() for col in self.contents]
            self.columnType = [col.type for col in self.contents]
            self.rowSize = sum(self.columnSize)
            self.columnIndex = {self.contents[i].name: i for i in range(len(self.contents))}
        else:
            print("OH NO")
            raise ColumnNotExist("this name doesn't exist")
        return

    def addForeign(self, column: str, foreign):
        self.foreign[column] = foreign
        return

    def removeForeign(self, column: str):
        if column in self.foreign:
            self.foreign.pop(column)
        return

    def addUnique(self, column: str, uniq):
        self.unique[column] = uniq
        return

    def buildRecord(self, val: list):
        if len(val) != len(self.columnSize):
            print("OH NO")
            raise ValueNumError("the number of value doesn't match this table")
        record = np.zeros(self.rowSize, dtype=np.uint8)
        pos = 0
        for i in range(len(self.columnSize)):
            size = self.columnSize[i]
            type = self.columnType[i]
            value = val[i]
            if type == "VARCHAR":
                length = 0
                byte = (1, )
                if value is not None:
                    try:
                        byte = (0, ) + tuple(value.encode())
                        if len(byte) > size:
                            print("OH NO")
                            raise VarcharTooLong("too long. max size is " + str(size - 1))
                    except AttributeError:
                        raise ValueTypeError("wrong value type")
                else:
                    byte = (1, )
                length = len(byte)
                record[pos: pos + length] = byte
                for i in range(pos + length, pos + size):
                    record[i] = 0
            else:
                for i in range(size):
                    record[i + pos] = self.serialedValue(value, type)[i]
            pos = pos + size
        return record

    def serialedValue(self, val, type: str):
        if val is None:
            val = NULL_VALUE
            if type == "FLOAT":
                return struct.pack('<d', val)
            else:
                return struct.pack('<q', val)
        else:
            if type == "DATE":
                try:
                    val = val.replace("/", "-")
                    vals = val.split("-")
                except AttributeError:
                    raise DateValueError("date value invalid")
                try:
                    d = date(*map(int, vals))
                except ValueError:
                    raise DateValueError("date value invalid")
                return struct.pack('<q', d.toordinal())
            elif type == "INT":
                if isinstance(val, int):
                    return struct.pack('<q', val)
                else:
                    print("OH NO")
                    raise ValueTypeError("expect int")
            elif type == "FLOAT":
                if isinstance(val, Number):
                    return struct.pack('<d', val)
                else:
                    print("OH NO")
                    raise ValueTypeError("expect float")
            else:
                print("OH NO")
                raise ValueTypeError("expect varchar, int, float or date")

    def loadRecord(self, record: Record):
        pos = 0
        result = []
        row = record.record
        for i in range(len(self.columnSize)):
            type = self.columnType[i]
            size = self.columnSize[i]
            data = row[pos: pos + size]
            val = None
            if type == "VARCHAR":
                if data[0]:
                    val = None
                else:
                    val = data.tobytes()[1:].rstrip(b'\x00').decode('utf-8')
            elif type == "DATE" or type == "INT":
                val = struct.unpack('<q', data)[0]
                if val > 0 and type == "DATE":
                    val = date.fromordinal(val)
            elif type == "FLOAT":
                val = struct.unpack('<d', data)
                val = val[0]
            else:
                print("OH NO")
                raise ValueTypeError("expect varchar, int, float or date")
            if val == NULL_VALUE:
                result.append(None)
            else:
                result.append(val)
            pos += size
        return tuple(result)

    def getColumnIndex(self, name: str):
        index = self.columnIndex.get(name)
        return index

    def checkValue(self, valueMap: dict):
        for name in valueMap:
            columnInfo = self.columnMap.get(name)
            if columnInfo is not None:
                val = valueMap.get(name)
                if columnInfo.type == "INT":
                    if type(val) is not int:
                        print("OH NO")
                        raise ValueTypeError(name + " expect int")
                elif columnInfo.type == "FLOAT":
                    if type(val) not in (int, float):
                        print("OH NO")
                        raise ValueTypeError(name + " expect float")
                elif columnInfo.type == "VARCHAR":
                    if type(val) is not str:
                        print("OH NO")
                        raise ValueTypeError(name + " expect varchar")
                elif columnInfo.type == "DATE":
                    if type(val) not in (date, str):
                        print("OH NO")
                        raise ValueTypeError(name + " expect date")
                    if type(val) is str:
                        val = val.replace("/", "-")
                        vals = val.split("-")
                        valueMap[name] = date(*map(int, vals))
            else:
                raise ValueTypeError("unknown field: " + name)
        return

class DatabaseInfo:
    def __init__(self, name, tables):
        self.name = name
        self.tableMap = {}
        for table in tables:
            self.tableMap[table.name] = table
        self.indexMap = {}

    def insertTable(self, table: TableInfo):
        if self.tableMap.get(table.name) is None:
            self.tableMap[table.name] = table
        else:
            print("OH NO")
            raise TableAlreadyExist("this name exists")
        return

    def insertColumn(self, table: str, col: ColumnInfo):
        if self.tableMap.get(table) is None:
            print("OH NO")
            raise TableNotExist("this name doesn't exist")
        self.tableMap[table].insertColumn(col)
        return

    def removeTable(self, table: str):
        if self.tableMap.get(table) is None:
            print("OH NO")
            raise TableNotExist("this name doesn't exist")
        self.tableMap.pop(table)
        return

    def removeColumn(self, table: str, col: str):
        if self.tableMap.get(table) is None:
            print("OH NO")
            raise TableNotExist("this name doesn't exist")
        self.tableMap[table].removeColumn(col)
        return

    def createIndex(self, index: str, table: str, col: str):
        if self.indexMap.get(index) is None:
            self.indexMap[index] = (table, col)
        else:
            print("OH NO")
            raise IndexAlreadyExist("this name exists")
        return

    def removeIndex(self, index: str):
        if self.indexMap.get(index) is None:
            print("OH NO")
            raise IndexNotExist("this name doesn't exist")
        self.indexMap.pop(index)
        return

    def getIndex(self, index: str):
        if self.indexMap.get(index) is None:
            raise IndexNotExist("this name doesn't exist")
        return self.indexMap.get(index)

PAGE_SIZE = 8192

PAGE_INT_NUM = 2048

PAGE_SIZE_IDX = 13
MAX_FMT_INT_NUM = 128
BUF_PAGE_NUM = 65536
MAX_FILE_NUM = 128
MAX_TYPE_NUM = 256


RECORD_PAGE_NEXT_OFFSET = 1

RECORD_PAGE_FIXED_HEADER = RECORD_PAGE_NEXT_OFFSET + 4

NULL_VALUE = -1 << 32

PAGE_FLAG_OFFSET = 0

RECORD_PAGE_FLAG = 0

VARCHAR_PAGE_FLAG = 1

IN_DEBUG = 0
DEBUG_DELETE = 0
DEBUG_ERASE = 1
DEBUG_NEXT = 1

MAX_COL_NUM = 31

MAX_TB_NUM = 31
RELEASE = 1

import numpy as np
from json import loads

# from .RecordManager import RecordManager
from .macro import *
from .rid import RID
from .record import Record

class FileHandler:

    def __init__(self, rm, fid: int, name: str):
        self.RM = rm
        self.fileID = fid
        self.name = name

        self.headChanged = False
        self.open = True # not opened
        self.headpage = self.RM.BM.getPage(self.fileID, 0)
        self.head = loads(self.headpage.tobytes().decode('utf-8').rstrip('\0'))

    def changeHead(self):
        serialized = self.RM.toSerial(self.head)
        self.RM.BM.fetchPage(self.fileID, 0, serialized)
        return

    def getBitmap(self, pageBuf: np.ndarray):
        num = self.head['RecordNum']
        bml = self.head['BitmapLen']
        bitinfo = pageBuf[RECORD_PAGE_FIXED_HEADER: RECORD_PAGE_FIXED_HEADER + bml]
        bitmap = np.unpackbits(bitinfo)[:num]
        return bitmap

    def getNextAvai(self, pageBuf: np.ndarray):
        byte = pageBuf[RECORD_PAGE_NEXT_OFFSET: RECORD_PAGE_NEXT_OFFSET + 4].tobytes()
        return int.from_bytes(byte, 'big')

    def setNextAvai(self, pageBuf: np.ndarray, pageID: int):
        data = np.frombuffer(pageID.to_bytes(4, 'big'), dtype=np.uint8)
        pageBuf[RECORD_PAGE_NEXT_OFFSET: RECORD_PAGE_NEXT_OFFSET + 4] = data
        return

    def getPage(self, pageID: int):
        return self.RM.BM.getPage(self.fileID, pageID)

    def fetchPage(self, pageID: int, buf: np.ndarray):
        self.RM.BM.fetchPage(self.fileID, pageID, buf)
        return

    def newPage(self):
        pID = self.RM.BM.FM.newPage(self.fileID, np.zeros(PAGE_SIZE, dtype=np.uint8))
        return pID

    def getRecord(self, rid: RID, buf=None):
        slotID = rid.slot
        if buf is None:
            pID = rid.page
            buf = self.RM.BM.getPage(self.fileID, pID)
        recordLen = self.head['RecordLen']
        bitmapLen = self.head['BitmapLen']
        recordOff = RECORD_PAGE_FIXED_HEADER + bitmapLen + slotID * recordLen
        record = Record(rid, buf[recordOff: recordOff + recordLen])
        return record

    def Nextvalid(self):
        if self.head['NextAvai'] == 0:
            return False
        return True

    def insertRecord(self, record: np.ndarray):
        if not self.Nextvalid():
            self.appendPage()
        nextAvai = self.head['NextAvai']
        page = self.RM.BM.getPage(self.fileID, nextAvai)
        bitmap = self.getBitmap(page)

        slotID = np.where(bitmap)[0][0]
        if len(np.where(bitmap)[0]) == 1:
            self.head['NextAvai'] = self.getNextAvai(page)
            self.setNextAvai(page, nextAvai)

        recordLen = self.head['RecordLen']
        bitmapLen = self.head['BitmapLen']
        recordOff = RECORD_PAGE_FIXED_HEADER + bitmapLen + slotID * recordLen

        bitmap[slotID] = False
        self.headChanged = True
        page[recordOff: recordOff + recordLen] = record
        bitmap = np.packbits(bitmap)
        page[RECORD_PAGE_FIXED_HEADER: RECORD_PAGE_FIXED_HEADER + bitmapLen] = bitmap
        self.head['AllRecord'] += 1

        self.RM.BM.fetchPage(self.fileID, nextAvai, page)
        rid = RID(nextAvai, slotID)
        return rid

    def appendPage(self):
        buf = np.full(PAGE_SIZE, -1, dtype=np.uint8)
        buf[PAGE_FLAG_OFFSET] = RECORD_PAGE_FLAG
        self.setNextAvai(buf, self.head['NextAvai'])
        pID = self.RM.BM.FM.newPage(self.fileID, buf)
        self.head['NextAvai'] = pID
        self.headChanged = True
        self.head['PageNum'] += 1
        return

    def deleteRecord(self, rid: RID):
        page = self.RM.BM.getPage(self.fileID, rid.page)
        bitmap = self.getBitmap(page)
        bitmapLen = self.head['BitmapLen']
        self.head['AllRecord'] -= 1
        if bitmap[rid.slot] == 0:
            bitmap[rid.slot] = True
        self.headChanged = True

        # bitmap = np.packbits(bitmap)
        page[RECORD_PAGE_FIXED_HEADER: RECORD_PAGE_FIXED_HEADER + bitmapLen] = np.packbits(bitmap)
        if self.getNextAvai(page) == rid.page:
            self.setNextAvai(page, self.head['NextAvai'])
            self.head['NextAvai'] = rid.page
        self.RM.BM.fetchPage(self.fileID, rid.page, page)
        return

    def updateRecord(self, record: Record):
        page = self.RM.BM.getPage(self.fileID, record.rid.page)

        recordLen = self.head['RecordLen']
        bitmapLen = self.head['BitmapLen']
        recordOff = RECORD_PAGE_FIXED_HEADER + bitmapLen + record.rid.slot * recordLen
        page[recordOff: recordOff + recordLen] = record.record
        self.RM.BM.fetchPage(self.fileID, record.rid.page, page)

    def __del__(self):
        if self.open:
            self.RM.closeFile(self.name)
class RID:
    def __init__(self, page_value, slot_value):
        self._page = page_value
        self._slot = slot_value

    @property
    def page(self):
        return self._page



    @property
    def slot(self):
        return self._slot


    def __str__(self):
        return f'{{page: {self.page}, slot: {self.slot}}}'

    def __eq__(self, other):
        if other is None:
            return False
        return self._page == other.page and self._slot == other.slot

    def __hash__(self):
        return hash((self._page, self._slot))
from .macro import *
from .FileHandler import FileHandler
from .rid import RID

import numpy as np

class FileScan:
    def __init__(self, handler: FileHandler):
        self.handler = handler

    def __iter__(self):
        pageNum = self.handler.head['PageNum']
        for pID in range(1, pageNum):
            page = self.handler.RM.BM.getPage(self.handler.fileID, pID)
            if page[PAGE_FLAG_OFFSET] == RECORD_PAGE_FLAG:
                bitmap = self.handler.getBitmap(page)
                for slot in range(len(bitmap)):
                    if bitmap[slot] == 0:
                        rid = RID(pID, slot)
                        yield self.handler.getRecord(rid, page)

import numpy as np
from .rid import RID

class Record:

    def __init__(self, rid: RID, record: np.ndarray):
        self.record = record
        self.rid = rid

from Exceptions.exception import *
from FileSystem.FileManager import FileManager
from FileSystem.BufManager import BufManager
from .FileHandler import FileHandler
from .macro import *
from json import dumps, loads

import numpy as np

class RecordManager:

    def __init__(self, bm: BufManager):
        self.BM = bm
        self.opened = {}

    def createFile(self, name: str, recordLen: int):
        self.BM.FM.createFile(name)
        fileID = self.BM.openFile(name)

        self.BM.FM.newPage(fileID, self.toSerial(self.getHead(recordLen, name)))
        self.BM.closeFile(fileID)
        return

    def getHead(self, recordLen: int, name: str):
        recordNum = self.getRecordNum(recordLen)
        bitmapLen = self.getBitmapLen(recordNum)
        return {'RecordLen': recordLen, 'RecordNum': recordNum, 'PageNum': 1,
                'AllRecord': 0, 'NextAvai': 0, 'BitmapLen': bitmapLen, 'filename': str(name)}

    def openFile(self, name: str):
        if name in self.opened:
            handler = self.opened[name]
            return handler
        fID = self.BM.openFile(name)
        self.opened[name] = FileHandler(self, fID, name)
        return self.opened[name]

    def destroyFile(self, name: str):
        self.BM.FM.destroyFile(name)
        return

    def renameFile(self, src: str, dst: str):
        if self.opened.get(src) is None:
            self.BM.FM.renameFile(src, dst)
        else:
            self.closeFile(src)
            self.BM.FM.renameFile(src, dst)
        return

    def closeFile(self, name: str):
        if self.opened.get(name) is None:
            return
        handler = self.opened.get(name)
        if handler.headChanged:
            handler.changeHead()
        self.BM.closeFile(handler.fileID)
        self.opened.pop(name)
        handler.open = False
        return

    @staticmethod
    def getRecordNum(recordLen: int):
        nohead = PAGE_SIZE - RECORD_PAGE_FIXED_HEADER
        num = (nohead * 8) // (1 + (recordLen * 8)) + 1
        total = ((num + 7) / 8) + num * recordLen
        while total > nohead:
            num = num - 1
            total = ((num + 7) / 8) + num * recordLen
        if num <= 0:
            print("OH NO")
            raise RecordTooLong("record too long")
        return num

    @staticmethod
    def getBitmapLen(recordNum: int) -> int:
        length = (recordNum + 7) / 8
        return int(length)


    def replaceFile(self, src: str, dst: str):
        if self.opened.get(src) is not None:
            self.closeFile(src)
        if self.opened.get(dst) is not None:
            self.closeFile(dst)
        self.destroyFile(dst)
        self.BM.FM.renameFile(src, dst)
        return

    def shutdown(self):
        for name in tuple(self.opened.keys()):
            self.closeFile(name)

    @staticmethod
    def toSerial(d: dict):
        serial = dumps(d, ensure_ascii=False).encode('utf-8')
        empty = np.zeros(PAGE_SIZE, dtype=np.uint8)
        for i in range(len(serial)):
            empty[i] = list(serial)[i]
        return empty
PAGE_SIZE = 8192

PAGE_INT_NUM = 2048

PAGE_SIZE_IDX = 13
MAX_FMT_INT_NUM = 128
BUF_PAGE_NUM = 65536
MAX_FILE_NUM = 128
MAX_TYPE_NUM = 256


RECORD_PAGE_NEXT_OFFSET = 1

RECORD_PAGE_FIXED_HEADER = RECORD_PAGE_NEXT_OFFSET + 4


PAGE_FLAG_OFFSET = 0

RECORD_PAGE_FLAG = 0

VARCHAR_PAGE_FLAG = 1

IN_DEBUG = 0
DEBUG_DELETE = 0
DEBUG_ERASE = 1
DEBUG_NEXT = 1

MAX_COL_NUM = 31

MAX_TB_NUM = 31
RELEASE = 1
