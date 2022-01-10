
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
# Generated from /Users/liuxinghan/IdeaProjects/test/SQL.g4 by ANTLR 4.9.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .SQLParser import SQLParser
else:
    from SQLParser import SQLParser

# This class defines a complete generic visitor for a parse tree produced by SQLParser.

class SQLVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by SQLParser#program.
    def visitProgram(self, ctx:SQLParser.ProgramContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#statement.
    def visitStatement(self, ctx:SQLParser.StatementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#system_statement.
    def visitSystem_statement(self, ctx:SQLParser.System_statementContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#create_db.
    def visitCreate_db(self, ctx:SQLParser.Create_dbContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#drop_db.
    def visitDrop_db(self, ctx:SQLParser.Drop_dbContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#show_dbs.
    def visitShow_dbs(self, ctx:SQLParser.Show_dbsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#use_db.
    def visitUse_db(self, ctx:SQLParser.Use_dbContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#show_tables.
    def visitShow_tables(self, ctx:SQLParser.Show_tablesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#show_indexes.
    def visitShow_indexes(self, ctx:SQLParser.Show_indexesContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#load_data.
    def visitLoad_data(self, ctx:SQLParser.Load_dataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#dump_data.
    def visitDump_data(self, ctx:SQLParser.Dump_dataContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#create_table.
    def visitCreate_table(self, ctx:SQLParser.Create_tableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#drop_table.
    def visitDrop_table(self, ctx:SQLParser.Drop_tableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#describe_table.
    def visitDescribe_table(self, ctx:SQLParser.Describe_tableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#insert_into_table.
    def visitInsert_into_table(self, ctx:SQLParser.Insert_into_tableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#delete_from_table.
    def visitDelete_from_table(self, ctx:SQLParser.Delete_from_tableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#update_table.
    def visitUpdate_table(self, ctx:SQLParser.Update_tableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#select_table_.
    def visitSelect_table_(self, ctx:SQLParser.Select_table_Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#select_table.
    def visitSelect_table(self, ctx:SQLParser.Select_tableContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#create_index.
    def visitCreate_index(self, ctx:SQLParser.Create_indexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#drop_index.
    def visitDrop_index(self, ctx:SQLParser.Drop_indexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_add_index.
    def visitAlter_add_index(self, ctx:SQLParser.Alter_add_indexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_drop_index.
    def visitAlter_drop_index(self, ctx:SQLParser.Alter_drop_indexContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_table_add.
    def visitAlter_table_add(self, ctx:SQLParser.Alter_table_addContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_table_drop.
    def visitAlter_table_drop(self, ctx:SQLParser.Alter_table_dropContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_table_change.
    def visitAlter_table_change(self, ctx:SQLParser.Alter_table_changeContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_table_rename.
    def visitAlter_table_rename(self, ctx:SQLParser.Alter_table_renameContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_table_drop_pk.
    def visitAlter_table_drop_pk(self, ctx:SQLParser.Alter_table_drop_pkContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_table_drop_foreign_key.
    def visitAlter_table_drop_foreign_key(self, ctx:SQLParser.Alter_table_drop_foreign_keyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_table_add_pk.
    def visitAlter_table_add_pk(self, ctx:SQLParser.Alter_table_add_pkContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_table_add_foreign_key.
    def visitAlter_table_add_foreign_key(self, ctx:SQLParser.Alter_table_add_foreign_keyContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#alter_table_add_unique.
    def visitAlter_table_add_unique(self, ctx:SQLParser.Alter_table_add_uniqueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#field_list.
    def visitField_list(self, ctx:SQLParser.Field_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#normal_field.
    def visitNormal_field(self, ctx:SQLParser.Normal_fieldContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#primary_key_field.
    def visitPrimary_key_field(self, ctx:SQLParser.Primary_key_fieldContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#foreign_key_field.
    def visitForeign_key_field(self, ctx:SQLParser.Foreign_key_fieldContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#type_.
    def visitType_(self, ctx:SQLParser.Type_Context):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#value_lists.
    def visitValue_lists(self, ctx:SQLParser.Value_listsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#value_list.
    def visitValue_list(self, ctx:SQLParser.Value_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#value.
    def visitValue(self, ctx:SQLParser.ValueContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#where_and_clause.
    def visitWhere_and_clause(self, ctx:SQLParser.Where_and_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#where_operator_expression.
    def visitWhere_operator_expression(self, ctx:SQLParser.Where_operator_expressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#where_operator_select.
    def visitWhere_operator_select(self, ctx:SQLParser.Where_operator_selectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#where_null.
    def visitWhere_null(self, ctx:SQLParser.Where_nullContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#where_in_list.
    def visitWhere_in_list(self, ctx:SQLParser.Where_in_listContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#where_in_select.
    def visitWhere_in_select(self, ctx:SQLParser.Where_in_selectContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#where_like_string.
    def visitWhere_like_string(self, ctx:SQLParser.Where_like_stringContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#column.
    def visitColumn(self, ctx:SQLParser.ColumnContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#expression.
    def visitExpression(self, ctx:SQLParser.ExpressionContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#set_clause.
    def visitSet_clause(self, ctx:SQLParser.Set_clauseContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#selectors.
    def visitSelectors(self, ctx:SQLParser.SelectorsContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#selector.
    def visitSelector(self, ctx:SQLParser.SelectorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#identifiers.
    def visitIdentifiers(self, ctx:SQLParser.IdentifiersContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#operator.
    def visitOperator(self, ctx:SQLParser.OperatorContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by SQLParser#aggregator.
    def visitAggregator(self, ctx:SQLParser.AggregatorContext):
        return self.visitChildren(ctx)



del SQLParsertoken literal names:
null
';'
'SHOW'
'DATABASES'
'CREATE'
'DATABASE'
'DROP'
'USE'
'TABLES'
'INDEXES'
'LOAD'
'FROM'
'FILE'
'TO'
'TABLE'
'DUMP'
'('
')'
'DESC'
'INSERT'
'INTO'
'VALUES'
'DELETE'
'WHERE'
'UPDATE'
'SET'
'SELECT'
'GROUP'
'BY'
'LIMIT'
'OFFSET'
'INDEX'
'ON'
'ALTER'
'ADD'
'CHANGE'
'RENAME'
'PRIMARY'
'KEY'
'FOREIGN'
'CONSTRAINT'
'REFERENCES'
'UNIQUE'
','
'NOT'
'DEFAULT'
'INT'
'VARCHAR'
'DATE'
'FLOAT'
'AND'
'IS'
'IN'
'LIKE'
'.'
'*'
'='
'<'
'<='
'>'
'>='
'<>'
'COUNT'
'AVG'
'MAX'
'MIN'
'SUM'
'NULL'
null
null
null
null
null
null

token symbolic names:
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
EqualOrAssign
Less
LessEqual
Greater
GreaterEqual
NotEqual
Count
Average
Max
Min
Sum
Null
Identifier
Integer
String
Float
Whitespace
Annotation

rule names:
T__0
T__1
T__2
T__3
T__4
T__5
T__6
T__7
T__8
T__9
T__10
T__11
T__12
T__13
T__14
T__15
T__16
T__17
T__18
T__19
T__20
T__21
T__22
T__23
T__24
T__25
T__26
T__27
T__28
T__29
T__30
T__31
T__32
T__33
T__34
T__35
T__36
T__37
T__38
T__39
T__40
T__41
T__42
T__43
T__44
T__45
T__46
T__47
T__48
T__49
T__50
T__51
T__52
T__53
T__54
EqualOrAssign
Less
LessEqual
Greater
GreaterEqual
NotEqual
Count
Average
Max
Min
Sum
Null
Identifier
Integer
String
Float
Whitespace
Annotation

channel names:
DEFAULT_TOKEN_CHANNEL
HIDDEN

mode names:
DEFAULT_MODE

atn:
[3, 24715, 42794, 33075, 47597, 16764, 15335, 30598, 22884, 2, 75, 546, 8, 1, 4, 2, 9, 2, 4, 3, 9, 3, 4, 4, 9, 4, 4, 5, 9, 5, 4, 6, 9, 6, 4, 7, 9, 7, 4, 8, 9, 8, 4, 9, 9, 9, 4, 10, 9, 10, 4, 11, 9, 11, 4, 12, 9, 12, 4, 13, 9, 13, 4, 14, 9, 14, 4, 15, 9, 15, 4, 16, 9, 16, 4, 17, 9, 17, 4, 18, 9, 18, 4, 19, 9, 19, 4, 20, 9, 20, 4, 21, 9, 21, 4, 22, 9, 22, 4, 23, 9, 23, 4, 24, 9, 24, 4, 25, 9, 25, 4, 26, 9, 26, 4, 27, 9, 27, 4, 28, 9, 28, 4, 29, 9, 29, 4, 30, 9, 30, 4, 31, 9, 31, 4, 32, 9, 32, 4, 33, 9, 33, 4, 34, 9, 34, 4, 35, 9, 35, 4, 36, 9, 36, 4, 37, 9, 37, 4, 38, 9, 38, 4, 39, 9, 39, 4, 40, 9, 40, 4, 41, 9, 41, 4, 42, 9, 42, 4, 43, 9, 43, 4, 44, 9, 44, 4, 45, 9, 45, 4, 46, 9, 46, 4, 47, 9, 47, 4, 48, 9, 48, 4, 49, 9, 49, 4, 50, 9, 50, 4, 51, 9, 51, 4, 52, 9, 52, 4, 53, 9, 53, 4, 54, 9, 54, 4, 55, 9, 55, 4, 56, 9, 56, 4, 57, 9, 57, 4, 58, 9, 58, 4, 59, 9, 59, 4, 60, 9, 60, 4, 61, 9, 61, 4, 62, 9, 62, 4, 63, 9, 63, 4, 64, 9, 64, 4, 65, 9, 65, 4, 66, 9, 66, 4, 67, 9, 67, 4, 68, 9, 68, 4, 69, 9, 69, 4, 70, 9, 70, 4, 71, 9, 71, 4, 72, 9, 72, 4, 73, 9, 73, 4, 74, 9, 74, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 4, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 8, 3, 8, 3, 8, 3, 8, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 11, 3, 11, 3, 11, 3, 11, 3, 11, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 14, 3, 14, 3, 14, 3, 15, 3, 15, 3, 15, 3, 15, 3, 15, 3, 15, 3, 16, 3, 16, 3, 16, 3, 16, 3, 16, 3, 17, 3, 17, 3, 18, 3, 18, 3, 19, 3, 19, 3, 19, 3, 19, 3, 19, 3, 20, 3, 20, 3, 20, 3, 20, 3, 20, 3, 20, 3, 20, 3, 21, 3, 21, 3, 21, 3, 21, 3, 21, 3, 22, 3, 22, 3, 22, 3, 22, 3, 22, 3, 22, 3, 22, 3, 23, 3, 23, 3, 23, 3, 23, 3, 23, 3, 23, 3, 23, 3, 24, 3, 24, 3, 24, 3, 24, 3, 24, 3, 24, 3, 25, 3, 25, 3, 25, 3, 25, 3, 25, 3, 25, 3, 25, 3, 26, 3, 26, 3, 26, 3, 26, 3, 27, 3, 27, 3, 27, 3, 27, 3, 27, 3, 27, 3, 27, 3, 28, 3, 28, 3, 28, 3, 28, 3, 28, 3, 28, 3, 29, 3, 29, 3, 29, 3, 30, 3, 30, 3, 30, 3, 30, 3, 30, 3, 30, 3, 31, 3, 31, 3, 31, 3, 31, 3, 31, 3, 31, 3, 31, 3, 32, 3, 32, 3, 32, 3, 32, 3, 32, 3, 32, 3, 33, 3, 33, 3, 33, 3, 34, 3, 34, 3, 34, 3, 34, 3, 34, 3, 34, 3, 35, 3, 35, 3, 35, 3, 35, 3, 36, 3, 36, 3, 36, 3, 36, 3, 36, 3, 36, 3, 36, 3, 37, 3, 37, 3, 37, 3, 37, 3, 37, 3, 37, 3, 37, 3, 38, 3, 38, 3, 38, 3, 38, 3, 38, 3, 38, 3, 38, 3, 38, 3, 39, 3, 39, 3, 39, 3, 39, 3, 40, 3, 40, 3, 40, 3, 40, 3, 40, 3, 40, 3, 40, 3, 40, 3, 41, 3, 41, 3, 41, 3, 41, 3, 41, 3, 41, 3, 41, 3, 41, 3, 41, 3, 41, 3, 41, 3, 42, 3, 42, 3, 42, 3, 42, 3, 42, 3, 42, 3, 42, 3, 42, 3, 42, 3, 42, 3, 42, 3, 43, 3, 43, 3, 43, 3, 43, 3, 43, 3, 43, 3, 43, 3, 44, 3, 44, 3, 45, 3, 45, 3, 45, 3, 45, 3, 46, 3, 46, 3, 46, 3, 46, 3, 46, 3, 46, 3, 46, 3, 46, 3, 47, 3, 47, 3, 47, 3, 47, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 48, 3, 49, 3, 49, 3, 49, 3, 49, 3, 49, 3, 50, 3, 50, 3, 50, 3, 50, 3, 50, 3, 50, 3, 51, 3, 51, 3, 51, 3, 51, 3, 52, 3, 52, 3, 52, 3, 53, 3, 53, 3, 53, 3, 54, 3, 54, 3, 54, 3, 54, 3, 54, 3, 55, 3, 55, 3, 56, 3, 56, 3, 57, 3, 57, 3, 58, 3, 58, 3, 59, 3, 59, 3, 59, 3, 60, 3, 60, 3, 61, 3, 61, 3, 61, 3, 62, 3, 62, 3, 62, 3, 63, 3, 63, 3, 63, 3, 63, 3, 63, 3, 63, 3, 64, 3, 64, 3, 64, 3, 64, 3, 65, 3, 65, 3, 65, 3, 65, 3, 66, 3, 66, 3, 66, 3, 66, 3, 67, 3, 67, 3, 67, 3, 67, 3, 68, 3, 68, 3, 68, 3, 68, 3, 68, 3, 69, 3, 69, 7, 69, 499, 10, 69, 12, 69, 14, 69, 502, 11, 69, 3, 70, 6, 70, 505, 10, 70, 13, 70, 14, 70, 506, 3, 71, 3, 71, 7, 71, 511, 10, 71, 12, 71, 14, 71, 514, 11, 71, 3, 71, 3, 71, 3, 72, 5, 72, 519, 10, 72, 3, 72, 6, 72, 522, 10, 72, 13, 72, 14, 72, 523, 3, 72, 3, 72, 7, 72, 528, 10, 72, 12, 72, 14, 72, 531, 11, 72, 3, 73, 6, 73, 534, 10, 73, 13, 73, 14, 73, 535, 3, 73, 3, 73, 3, 74, 3, 74, 3, 74, 6, 74, 543, 10, 74, 13, 74, 14, 74, 544, 2, 2, 75, 3, 3, 5, 4, 7, 5, 9, 6, 11, 7, 13, 8, 15, 9, 17, 10, 19, 11, 21, 12, 23, 13, 25, 14, 27, 15, 29, 16, 31, 17, 33, 18, 35, 19, 37, 20, 39, 21, 41, 22, 43, 23, 45, 24, 47, 25, 49, 26, 51, 27, 53, 28, 55, 29, 57, 30, 59, 31, 61, 32, 63, 33, 65, 34, 67, 35, 69, 36, 71, 37, 73, 38, 75, 39, 77, 40, 79, 41, 81, 42, 83, 43, 85, 44, 87, 45, 89, 46, 91, 47, 93, 48, 95, 49, 97, 50, 99, 51, 101, 52, 103, 53, 105, 54, 107, 55, 109, 56, 111, 57, 113, 58, 115, 59, 117, 60, 119, 61, 121, 62, 123, 63, 125, 64, 127, 65, 129, 66, 131, 67, 133, 68, 135, 69, 137, 70, 139, 71, 141, 72, 143, 73, 145, 74, 147, 75, 3, 2, 8, 5, 2, 67, 92, 97, 97, 99, 124, 6, 2, 50, 59, 67, 92, 97, 97, 99, 124, 3, 2, 50, 59, 3, 2, 41, 41, 5, 2, 11, 12, 15, 15, 34, 34, 3, 2, 61, 61, 2, 553, 2, 3, 3, 2, 2, 2, 2, 5, 3, 2, 2, 2, 2, 7, 3, 2, 2, 2, 2, 9, 3, 2, 2, 2, 2, 11, 3, 2, 2, 2, 2, 13, 3, 2, 2, 2, 2, 15, 3, 2, 2, 2, 2, 17, 3, 2, 2, 2, 2, 19, 3, 2, 2, 2, 2, 21, 3, 2, 2, 2, 2, 23, 3, 2, 2, 2, 2, 25, 3, 2, 2, 2, 2, 27, 3, 2, 2, 2, 2, 29, 3, 2, 2, 2, 2, 31, 3, 2, 2, 2, 2, 33, 3, 2, 2, 2, 2, 35, 3, 2, 2, 2, 2, 37, 3, 2, 2, 2, 2, 39, 3, 2, 2, 2, 2, 41, 3, 2, 2, 2, 2, 43, 3, 2, 2, 2, 2, 45, 3, 2, 2, 2, 2, 47, 3, 2, 2, 2, 2, 49, 3, 2, 2, 2, 2, 51, 3, 2, 2, 2, 2, 53, 3, 2, 2, 2, 2, 55, 3, 2, 2, 2, 2, 57, 3, 2, 2, 2, 2, 59, 3, 2, 2, 2, 2, 61, 3, 2, 2, 2, 2, 63, 3, 2, 2, 2, 2, 65, 3, 2, 2, 2, 2, 67, 3, 2, 2, 2, 2, 69, 3, 2, 2, 2, 2, 71, 3, 2, 2, 2, 2, 73, 3, 2, 2, 2, 2, 75, 3, 2, 2, 2, 2, 77, 3, 2, 2, 2, 2, 79, 3, 2, 2, 2, 2, 81, 3, 2, 2, 2, 2, 83, 3, 2, 2, 2, 2, 85, 3, 2, 2, 2, 2, 87, 3, 2, 2, 2, 2, 89, 3, 2, 2, 2, 2, 91, 3, 2, 2, 2, 2, 93, 3, 2, 2, 2, 2, 95, 3, 2, 2, 2, 2, 97, 3, 2, 2, 2, 2, 99, 3, 2, 2, 2, 2, 101, 3, 2, 2, 2, 2, 103, 3, 2, 2, 2, 2, 105, 3, 2, 2, 2, 2, 107, 3, 2, 2, 2, 2, 109, 3, 2, 2, 2, 2, 111, 3, 2, 2, 2, 2, 113, 3, 2, 2, 2, 2, 115, 3, 2, 2, 2, 2, 117, 3, 2, 2, 2, 2, 119, 3, 2, 2, 2, 2, 121, 3, 2, 2, 2, 2, 123, 3, 2, 2, 2, 2, 125, 3, 2, 2, 2, 2, 127, 3, 2, 2, 2, 2, 129, 3, 2, 2, 2, 2, 131, 3, 2, 2, 2, 2, 133, 3, 2, 2, 2, 2, 135, 3, 2, 2, 2, 2, 137, 3, 2, 2, 2, 2, 139, 3, 2, 2, 2, 2, 141, 3, 2, 2, 2, 2, 143, 3, 2, 2, 2, 2, 145, 3, 2, 2, 2, 2, 147, 3, 2, 2, 2, 3, 149, 3, 2, 2, 2, 5, 151, 3, 2, 2, 2, 7, 156, 3, 2, 2, 2, 9, 166, 3, 2, 2, 2, 11, 173, 3, 2, 2, 2, 13, 182, 3, 2, 2, 2, 15, 187, 3, 2, 2, 2, 17, 191, 3, 2, 2, 2, 19, 198, 3, 2, 2, 2, 21, 206, 3, 2, 2, 2, 23, 211, 3, 2, 2, 2, 25, 216, 3, 2, 2, 2, 27, 221, 3, 2, 2, 2, 29, 224, 3, 2, 2, 2, 31, 230, 3, 2, 2, 2, 33, 235, 3, 2, 2, 2, 35, 237, 3, 2, 2, 2, 37, 239, 3, 2, 2, 2, 39, 244, 3, 2, 2, 2, 41, 251, 3, 2, 2, 2, 43, 256, 3, 2, 2, 2, 45, 263, 3, 2, 2, 2, 47, 270, 3, 2, 2, 2, 49, 276, 3, 2, 2, 2, 51, 283, 3, 2, 2, 2, 53, 287, 3, 2, 2, 2, 55, 294, 3, 2, 2, 2, 57, 300, 3, 2, 2, 2, 59, 303, 3, 2, 2, 2, 61, 309, 3, 2, 2, 2, 63, 316, 3, 2, 2, 2, 65, 322, 3, 2, 2, 2, 67, 325, 3, 2, 2, 2, 69, 331, 3, 2, 2, 2, 71, 335, 3, 2, 2, 2, 73, 342, 3, 2, 2, 2, 75, 349, 3, 2, 2, 2, 77, 357, 3, 2, 2, 2, 79, 361, 3, 2, 2, 2, 81, 369, 3, 2, 2, 2, 83, 380, 3, 2, 2, 2, 85, 391, 3, 2, 2, 2, 87, 398, 3, 2, 2, 2, 89, 400, 3, 2, 2, 2, 91, 404, 3, 2, 2, 2, 93, 412, 3, 2, 2, 2, 95, 416, 3, 2, 2, 2, 97, 424, 3, 2, 2, 2, 99, 429, 3, 2, 2, 2, 101, 435, 3, 2, 2, 2, 103, 439, 3, 2, 2, 2, 105, 442, 3, 2, 2, 2, 107, 445, 3, 2, 2, 2, 109, 450, 3, 2, 2, 2, 111, 452, 3, 2, 2, 2, 113, 454, 3, 2, 2, 2, 115, 456, 3, 2, 2, 2, 117, 458, 3, 2, 2, 2, 119, 461, 3, 2, 2, 2, 121, 463, 3, 2, 2, 2, 123, 466, 3, 2, 2, 2, 125, 469, 3, 2, 2, 2, 127, 475, 3, 2, 2, 2, 129, 479, 3, 2, 2, 2, 131, 483, 3, 2, 2, 2, 133, 487, 3, 2, 2, 2, 135, 491, 3, 2, 2, 2, 137, 496, 3, 2, 2, 2, 139, 504, 3, 2, 2, 2, 141, 508, 3, 2, 2, 2, 143, 518, 3, 2, 2, 2, 145, 533, 3, 2, 2, 2, 147, 539, 3, 2, 2, 2, 149, 150, 7, 61, 2, 2, 150, 4, 3, 2, 2, 2, 151, 152, 7, 85, 2, 2, 152, 153, 7, 74, 2, 2, 153, 154, 7, 81, 2, 2, 154, 155, 7, 89, 2, 2, 155, 6, 3, 2, 2, 2, 156, 157, 7, 70, 2, 2, 157, 158, 7, 67, 2, 2, 158, 159, 7, 86, 2, 2, 159, 160, 7, 67, 2, 2, 160, 161, 7, 68, 2, 2, 161, 162, 7, 67, 2, 2, 162, 163, 7, 85, 2, 2, 163, 164, 7, 71, 2, 2, 164, 165, 7, 85, 2, 2, 165, 8, 3, 2, 2, 2, 166, 167, 7, 69, 2, 2, 167, 168, 7, 84, 2, 2, 168, 169, 7, 71, 2, 2, 169, 170, 7, 67, 2, 2, 170, 171, 7, 86, 2, 2, 171, 172, 7, 71, 2, 2, 172, 10, 3, 2, 2, 2, 173, 174, 7, 70, 2, 2, 174, 175, 7, 67, 2, 2, 175, 176, 7, 86, 2, 2, 176, 177, 7, 67, 2, 2, 177, 178, 7, 68, 2, 2, 178, 179, 7, 67, 2, 2, 179, 180, 7, 85, 2, 2, 180, 181, 7, 71, 2, 2, 181, 12, 3, 2, 2, 2, 182, 183, 7, 70, 2, 2, 183, 184, 7, 84, 2, 2, 184, 185, 7, 81, 2, 2, 185, 186, 7, 82, 2, 2, 186, 14, 3, 2, 2, 2, 187, 188, 7, 87, 2, 2, 188, 189, 7, 85, 2, 2, 189, 190, 7, 71, 2, 2, 190, 16, 3, 2, 2, 2, 191, 192, 7, 86, 2, 2, 192, 193, 7, 67, 2, 2, 193, 194, 7, 68, 2, 2, 194, 195, 7, 78, 2, 2, 195, 196, 7, 71, 2, 2, 196, 197, 7, 85, 2, 2, 197, 18, 3, 2, 2, 2, 198, 199, 7, 75, 2, 2, 199, 200, 7, 80, 2, 2, 200, 201, 7, 70, 2, 2, 201, 202, 7, 71, 2, 2, 202, 203, 7, 90, 2, 2, 203, 204, 7, 71, 2, 2, 204, 205, 7, 85, 2, 2, 205, 20, 3, 2, 2, 2, 206, 207, 7, 78, 2, 2, 207, 208, 7, 81, 2, 2, 208, 209, 7, 67, 2, 2, 209, 210, 7, 70, 2, 2, 210, 22, 3, 2, 2, 2, 211, 212, 7, 72, 2, 2, 212, 213, 7, 84, 2, 2, 213, 214, 7, 81, 2, 2, 214, 215, 7, 79, 2, 2, 215, 24, 3, 2, 2, 2, 216, 217, 7, 72, 2, 2, 217, 218, 7, 75, 2, 2, 218, 219, 7, 78, 2, 2, 219, 220, 7, 71, 2, 2, 220, 26, 3, 2, 2, 2, 221, 222, 7, 86, 2, 2, 222, 223, 7, 81, 2, 2, 223, 28, 3, 2, 2, 2, 224, 225, 7, 86, 2, 2, 225, 226, 7, 67, 2, 2, 226, 227, 7, 68, 2, 2, 227, 228, 7, 78, 2, 2, 228, 229, 7, 71, 2, 2, 229, 30, 3, 2, 2, 2, 230, 231, 7, 70, 2, 2, 231, 232, 7, 87, 2, 2, 232, 233, 7, 79, 2, 2, 233, 234, 7, 82, 2, 2, 234, 32, 3, 2, 2, 2, 235, 236, 7, 42, 2, 2, 236, 34, 3, 2, 2, 2, 237, 238, 7, 43, 2, 2, 238, 36, 3, 2, 2, 2, 239, 240, 7, 70, 2, 2, 240, 241, 7, 71, 2, 2, 241, 242, 7, 85, 2, 2, 242, 243, 7, 69, 2, 2, 243, 38, 3, 2, 2, 2, 244, 245, 7, 75, 2, 2, 245, 246, 7, 80, 2, 2, 246, 247, 7, 85, 2, 2, 247, 248, 7, 71, 2, 2, 248, 249, 7, 84, 2, 2, 249, 250, 7, 86, 2, 2, 250, 40, 3, 2, 2, 2, 251, 252, 7, 75, 2, 2, 252, 253, 7, 80, 2, 2, 253, 254, 7, 86, 2, 2, 254, 255, 7, 81, 2, 2, 255, 42, 3, 2, 2, 2, 256, 257, 7, 88, 2, 2, 257, 258, 7, 67, 2, 2, 258, 259, 7, 78, 2, 2, 259, 260, 7, 87, 2, 2, 260, 261, 7, 71, 2, 2, 261, 262, 7, 85, 2, 2, 262, 44, 3, 2, 2, 2, 263, 264, 7, 70, 2, 2, 264, 265, 7, 71, 2, 2, 265, 266, 7, 78, 2, 2, 266, 267, 7, 71, 2, 2, 267, 268, 7, 86, 2, 2, 268, 269, 7, 71, 2, 2, 269, 46, 3, 2, 2, 2, 270, 271, 7, 89, 2, 2, 271, 272, 7, 74, 2, 2, 272, 273, 7, 71, 2, 2, 273, 274, 7, 84, 2, 2, 274, 275, 7, 71, 2, 2, 275, 48, 3, 2, 2, 2, 276, 277, 7, 87, 2, 2, 277, 278, 7, 82, 2, 2, 278, 279, 7, 70, 2, 2, 279, 280, 7, 67, 2, 2, 280, 281, 7, 86, 2, 2, 281, 282, 7, 71, 2, 2, 282, 50, 3, 2, 2, 2, 283, 284, 7, 85, 2, 2, 284, 285, 7, 71, 2, 2, 285, 286, 7, 86, 2, 2, 286, 52, 3, 2, 2, 2, 287, 288, 7, 85, 2, 2, 288, 289, 7, 71, 2, 2, 289, 290, 7, 78, 2, 2, 290, 291, 7, 71, 2, 2, 291, 292, 7, 69, 2, 2, 292, 293, 7, 86, 2, 2, 293, 54, 3, 2, 2, 2, 294, 295, 7, 73, 2, 2, 295, 296, 7, 84, 2, 2, 296, 297, 7, 81, 2, 2, 297, 298, 7, 87, 2, 2, 298, 299, 7, 82, 2, 2, 299, 56, 3, 2, 2, 2, 300, 301, 7, 68, 2, 2, 301, 302, 7, 91, 2, 2, 302, 58, 3, 2, 2, 2, 303, 304, 7, 78, 2, 2, 304, 305, 7, 75, 2, 2, 305, 306, 7, 79, 2, 2, 306, 307, 7, 75, 2, 2, 307, 308, 7, 86, 2, 2, 308, 60, 3, 2, 2, 2, 309, 310, 7, 81, 2, 2, 310, 311, 7, 72, 2, 2, 311, 312, 7, 72, 2, 2, 312, 313, 7, 85, 2, 2, 313, 314, 7, 71, 2, 2, 314, 315, 7, 86, 2, 2, 315, 62, 3, 2, 2, 2, 316, 317, 7, 75, 2, 2, 317, 318, 7, 80, 2, 2, 318, 319, 7, 70, 2, 2, 319, 320, 7, 71, 2, 2, 320, 321, 7, 90, 2, 2, 321, 64, 3, 2, 2, 2, 322, 323, 7, 81, 2, 2, 323, 324, 7, 80, 2, 2, 324, 66, 3, 2, 2, 2, 325, 326, 7, 67, 2, 2, 326, 327, 7, 78, 2, 2, 327, 328, 7, 86, 2, 2, 328, 329, 7, 71, 2, 2, 329, 330, 7, 84, 2, 2, 330, 68, 3, 2, 2, 2, 331, 332, 7, 67, 2, 2, 332, 333, 7, 70, 2, 2, 333, 334, 7, 70, 2, 2, 334, 70, 3, 2, 2, 2, 335, 336, 7, 69, 2, 2, 336, 337, 7, 74, 2, 2, 337, 338, 7, 67, 2, 2, 338, 339, 7, 80, 2, 2, 339, 340, 7, 73, 2, 2, 340, 341, 7, 71, 2, 2, 341, 72, 3, 2, 2, 2, 342, 343, 7, 84, 2, 2, 343, 344, 7, 71, 2, 2, 344, 345, 7, 80, 2, 2, 345, 346, 7, 67, 2, 2, 346, 347, 7, 79, 2, 2, 347, 348, 7, 71, 2, 2, 348, 74, 3, 2, 2, 2, 349, 350, 7, 82, 2, 2, 350, 351, 7, 84, 2, 2, 351, 352, 7, 75, 2, 2, 352, 353, 7, 79, 2, 2, 353, 354, 7, 67, 2, 2, 354, 355, 7, 84, 2, 2, 355, 356, 7, 91, 2, 2, 356, 76, 3, 2, 2, 2, 357, 358, 7, 77, 2, 2, 358, 359, 7, 71, 2, 2, 359, 360, 7, 91, 2, 2, 360, 78, 3, 2, 2, 2, 361, 362, 7, 72, 2, 2, 362, 363, 7, 81, 2, 2, 363, 364, 7, 84, 2, 2, 364, 365, 7, 71, 2, 2, 365, 366, 7, 75, 2, 2, 366, 367, 7, 73, 2, 2, 367, 368, 7, 80, 2, 2, 368, 80, 3, 2, 2, 2, 369, 370, 7, 69, 2, 2, 370, 371, 7, 81, 2, 2, 371, 372, 7, 80, 2, 2, 372, 373, 7, 85, 2, 2, 373, 374, 7, 86, 2, 2, 374, 375, 7, 84, 2, 2, 375, 376, 7, 67, 2, 2, 376, 377, 7, 75, 2, 2, 377, 378, 7, 80, 2, 2, 378, 379, 7, 86, 2, 2, 379, 82, 3, 2, 2, 2, 380, 381, 7, 84, 2, 2, 381, 382, 7, 71, 2, 2, 382, 383, 7, 72, 2, 2, 383, 384, 7, 71, 2, 2, 384, 385, 7, 84, 2, 2, 385, 386, 7, 71, 2, 2, 386, 387, 7, 80, 2, 2, 387, 388, 7, 69, 2, 2, 388, 389, 7, 71, 2, 2, 389, 390, 7, 85, 2, 2, 390, 84, 3, 2, 2, 2, 391, 392, 7, 87, 2, 2, 392, 393, 7, 80, 2, 2, 393, 394, 7, 75, 2, 2, 394, 395, 7, 83, 2, 2, 395, 396, 7, 87, 2, 2, 396, 397, 7, 71, 2, 2, 397, 86, 3, 2, 2, 2, 398, 399, 7, 46, 2, 2, 399, 88, 3, 2, 2, 2, 400, 401, 7, 80, 2, 2, 401, 402, 7, 81, 2, 2, 402, 403, 7, 86, 2, 2, 403, 90, 3, 2, 2, 2, 404, 405, 7, 70, 2, 2, 405, 406, 7, 71, 2, 2, 406, 407, 7, 72, 2, 2, 407, 408, 7, 67, 2, 2, 408, 409, 7, 87, 2, 2, 409, 410, 7, 78, 2, 2, 410, 411, 7, 86, 2, 2, 411, 92, 3, 2, 2, 2, 412, 413, 7, 75, 2, 2, 413, 414, 7, 80, 2, 2, 414, 415, 7, 86, 2, 2, 415, 94, 3, 2, 2, 2, 416, 417, 7, 88, 2, 2, 417, 418, 7, 67, 2, 2, 418, 419, 7, 84, 2, 2, 419, 420, 7, 69, 2, 2, 420, 421, 7, 74, 2, 2, 421, 422, 7, 67, 2, 2, 422, 423, 7, 84, 2, 2, 423, 96, 3, 2, 2, 2, 424, 425, 7, 70, 2, 2, 425, 426, 7, 67, 2, 2, 426, 427, 7, 86, 2, 2, 427, 428, 7, 71, 2, 2, 428, 98, 3, 2, 2, 2, 429, 430, 7, 72, 2, 2, 430, 431, 7, 78, 2, 2, 431, 432, 7, 81, 2, 2, 432, 433, 7, 67, 2, 2, 433, 434, 7, 86, 2, 2, 434, 100, 3, 2, 2, 2, 435, 436, 7, 67, 2, 2, 436, 437, 7, 80, 2, 2, 437, 438, 7, 70, 2, 2, 438, 102, 3, 2, 2, 2, 439, 440, 7, 75, 2, 2, 440, 441, 7, 85, 2, 2, 441, 104, 3, 2, 2, 2, 442, 443, 7, 75, 2, 2, 443, 444, 7, 80, 2, 2, 444, 106, 3, 2, 2, 2, 445, 446, 7, 78, 2, 2, 446, 447, 7, 75, 2, 2, 447, 448, 7, 77, 2, 2, 448, 449, 7, 71, 2, 2, 449, 108, 3, 2, 2, 2, 450, 451, 7, 48, 2, 2, 451, 110, 3, 2, 2, 2, 452, 453, 7, 44, 2, 2, 453, 112, 3, 2, 2, 2, 454, 455, 7, 63, 2, 2, 455, 114, 3, 2, 2, 2, 456, 457, 7, 62, 2, 2, 457, 116, 3, 2, 2, 2, 458, 459, 7, 62, 2, 2, 459, 460, 7, 63, 2, 2, 460, 118, 3, 2, 2, 2, 461, 462, 7, 64, 2, 2, 462, 120, 3, 2, 2, 2, 463, 464, 7, 64, 2, 2, 464, 465, 7, 63, 2, 2, 465, 122, 3, 2, 2, 2, 466, 467, 7, 62, 2, 2, 467, 468, 7, 64, 2, 2, 468, 124, 3, 2, 2, 2, 469, 470, 7, 69, 2, 2, 470, 471, 7, 81, 2, 2, 471, 472, 7, 87, 2, 2, 472, 473, 7, 80, 2, 2, 473, 474, 7, 86, 2, 2, 474, 126, 3, 2, 2, 2, 475, 476, 7, 67, 2, 2, 476, 477, 7, 88, 2, 2, 477, 478, 7, 73, 2, 2, 478, 128, 3, 2, 2, 2, 479, 480, 7, 79, 2, 2, 480, 481, 7, 67, 2, 2, 481, 482, 7, 90, 2, 2, 482, 130, 3, 2, 2, 2, 483, 484, 7, 79, 2, 2, 484, 485, 7, 75, 2, 2, 485, 486, 7, 80, 2, 2, 486, 132, 3, 2, 2, 2, 487, 488, 7, 85, 2, 2, 488, 489, 7, 87, 2, 2, 489, 490, 7, 79, 2, 2, 490, 134, 3, 2, 2, 2, 491, 492, 7, 80, 2, 2, 492, 493, 7, 87, 2, 2, 493, 494, 7, 78, 2, 2, 494, 495, 7, 78, 2, 2, 495, 136, 3, 2, 2, 2, 496, 500, 9, 2, 2, 2, 497, 499, 9, 3, 2, 2, 498, 497, 3, 2, 2, 2, 499, 502, 3, 2, 2, 2, 500, 498, 3, 2, 2, 2, 500, 501, 3, 2, 2, 2, 501, 138, 3, 2, 2, 2, 502, 500, 3, 2, 2, 2, 503, 505, 9, 4, 2, 2, 504, 503, 3, 2, 2, 2, 505, 506, 3, 2, 2, 2, 506, 504, 3, 2, 2, 2, 506, 507, 3, 2, 2, 2, 507, 140, 3, 2, 2, 2, 508, 512, 7, 41, 2, 2, 509, 511, 10, 5, 2, 2, 510, 509, 3, 2, 2, 2, 511, 514, 3, 2, 2, 2, 512, 510, 3, 2, 2, 2, 512, 513, 3, 2, 2, 2, 513, 515, 3, 2, 2, 2, 514, 512, 3, 2, 2, 2, 515, 516, 7, 41, 2, 2, 516, 142, 3, 2, 2, 2, 517, 519, 7, 47, 2, 2, 518, 517, 3, 2, 2, 2, 518, 519, 3, 2, 2, 2, 519, 521, 3, 2, 2, 2, 520, 522, 9, 4, 2, 2, 521, 520, 3, 2, 2, 2, 522, 523, 3, 2, 2, 2, 523, 521, 3, 2, 2, 2, 523, 524, 3, 2, 2, 2, 524, 525, 3, 2, 2, 2, 525, 529, 7, 48, 2, 2, 526, 528, 9, 4, 2, 2, 527, 526, 3, 2, 2, 2, 528, 531, 3, 2, 2, 2, 529, 527, 3, 2, 2, 2, 529, 530, 3, 2, 2, 2, 530, 144, 3, 2, 2, 2, 531, 529, 3, 2, 2, 2, 532, 534, 9, 6, 2, 2, 533, 532, 3, 2, 2, 2, 534, 535, 3, 2, 2, 2, 535, 533, 3, 2, 2, 2, 535, 536, 3, 2, 2, 2, 536, 537, 3, 2, 2, 2, 537, 538, 8, 73, 2, 2, 538, 146, 3, 2, 2, 2, 539, 540, 7, 47, 2, 2, 540, 542, 7, 47, 2, 2, 541, 543, 10, 7, 2, 2, 542, 541, 3, 2, 2, 2, 543, 544, 3, 2, 2, 2, 544, 542, 3, 2, 2, 2, 544, 545, 3, 2, 2, 2, 545, 148, 3, 2, 2, 2, 11, 2, 500, 506, 512, 518, 523, 529, 535, 544, 3, 8, 2, 2]T__0=1
T__1=2
T__2=3
T__3=4
T__4=5
T__5=6
T__6=7
T__7=8
T__8=9
T__9=10
T__10=11
T__11=12
T__12=13
T__13=14
T__14=15
T__15=16
T__16=17
T__17=18
T__18=19
T__19=20
T__20=21
T__21=22
T__22=23
T__23=24
T__24=25
T__25=26
T__26=27
T__27=28
T__28=29
T__29=30
T__30=31
T__31=32
T__32=33
T__33=34
T__34=35
T__35=36
T__36=37
T__37=38
T__38=39
T__39=40
T__40=41
T__41=42
T__42=43
T__43=44
T__44=45
T__45=46
T__46=47
T__47=48
T__48=49
T__49=50
T__50=51
T__51=52
T__52=53
T__53=54
T__54=55
EqualOrAssign=56
Less=57
LessEqual=58
Greater=59
GreaterEqual=60
NotEqual=61
Count=62
Average=63
Max=64
Min=65
Sum=66
Null=67
Identifier=68
Integer=69
String=70
Float=71
Whitespace=72
Annotation=73
';'=1
'SHOW'=2
'DATABASES'=3
'CREATE'=4
'DATABASE'=5
'DROP'=6
'USE'=7
'TABLES'=8
'INDEXES'=9
'LOAD'=10
'FROM'=11
'FILE'=12
'TO'=13
'TABLE'=14
'DUMP'=15
'('=16
')'=17
'DESC'=18
'INSERT'=19
'INTO'=20
'VALUES'=21
'DELETE'=22
'WHERE'=23
'UPDATE'=24
'SET'=25
'SELECT'=26
'GROUP'=27
'BY'=28
'LIMIT'=29
'OFFSET'=30
'INDEX'=31
'ON'=32
'ALTER'=33
'ADD'=34
'CHANGE'=35
'RENAME'=36
'PRIMARY'=37
'KEY'=38
'FOREIGN'=39
'CONSTRAINT'=40
'REFERENCES'=41
'UNIQUE'=42
','=43
'NOT'=44
'DEFAULT'=45
'INT'=46
'VARCHAR'=47
'DATE'=48
'FLOAT'=49
'AND'=50
'IS'=51
'IN'=52
'LIKE'=53
'.'=54
'*'=55
'='=56
'<'=57
'<='=58
'>'=59
'>='=60
'<>'=61
'COUNT'=62
'AVG'=63
'MAX'=64
'MIN'=65
'SUM'=66
'NULL'=67
# Generated from /Users/liuxinghan/IdeaProjects/test/SQL.g4 by ANTLR 4.9.2
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
    from typing import TextIO
else:
    from typing.io import TextIO



def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\2K")
        buf.write("\u0222\b\1\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7")
        buf.write("\t\7\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r\t\r")
        buf.write("\4\16\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22\4\23")
        buf.write("\t\23\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30")
        buf.write("\4\31\t\31\4\32\t\32\4\33\t\33\4\34\t\34\4\35\t\35\4\36")
        buf.write("\t\36\4\37\t\37\4 \t \4!\t!\4\"\t\"\4#\t#\4$\t$\4%\t%")
        buf.write("\4&\t&\4\'\t\'\4(\t(\4)\t)\4*\t*\4+\t+\4,\t,\4-\t-\4.")
        buf.write("\t.\4/\t/\4\60\t\60\4\61\t\61\4\62\t\62\4\63\t\63\4\64")
        buf.write("\t\64\4\65\t\65\4\66\t\66\4\67\t\67\48\t8\49\t9\4:\t:")
        buf.write("\4;\t;\4<\t<\4=\t=\4>\t>\4?\t?\4@\t@\4A\tA\4B\tB\4C\t")
        buf.write("C\4D\tD\4E\tE\4F\tF\4G\tG\4H\tH\4I\tI\4J\tJ\3\2\3\2\3")
        buf.write("\3\3\3\3\3\3\3\3\3\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4\3\4")
        buf.write("\3\4\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\6\3\6\3\6\3\6\3\6\3")
        buf.write("\6\3\6\3\6\3\6\3\7\3\7\3\7\3\7\3\7\3\b\3\b\3\b\3\b\3\t")
        buf.write("\3\t\3\t\3\t\3\t\3\t\3\t\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3")
        buf.write("\n\3\13\3\13\3\13\3\13\3\13\3\f\3\f\3\f\3\f\3\f\3\r\3")
        buf.write("\r\3\r\3\r\3\r\3\16\3\16\3\16\3\17\3\17\3\17\3\17\3\17")
        buf.write("\3\17\3\20\3\20\3\20\3\20\3\20\3\21\3\21\3\22\3\22\3\23")
        buf.write("\3\23\3\23\3\23\3\23\3\24\3\24\3\24\3\24\3\24\3\24\3\24")
        buf.write("\3\25\3\25\3\25\3\25\3\25\3\26\3\26\3\26\3\26\3\26\3\26")
        buf.write("\3\26\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3\30\3\30\3\30")
        buf.write("\3\30\3\30\3\30\3\31\3\31\3\31\3\31\3\31\3\31\3\31\3\32")
        buf.write("\3\32\3\32\3\32\3\33\3\33\3\33\3\33\3\33\3\33\3\33\3\34")
        buf.write("\3\34\3\34\3\34\3\34\3\34\3\35\3\35\3\35\3\36\3\36\3\36")
        buf.write("\3\36\3\36\3\36\3\37\3\37\3\37\3\37\3\37\3\37\3\37\3 ")
        buf.write("\3 \3 \3 \3 \3 \3!\3!\3!\3\"\3\"\3\"\3\"\3\"\3\"\3#\3")
        buf.write("#\3#\3#\3$\3$\3$\3$\3$\3$\3$\3%\3%\3%\3%\3%\3%\3%\3&\3")
        buf.write("&\3&\3&\3&\3&\3&\3&\3\'\3\'\3\'\3\'\3(\3(\3(\3(\3(\3(")
        buf.write("\3(\3(\3)\3)\3)\3)\3)\3)\3)\3)\3)\3)\3)\3*\3*\3*\3*\3")
        buf.write("*\3*\3*\3*\3*\3*\3*\3+\3+\3+\3+\3+\3+\3+\3,\3,\3-\3-\3")
        buf.write("-\3-\3.\3.\3.\3.\3.\3.\3.\3.\3/\3/\3/\3/\3\60\3\60\3\60")
        buf.write("\3\60\3\60\3\60\3\60\3\60\3\61\3\61\3\61\3\61\3\61\3\62")
        buf.write("\3\62\3\62\3\62\3\62\3\62\3\63\3\63\3\63\3\63\3\64\3\64")
        buf.write("\3\64\3\65\3\65\3\65\3\66\3\66\3\66\3\66\3\66\3\67\3\67")
        buf.write("\38\38\39\39\3:\3:\3;\3;\3;\3<\3<\3=\3=\3=\3>\3>\3>\3")
        buf.write("?\3?\3?\3?\3?\3?\3@\3@\3@\3@\3A\3A\3A\3A\3B\3B\3B\3B\3")
        buf.write("C\3C\3C\3C\3D\3D\3D\3D\3D\3E\3E\7E\u01f3\nE\fE\16E\u01f6")
        buf.write("\13E\3F\6F\u01f9\nF\rF\16F\u01fa\3G\3G\7G\u01ff\nG\fG")
        buf.write("\16G\u0202\13G\3G\3G\3H\5H\u0207\nH\3H\6H\u020a\nH\rH")
        buf.write("\16H\u020b\3H\3H\7H\u0210\nH\fH\16H\u0213\13H\3I\6I\u0216")
        buf.write("\nI\rI\16I\u0217\3I\3I\3J\3J\3J\6J\u021f\nJ\rJ\16J\u0220")
        buf.write("\2\2K\3\3\5\4\7\5\t\6\13\7\r\b\17\t\21\n\23\13\25\f\27")
        buf.write("\r\31\16\33\17\35\20\37\21!\22#\23%\24\'\25)\26+\27-\30")
        buf.write("/\31\61\32\63\33\65\34\67\359\36;\37= ?!A\"C#E$G%I&K\'")
        buf.write("M(O)Q*S+U,W-Y.[/]\60_\61a\62c\63e\64g\65i\66k\67m8o9q")
        buf.write(":s;u<w=y>{?}@\177A\u0081B\u0083C\u0085D\u0087E\u0089F")
        buf.write("\u008bG\u008dH\u008fI\u0091J\u0093K\3\2\b\5\2C\\aac|\6")
        buf.write("\2\62;C\\aac|\3\2\62;\3\2))\5\2\13\f\17\17\"\"\3\2==\2")
        buf.write("\u0229\2\3\3\2\2\2\2\5\3\2\2\2\2\7\3\2\2\2\2\t\3\2\2\2")
        buf.write("\2\13\3\2\2\2\2\r\3\2\2\2\2\17\3\2\2\2\2\21\3\2\2\2\2")
        buf.write("\23\3\2\2\2\2\25\3\2\2\2\2\27\3\2\2\2\2\31\3\2\2\2\2\33")
        buf.write("\3\2\2\2\2\35\3\2\2\2\2\37\3\2\2\2\2!\3\2\2\2\2#\3\2\2")
        buf.write("\2\2%\3\2\2\2\2\'\3\2\2\2\2)\3\2\2\2\2+\3\2\2\2\2-\3\2")
        buf.write("\2\2\2/\3\2\2\2\2\61\3\2\2\2\2\63\3\2\2\2\2\65\3\2\2\2")
        buf.write("\2\67\3\2\2\2\29\3\2\2\2\2;\3\2\2\2\2=\3\2\2\2\2?\3\2")
        buf.write("\2\2\2A\3\2\2\2\2C\3\2\2\2\2E\3\2\2\2\2G\3\2\2\2\2I\3")
        buf.write("\2\2\2\2K\3\2\2\2\2M\3\2\2\2\2O\3\2\2\2\2Q\3\2\2\2\2S")
        buf.write("\3\2\2\2\2U\3\2\2\2\2W\3\2\2\2\2Y\3\2\2\2\2[\3\2\2\2\2")
        buf.write("]\3\2\2\2\2_\3\2\2\2\2a\3\2\2\2\2c\3\2\2\2\2e\3\2\2\2")
        buf.write("\2g\3\2\2\2\2i\3\2\2\2\2k\3\2\2\2\2m\3\2\2\2\2o\3\2\2")
        buf.write("\2\2q\3\2\2\2\2s\3\2\2\2\2u\3\2\2\2\2w\3\2\2\2\2y\3\2")
        buf.write("\2\2\2{\3\2\2\2\2}\3\2\2\2\2\177\3\2\2\2\2\u0081\3\2\2")
        buf.write("\2\2\u0083\3\2\2\2\2\u0085\3\2\2\2\2\u0087\3\2\2\2\2\u0089")
        buf.write("\3\2\2\2\2\u008b\3\2\2\2\2\u008d\3\2\2\2\2\u008f\3\2\2")
        buf.write("\2\2\u0091\3\2\2\2\2\u0093\3\2\2\2\3\u0095\3\2\2\2\5\u0097")
        buf.write("\3\2\2\2\7\u009c\3\2\2\2\t\u00a6\3\2\2\2\13\u00ad\3\2")
        buf.write("\2\2\r\u00b6\3\2\2\2\17\u00bb\3\2\2\2\21\u00bf\3\2\2\2")
        buf.write("\23\u00c6\3\2\2\2\25\u00ce\3\2\2\2\27\u00d3\3\2\2\2\31")
        buf.write("\u00d8\3\2\2\2\33\u00dd\3\2\2\2\35\u00e0\3\2\2\2\37\u00e6")
        buf.write("\3\2\2\2!\u00eb\3\2\2\2#\u00ed\3\2\2\2%\u00ef\3\2\2\2")
        buf.write("\'\u00f4\3\2\2\2)\u00fb\3\2\2\2+\u0100\3\2\2\2-\u0107")
        buf.write("\3\2\2\2/\u010e\3\2\2\2\61\u0114\3\2\2\2\63\u011b\3\2")
        buf.write("\2\2\65\u011f\3\2\2\2\67\u0126\3\2\2\29\u012c\3\2\2\2")
        buf.write(";\u012f\3\2\2\2=\u0135\3\2\2\2?\u013c\3\2\2\2A\u0142\3")
        buf.write("\2\2\2C\u0145\3\2\2\2E\u014b\3\2\2\2G\u014f\3\2\2\2I\u0156")
        buf.write("\3\2\2\2K\u015d\3\2\2\2M\u0165\3\2\2\2O\u0169\3\2\2\2")
        buf.write("Q\u0171\3\2\2\2S\u017c\3\2\2\2U\u0187\3\2\2\2W\u018e\3")
        buf.write("\2\2\2Y\u0190\3\2\2\2[\u0194\3\2\2\2]\u019c\3\2\2\2_\u01a0")
        buf.write("\3\2\2\2a\u01a8\3\2\2\2c\u01ad\3\2\2\2e\u01b3\3\2\2\2")
        buf.write("g\u01b7\3\2\2\2i\u01ba\3\2\2\2k\u01bd\3\2\2\2m\u01c2\3")
        buf.write("\2\2\2o\u01c4\3\2\2\2q\u01c6\3\2\2\2s\u01c8\3\2\2\2u\u01ca")
        buf.write("\3\2\2\2w\u01cd\3\2\2\2y\u01cf\3\2\2\2{\u01d2\3\2\2\2")
        buf.write("}\u01d5\3\2\2\2\177\u01db\3\2\2\2\u0081\u01df\3\2\2\2")
        buf.write("\u0083\u01e3\3\2\2\2\u0085\u01e7\3\2\2\2\u0087\u01eb\3")
        buf.write("\2\2\2\u0089\u01f0\3\2\2\2\u008b\u01f8\3\2\2\2\u008d\u01fc")
        buf.write("\3\2\2\2\u008f\u0206\3\2\2\2\u0091\u0215\3\2\2\2\u0093")
        buf.write("\u021b\3\2\2\2\u0095\u0096\7=\2\2\u0096\4\3\2\2\2\u0097")
        buf.write("\u0098\7U\2\2\u0098\u0099\7J\2\2\u0099\u009a\7Q\2\2\u009a")
        buf.write("\u009b\7Y\2\2\u009b\6\3\2\2\2\u009c\u009d\7F\2\2\u009d")
        buf.write("\u009e\7C\2\2\u009e\u009f\7V\2\2\u009f\u00a0\7C\2\2\u00a0")
        buf.write("\u00a1\7D\2\2\u00a1\u00a2\7C\2\2\u00a2\u00a3\7U\2\2\u00a3")
        buf.write("\u00a4\7G\2\2\u00a4\u00a5\7U\2\2\u00a5\b\3\2\2\2\u00a6")
        buf.write("\u00a7\7E\2\2\u00a7\u00a8\7T\2\2\u00a8\u00a9\7G\2\2\u00a9")
        buf.write("\u00aa\7C\2\2\u00aa\u00ab\7V\2\2\u00ab\u00ac\7G\2\2\u00ac")
        buf.write("\n\3\2\2\2\u00ad\u00ae\7F\2\2\u00ae\u00af\7C\2\2\u00af")
        buf.write("\u00b0\7V\2\2\u00b0\u00b1\7C\2\2\u00b1\u00b2\7D\2\2\u00b2")
        buf.write("\u00b3\7C\2\2\u00b3\u00b4\7U\2\2\u00b4\u00b5\7G\2\2\u00b5")
        buf.write("\f\3\2\2\2\u00b6\u00b7\7F\2\2\u00b7\u00b8\7T\2\2\u00b8")
        buf.write("\u00b9\7Q\2\2\u00b9\u00ba\7R\2\2\u00ba\16\3\2\2\2\u00bb")
        buf.write("\u00bc\7W\2\2\u00bc\u00bd\7U\2\2\u00bd\u00be\7G\2\2\u00be")
        buf.write("\20\3\2\2\2\u00bf\u00c0\7V\2\2\u00c0\u00c1\7C\2\2\u00c1")
        buf.write("\u00c2\7D\2\2\u00c2\u00c3\7N\2\2\u00c3\u00c4\7G\2\2\u00c4")
        buf.write("\u00c5\7U\2\2\u00c5\22\3\2\2\2\u00c6\u00c7\7K\2\2\u00c7")
        buf.write("\u00c8\7P\2\2\u00c8\u00c9\7F\2\2\u00c9\u00ca\7G\2\2\u00ca")
        buf.write("\u00cb\7Z\2\2\u00cb\u00cc\7G\2\2\u00cc\u00cd\7U\2\2\u00cd")
        buf.write("\24\3\2\2\2\u00ce\u00cf\7N\2\2\u00cf\u00d0\7Q\2\2\u00d0")
        buf.write("\u00d1\7C\2\2\u00d1\u00d2\7F\2\2\u00d2\26\3\2\2\2\u00d3")
        buf.write("\u00d4\7H\2\2\u00d4\u00d5\7T\2\2\u00d5\u00d6\7Q\2\2\u00d6")
        buf.write("\u00d7\7O\2\2\u00d7\30\3\2\2\2\u00d8\u00d9\7H\2\2\u00d9")
        buf.write("\u00da\7K\2\2\u00da\u00db\7N\2\2\u00db\u00dc\7G\2\2\u00dc")
        buf.write("\32\3\2\2\2\u00dd\u00de\7V\2\2\u00de\u00df\7Q\2\2\u00df")
        buf.write("\34\3\2\2\2\u00e0\u00e1\7V\2\2\u00e1\u00e2\7C\2\2\u00e2")
        buf.write("\u00e3\7D\2\2\u00e3\u00e4\7N\2\2\u00e4\u00e5\7G\2\2\u00e5")
        buf.write("\36\3\2\2\2\u00e6\u00e7\7F\2\2\u00e7\u00e8\7W\2\2\u00e8")
        buf.write("\u00e9\7O\2\2\u00e9\u00ea\7R\2\2\u00ea \3\2\2\2\u00eb")
        buf.write("\u00ec\7*\2\2\u00ec\"\3\2\2\2\u00ed\u00ee\7+\2\2\u00ee")
        buf.write("$\3\2\2\2\u00ef\u00f0\7F\2\2\u00f0\u00f1\7G\2\2\u00f1")
        buf.write("\u00f2\7U\2\2\u00f2\u00f3\7E\2\2\u00f3&\3\2\2\2\u00f4")
        buf.write("\u00f5\7K\2\2\u00f5\u00f6\7P\2\2\u00f6\u00f7\7U\2\2\u00f7")
        buf.write("\u00f8\7G\2\2\u00f8\u00f9\7T\2\2\u00f9\u00fa\7V\2\2\u00fa")
        buf.write("(\3\2\2\2\u00fb\u00fc\7K\2\2\u00fc\u00fd\7P\2\2\u00fd")
        buf.write("\u00fe\7V\2\2\u00fe\u00ff\7Q\2\2\u00ff*\3\2\2\2\u0100")
        buf.write("\u0101\7X\2\2\u0101\u0102\7C\2\2\u0102\u0103\7N\2\2\u0103")
        buf.write("\u0104\7W\2\2\u0104\u0105\7G\2\2\u0105\u0106\7U\2\2\u0106")
        buf.write(",\3\2\2\2\u0107\u0108\7F\2\2\u0108\u0109\7G\2\2\u0109")
        buf.write("\u010a\7N\2\2\u010a\u010b\7G\2\2\u010b\u010c\7V\2\2\u010c")
        buf.write("\u010d\7G\2\2\u010d.\3\2\2\2\u010e\u010f\7Y\2\2\u010f")
        buf.write("\u0110\7J\2\2\u0110\u0111\7G\2\2\u0111\u0112\7T\2\2\u0112")
        buf.write("\u0113\7G\2\2\u0113\60\3\2\2\2\u0114\u0115\7W\2\2\u0115")
        buf.write("\u0116\7R\2\2\u0116\u0117\7F\2\2\u0117\u0118\7C\2\2\u0118")
        buf.write("\u0119\7V\2\2\u0119\u011a\7G\2\2\u011a\62\3\2\2\2\u011b")
        buf.write("\u011c\7U\2\2\u011c\u011d\7G\2\2\u011d\u011e\7V\2\2\u011e")
        buf.write("\64\3\2\2\2\u011f\u0120\7U\2\2\u0120\u0121\7G\2\2\u0121")
        buf.write("\u0122\7N\2\2\u0122\u0123\7G\2\2\u0123\u0124\7E\2\2\u0124")
        buf.write("\u0125\7V\2\2\u0125\66\3\2\2\2\u0126\u0127\7I\2\2\u0127")
        buf.write("\u0128\7T\2\2\u0128\u0129\7Q\2\2\u0129\u012a\7W\2\2\u012a")
        buf.write("\u012b\7R\2\2\u012b8\3\2\2\2\u012c\u012d\7D\2\2\u012d")
        buf.write("\u012e\7[\2\2\u012e:\3\2\2\2\u012f\u0130\7N\2\2\u0130")
        buf.write("\u0131\7K\2\2\u0131\u0132\7O\2\2\u0132\u0133\7K\2\2\u0133")
        buf.write("\u0134\7V\2\2\u0134<\3\2\2\2\u0135\u0136\7Q\2\2\u0136")
        buf.write("\u0137\7H\2\2\u0137\u0138\7H\2\2\u0138\u0139\7U\2\2\u0139")
        buf.write("\u013a\7G\2\2\u013a\u013b\7V\2\2\u013b>\3\2\2\2\u013c")
        buf.write("\u013d\7K\2\2\u013d\u013e\7P\2\2\u013e\u013f\7F\2\2\u013f")
        buf.write("\u0140\7G\2\2\u0140\u0141\7Z\2\2\u0141@\3\2\2\2\u0142")
        buf.write("\u0143\7Q\2\2\u0143\u0144\7P\2\2\u0144B\3\2\2\2\u0145")
        buf.write("\u0146\7C\2\2\u0146\u0147\7N\2\2\u0147\u0148\7V\2\2\u0148")
        buf.write("\u0149\7G\2\2\u0149\u014a\7T\2\2\u014aD\3\2\2\2\u014b")
        buf.write("\u014c\7C\2\2\u014c\u014d\7F\2\2\u014d\u014e\7F\2\2\u014e")
        buf.write("F\3\2\2\2\u014f\u0150\7E\2\2\u0150\u0151\7J\2\2\u0151")
        buf.write("\u0152\7C\2\2\u0152\u0153\7P\2\2\u0153\u0154\7I\2\2\u0154")
        buf.write("\u0155\7G\2\2\u0155H\3\2\2\2\u0156\u0157\7T\2\2\u0157")
        buf.write("\u0158\7G\2\2\u0158\u0159\7P\2\2\u0159\u015a\7C\2\2\u015a")
        buf.write("\u015b\7O\2\2\u015b\u015c\7G\2\2\u015cJ\3\2\2\2\u015d")
        buf.write("\u015e\7R\2\2\u015e\u015f\7T\2\2\u015f\u0160\7K\2\2\u0160")
        buf.write("\u0161\7O\2\2\u0161\u0162\7C\2\2\u0162\u0163\7T\2\2\u0163")
        buf.write("\u0164\7[\2\2\u0164L\3\2\2\2\u0165\u0166\7M\2\2\u0166")
        buf.write("\u0167\7G\2\2\u0167\u0168\7[\2\2\u0168N\3\2\2\2\u0169")
        buf.write("\u016a\7H\2\2\u016a\u016b\7Q\2\2\u016b\u016c\7T\2\2\u016c")
        buf.write("\u016d\7G\2\2\u016d\u016e\7K\2\2\u016e\u016f\7I\2\2\u016f")
        buf.write("\u0170\7P\2\2\u0170P\3\2\2\2\u0171\u0172\7E\2\2\u0172")
        buf.write("\u0173\7Q\2\2\u0173\u0174\7P\2\2\u0174\u0175\7U\2\2\u0175")
        buf.write("\u0176\7V\2\2\u0176\u0177\7T\2\2\u0177\u0178\7C\2\2\u0178")
        buf.write("\u0179\7K\2\2\u0179\u017a\7P\2\2\u017a\u017b\7V\2\2\u017b")
        buf.write("R\3\2\2\2\u017c\u017d\7T\2\2\u017d\u017e\7G\2\2\u017e")
        buf.write("\u017f\7H\2\2\u017f\u0180\7G\2\2\u0180\u0181\7T\2\2\u0181")
        buf.write("\u0182\7G\2\2\u0182\u0183\7P\2\2\u0183\u0184\7E\2\2\u0184")
        buf.write("\u0185\7G\2\2\u0185\u0186\7U\2\2\u0186T\3\2\2\2\u0187")
        buf.write("\u0188\7W\2\2\u0188\u0189\7P\2\2\u0189\u018a\7K\2\2\u018a")
        buf.write("\u018b\7S\2\2\u018b\u018c\7W\2\2\u018c\u018d\7G\2\2\u018d")
        buf.write("V\3\2\2\2\u018e\u018f\7.\2\2\u018fX\3\2\2\2\u0190\u0191")
        buf.write("\7P\2\2\u0191\u0192\7Q\2\2\u0192\u0193\7V\2\2\u0193Z\3")
        buf.write("\2\2\2\u0194\u0195\7F\2\2\u0195\u0196\7G\2\2\u0196\u0197")
        buf.write("\7H\2\2\u0197\u0198\7C\2\2\u0198\u0199\7W\2\2\u0199\u019a")
        buf.write("\7N\2\2\u019a\u019b\7V\2\2\u019b\\\3\2\2\2\u019c\u019d")
        buf.write("\7K\2\2\u019d\u019e\7P\2\2\u019e\u019f\7V\2\2\u019f^\3")
        buf.write("\2\2\2\u01a0\u01a1\7X\2\2\u01a1\u01a2\7C\2\2\u01a2\u01a3")
        buf.write("\7T\2\2\u01a3\u01a4\7E\2\2\u01a4\u01a5\7J\2\2\u01a5\u01a6")
        buf.write("\7C\2\2\u01a6\u01a7\7T\2\2\u01a7`\3\2\2\2\u01a8\u01a9")
        buf.write("\7F\2\2\u01a9\u01aa\7C\2\2\u01aa\u01ab\7V\2\2\u01ab\u01ac")
        buf.write("\7G\2\2\u01acb\3\2\2\2\u01ad\u01ae\7H\2\2\u01ae\u01af")
        buf.write("\7N\2\2\u01af\u01b0\7Q\2\2\u01b0\u01b1\7C\2\2\u01b1\u01b2")
        buf.write("\7V\2\2\u01b2d\3\2\2\2\u01b3\u01b4\7C\2\2\u01b4\u01b5")
        buf.write("\7P\2\2\u01b5\u01b6\7F\2\2\u01b6f\3\2\2\2\u01b7\u01b8")
        buf.write("\7K\2\2\u01b8\u01b9\7U\2\2\u01b9h\3\2\2\2\u01ba\u01bb")
        buf.write("\7K\2\2\u01bb\u01bc\7P\2\2\u01bcj\3\2\2\2\u01bd\u01be")
        buf.write("\7N\2\2\u01be\u01bf\7K\2\2\u01bf\u01c0\7M\2\2\u01c0\u01c1")
        buf.write("\7G\2\2\u01c1l\3\2\2\2\u01c2\u01c3\7\60\2\2\u01c3n\3\2")
        buf.write("\2\2\u01c4\u01c5\7,\2\2\u01c5p\3\2\2\2\u01c6\u01c7\7?")
        buf.write("\2\2\u01c7r\3\2\2\2\u01c8\u01c9\7>\2\2\u01c9t\3\2\2\2")
        buf.write("\u01ca\u01cb\7>\2\2\u01cb\u01cc\7?\2\2\u01ccv\3\2\2\2")
        buf.write("\u01cd\u01ce\7@\2\2\u01cex\3\2\2\2\u01cf\u01d0\7@\2\2")
        buf.write("\u01d0\u01d1\7?\2\2\u01d1z\3\2\2\2\u01d2\u01d3\7>\2\2")
        buf.write("\u01d3\u01d4\7@\2\2\u01d4|\3\2\2\2\u01d5\u01d6\7E\2\2")
        buf.write("\u01d6\u01d7\7Q\2\2\u01d7\u01d8\7W\2\2\u01d8\u01d9\7P")
        buf.write("\2\2\u01d9\u01da\7V\2\2\u01da~\3\2\2\2\u01db\u01dc\7C")
        buf.write("\2\2\u01dc\u01dd\7X\2\2\u01dd\u01de\7I\2\2\u01de\u0080")
        buf.write("\3\2\2\2\u01df\u01e0\7O\2\2\u01e0\u01e1\7C\2\2\u01e1\u01e2")
        buf.write("\7Z\2\2\u01e2\u0082\3\2\2\2\u01e3\u01e4\7O\2\2\u01e4\u01e5")
        buf.write("\7K\2\2\u01e5\u01e6\7P\2\2\u01e6\u0084\3\2\2\2\u01e7\u01e8")
        buf.write("\7U\2\2\u01e8\u01e9\7W\2\2\u01e9\u01ea\7O\2\2\u01ea\u0086")
        buf.write("\3\2\2\2\u01eb\u01ec\7P\2\2\u01ec\u01ed\7W\2\2\u01ed\u01ee")
        buf.write("\7N\2\2\u01ee\u01ef\7N\2\2\u01ef\u0088\3\2\2\2\u01f0\u01f4")
        buf.write("\t\2\2\2\u01f1\u01f3\t\3\2\2\u01f2\u01f1\3\2\2\2\u01f3")
        buf.write("\u01f6\3\2\2\2\u01f4\u01f2\3\2\2\2\u01f4\u01f5\3\2\2\2")
        buf.write("\u01f5\u008a\3\2\2\2\u01f6\u01f4\3\2\2\2\u01f7\u01f9\t")
        buf.write("\4\2\2\u01f8\u01f7\3\2\2\2\u01f9\u01fa\3\2\2\2\u01fa\u01f8")
        buf.write("\3\2\2\2\u01fa\u01fb\3\2\2\2\u01fb\u008c\3\2\2\2\u01fc")
        buf.write("\u0200\7)\2\2\u01fd\u01ff\n\5\2\2\u01fe\u01fd\3\2\2\2")
        buf.write("\u01ff\u0202\3\2\2\2\u0200\u01fe\3\2\2\2\u0200\u0201\3")
        buf.write("\2\2\2\u0201\u0203\3\2\2\2\u0202\u0200\3\2\2\2\u0203\u0204")
        buf.write("\7)\2\2\u0204\u008e\3\2\2\2\u0205\u0207\7/\2\2\u0206\u0205")
        buf.write("\3\2\2\2\u0206\u0207\3\2\2\2\u0207\u0209\3\2\2\2\u0208")
        buf.write("\u020a\t\4\2\2\u0209\u0208\3\2\2\2\u020a\u020b\3\2\2\2")
        buf.write("\u020b\u0209\3\2\2\2\u020b\u020c\3\2\2\2\u020c\u020d\3")
        buf.write("\2\2\2\u020d\u0211\7\60\2\2\u020e\u0210\t\4\2\2\u020f")
        buf.write("\u020e\3\2\2\2\u0210\u0213\3\2\2\2\u0211\u020f\3\2\2\2")
        buf.write("\u0211\u0212\3\2\2\2\u0212\u0090\3\2\2\2\u0213\u0211\3")
        buf.write("\2\2\2\u0214\u0216\t\6\2\2\u0215\u0214\3\2\2\2\u0216\u0217")
        buf.write("\3\2\2\2\u0217\u0215\3\2\2\2\u0217\u0218\3\2\2\2\u0218")
        buf.write("\u0219\3\2\2\2\u0219\u021a\bI\2\2\u021a\u0092\3\2\2\2")
        buf.write("\u021b\u021c\7/\2\2\u021c\u021e\7/\2\2\u021d\u021f\n\7")
        buf.write("\2\2\u021e\u021d\3\2\2\2\u021f\u0220\3\2\2\2\u0220\u021e")
        buf.write("\3\2\2\2\u0220\u0221\3\2\2\2\u0221\u0094\3\2\2\2\13\2")
        buf.write("\u01f4\u01fa\u0200\u0206\u020b\u0211\u0217\u0220\3\b\2")
        buf.write("\2")
        return buf.getvalue()


class SQLLexer(Lexer):

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    T__0 = 1
    T__1 = 2
    T__2 = 3
    T__3 = 4
    T__4 = 5
    T__5 = 6
    T__6 = 7
    T__7 = 8
    T__8 = 9
    T__9 = 10
    T__10 = 11
    T__11 = 12
    T__12 = 13
    T__13 = 14
    T__14 = 15
    T__15 = 16
    T__16 = 17
    T__17 = 18
    T__18 = 19
    T__19 = 20
    T__20 = 21
    T__21 = 22
    T__22 = 23
    T__23 = 24
    T__24 = 25
    T__25 = 26
    T__26 = 27
    T__27 = 28
    T__28 = 29
    T__29 = 30
    T__30 = 31
    T__31 = 32
    T__32 = 33
    T__33 = 34
    T__34 = 35
    T__35 = 36
    T__36 = 37
    T__37 = 38
    T__38 = 39
    T__39 = 40
    T__40 = 41
    T__41 = 42
    T__42 = 43
    T__43 = 44
    T__44 = 45
    T__45 = 46
    T__46 = 47
    T__47 = 48
    T__48 = 49
    T__49 = 50
    T__50 = 51
    T__51 = 52
    T__52 = 53
    T__53 = 54
    T__54 = 55
    EqualOrAssign = 56
    Less = 57
    LessEqual = 58
    Greater = 59
    GreaterEqual = 60
    NotEqual = 61
    Count = 62
    Average = 63
    Max = 64
    Min = 65
    Sum = 66
    Null = 67
    Identifier = 68
    Integer = 69
    String = 70
    Float = 71
    Whitespace = 72
    Annotation = 73

    channelNames = [ u"DEFAULT_TOKEN_CHANNEL", u"HIDDEN" ]

    modeNames = [ "DEFAULT_MODE" ]

    literalNames = [ "<INVALID>",
            "';'", "'SHOW'", "'DATABASES'", "'CREATE'", "'DATABASE'", "'DROP'", 
            "'USE'", "'TABLES'", "'INDEXES'", "'LOAD'", "'FROM'", "'FILE'", 
            "'TO'", "'TABLE'", "'DUMP'", "'('", "')'", "'DESC'", "'INSERT'", 
            "'INTO'", "'VALUES'", "'DELETE'", "'WHERE'", "'UPDATE'", "'SET'", 
            "'SELECT'", "'GROUP'", "'BY'", "'LIMIT'", "'OFFSET'", "'INDEX'", 
            "'ON'", "'ALTER'", "'ADD'", "'CHANGE'", "'RENAME'", "'PRIMARY'", 
            "'KEY'", "'FOREIGN'", "'CONSTRAINT'", "'REFERENCES'", "'UNIQUE'", 
            "','", "'NOT'", "'DEFAULT'", "'INT'", "'VARCHAR'", "'DATE'", 
            "'FLOAT'", "'AND'", "'IS'", "'IN'", "'LIKE'", "'.'", "'*'", 
            "'='", "'<'", "'<='", "'>'", "'>='", "'<>'", "'COUNT'", "'AVG'", 
            "'MAX'", "'MIN'", "'SUM'", "'NULL'" ]

    symbolicNames = [ "<INVALID>",
            "EqualOrAssign", "Less", "LessEqual", "Greater", "GreaterEqual", 
            "NotEqual", "Count", "Average", "Max", "Min", "Sum", "Null", 
            "Identifier", "Integer", "String", "Float", "Whitespace", "Annotation" ]

    ruleNames = [ "T__0", "T__1", "T__2", "T__3", "T__4", "T__5", "T__6", 
                  "T__7", "T__8", "T__9", "T__10", "T__11", "T__12", "T__13", 
                  "T__14", "T__15", "T__16", "T__17", "T__18", "T__19", 
                  "T__20", "T__21", "T__22", "T__23", "T__24", "T__25", 
                  "T__26", "T__27", "T__28", "T__29", "T__30", "T__31", 
                  "T__32", "T__33", "T__34", "T__35", "T__36", "T__37", 
                  "T__38", "T__39", "T__40", "T__41", "T__42", "T__43", 
                  "T__44", "T__45", "T__46", "T__47", "T__48", "T__49", 
                  "T__50", "T__51", "T__52", "T__53", "T__54", "EqualOrAssign", 
                  "Less", "LessEqual", "Greater", "GreaterEqual", "NotEqual", 
                  "Count", "Average", "Max", "Min", "Sum", "Null", "Identifier", 
                  "Integer", "String", "Float", "Whitespace", "Annotation" ]

    grammarFileName = "SQL.g4"

    def __init__(self, input=None, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.9.2")
        self._interp = LexerATNSimulator(self, self.atn, self.decisionsToDFA, PredictionContextCache())
        self._actions = None
        self._predicates = None


# Generated from /Users/liuxinghan/IdeaProjects/test/SQL.g4 by ANTLR 4.9.2
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO


def serializedATN():
    with StringIO() as buf:
        buf.write("\3\u608b\ua72a\u8133\ub9ed\u417c\u3be7\u7786\u5964\3K")
        buf.write("\u01c0\4\2\t\2\4\3\t\3\4\4\t\4\4\5\t\5\4\6\t\6\4\7\t\7")
        buf.write("\4\b\t\b\4\t\t\t\4\n\t\n\4\13\t\13\4\f\t\f\4\r\t\r\4\16")
        buf.write("\t\16\4\17\t\17\4\20\t\20\4\21\t\21\4\22\t\22\4\23\t\23")
        buf.write("\4\24\t\24\4\25\t\25\4\26\t\26\4\27\t\27\4\30\t\30\4\31")
        buf.write("\t\31\4\32\t\32\3\2\7\2\66\n\2\f\2\16\29\13\2\3\2\3\2")
        buf.write("\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3")
        buf.write("\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\3\5\3S\n\3\3\4\3\4\3")
        buf.write("\4\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5\3\5")
        buf.write("\3\5\5\5f\n\5\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6\3\6")
        buf.write("\3\6\3\6\3\6\3\6\5\6v\n\6\3\7\3\7\3\7\3\7\3\7\3\7\3\7")
        buf.write("\3\7\3\7\3\7\3\7\3\7\3\7\3\7\3\7\3\7\3\7\3\7\3\7\3\7\3")
        buf.write("\7\3\7\3\7\3\7\3\7\3\7\3\7\3\7\3\7\3\7\5\7\u0096\n\7\3")
        buf.write("\b\3\b\3\b\3\b\3\b\3\b\5\b\u009e\n\b\3\b\3\b\3\b\5\b\u00a3")
        buf.write("\n\b\3\b\3\b\3\b\3\b\5\b\u00a9\n\b\5\b\u00ab\n\b\3\t\3")
        buf.write("\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t")
        buf.write("\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\3\t\5")
        buf.write("\t\u00c9\n\t\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3")
        buf.write("\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n")
        buf.write("\3\n\3\n\3\n\3\n\3\n\5\n\u00e8\n\n\3\n\3\n\3\n\3\n\3\n")
        buf.write("\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3")
        buf.write("\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n")
        buf.write("\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\3\n\5")
        buf.write("\n\u0117\n\n\3\13\3\13\3\13\7\13\u011c\n\13\f\13\16\13")
        buf.write("\u011f\13\13\3\f\3\f\3\f\3\f\5\f\u0125\n\f\3\f\3\f\5\f")
        buf.write("\u0129\n\f\3\f\3\f\3\f\3\f\3\f\3\f\3\f\3\f\3\f\3\f\3\f")
        buf.write("\3\f\3\f\3\f\3\f\3\f\5\f\u013b\n\f\3\r\3\r\3\r\3\r\3\r")
        buf.write("\3\r\3\r\3\r\3\r\3\r\5\r\u0147\n\r\3\16\3\16\3\16\7\16")
        buf.write("\u014c\n\16\f\16\16\16\u014f\13\16\3\17\3\17\3\17\3\17")
        buf.write("\7\17\u0155\n\17\f\17\16\17\u0158\13\17\3\17\3\17\3\20")
        buf.write("\3\20\3\21\3\21\3\21\7\21\u0161\n\21\f\21\16\21\u0164")
        buf.write("\13\21\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3")
        buf.write("\22\3\22\3\22\3\22\5\22\u0173\n\22\3\22\3\22\3\22\3\22")
        buf.write("\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22\3\22")
        buf.write("\3\22\5\22\u0185\n\22\3\23\3\23\5\23\u0189\n\23\3\23\3")
        buf.write("\23\3\24\3\24\5\24\u018f\n\24\3\25\3\25\3\25\3\25\3\25")
        buf.write("\3\25\3\25\7\25\u0198\n\25\f\25\16\25\u019b\13\25\3\26")
        buf.write("\3\26\3\26\3\26\7\26\u01a1\n\26\f\26\16\26\u01a4\13\26")
        buf.write("\5\26\u01a6\n\26\3\27\3\27\3\27\3\27\3\27\3\27\3\27\3")
        buf.write("\27\3\27\3\27\5\27\u01b2\n\27\3\30\3\30\3\30\7\30\u01b7")
        buf.write("\n\30\f\30\16\30\u01ba\13\30\3\31\3\31\3\32\3\32\3\32")
        buf.write("\2\2\33\2\4\6\b\n\f\16\20\22\24\26\30\32\34\36 \"$&(*")
        buf.write(",.\60\62\2\5\4\2EEGI\3\2:?\3\2@D\2\u01e3\2\67\3\2\2\2")
        buf.write("\4R\3\2\2\2\6T\3\2\2\2\be\3\2\2\2\nu\3\2\2\2\f\u0095\3")
        buf.write("\2\2\2\16\u0097\3\2\2\2\20\u00c8\3\2\2\2\22\u0116\3\2")
        buf.write("\2\2\24\u0118\3\2\2\2\26\u013a\3\2\2\2\30\u0146\3\2\2")
        buf.write("\2\32\u0148\3\2\2\2\34\u0150\3\2\2\2\36\u015b\3\2\2\2")
        buf.write(" \u015d\3\2\2\2\"\u0184\3\2\2\2$\u0188\3\2\2\2&\u018e")
        buf.write("\3\2\2\2(\u0190\3\2\2\2*\u01a5\3\2\2\2,\u01b1\3\2\2\2")
        buf.write(".\u01b3\3\2\2\2\60\u01bb\3\2\2\2\62\u01bd\3\2\2\2\64\66")
        buf.write("\5\4\3\2\65\64\3\2\2\2\669\3\2\2\2\67\65\3\2\2\2\678\3")
        buf.write("\2\2\28:\3\2\2\29\67\3\2\2\2:;\7\2\2\3;\3\3\2\2\2<=\5")
        buf.write("\6\4\2=>\7\3\2\2>S\3\2\2\2?@\5\b\5\2@A\7\3\2\2AS\3\2\2")
        buf.write("\2BC\5\n\6\2CD\7\3\2\2DS\3\2\2\2EF\5\f\7\2FG\7\3\2\2G")
        buf.write("S\3\2\2\2HI\5\20\t\2IJ\7\3\2\2JS\3\2\2\2KL\5\22\n\2LM")
        buf.write("\7\3\2\2MS\3\2\2\2NO\7K\2\2OS\7\3\2\2PQ\7E\2\2QS\7\3\2")
        buf.write("\2R<\3\2\2\2R?\3\2\2\2RB\3\2\2\2RE\3\2\2\2RH\3\2\2\2R")
        buf.write("K\3\2\2\2RN\3\2\2\2RP\3\2\2\2S\5\3\2\2\2TU\7\4\2\2UV\7")
        buf.write("\5\2\2V\7\3\2\2\2WX\7\6\2\2XY\7\7\2\2Yf\7F\2\2Z[\7\b\2")
        buf.write("\2[\\\7\7\2\2\\f\7F\2\2]^\7\4\2\2^f\7\5\2\2_`\7\t\2\2")
        buf.write("`f\7F\2\2ab\7\4\2\2bf\7\n\2\2cd\7\4\2\2df\7\13\2\2eW\3")
        buf.write("\2\2\2eZ\3\2\2\2e]\3\2\2\2e_\3\2\2\2ea\3\2\2\2ec\3\2\2")
        buf.write("\2f\t\3\2\2\2gh\7\f\2\2hi\7\r\2\2ij\7\16\2\2jk\7H\2\2")
        buf.write("kl\7\17\2\2lm\7\20\2\2mv\7F\2\2no\7\21\2\2op\7\17\2\2")
        buf.write("pq\7\16\2\2qr\7H\2\2rs\7\r\2\2st\7\20\2\2tv\7F\2\2ug\3")
        buf.write("\2\2\2un\3\2\2\2v\13\3\2\2\2wx\7\6\2\2xy\7\20\2\2yz\7")
        buf.write("F\2\2z{\7\22\2\2{|\5\24\13\2|}\7\23\2\2}\u0096\3\2\2\2")
        buf.write("~\177\7\b\2\2\177\u0080\7\20\2\2\u0080\u0096\7F\2\2\u0081")
        buf.write("\u0082\7\24\2\2\u0082\u0096\7F\2\2\u0083\u0084\7\25\2")
        buf.write("\2\u0084\u0085\7\26\2\2\u0085\u0086\7F\2\2\u0086\u0087")
        buf.write("\7\27\2\2\u0087\u0096\5\32\16\2\u0088\u0089\7\30\2\2\u0089")
        buf.write("\u008a\7\r\2\2\u008a\u008b\7F\2\2\u008b\u008c\7\31\2\2")
        buf.write("\u008c\u0096\5 \21\2\u008d\u008e\7\32\2\2\u008e\u008f")
        buf.write("\7F\2\2\u008f\u0090\7\33\2\2\u0090\u0091\5(\25\2\u0091")
        buf.write("\u0092\7\31\2\2\u0092\u0093\5 \21\2\u0093\u0096\3\2\2")
        buf.write("\2\u0094\u0096\5\16\b\2\u0095w\3\2\2\2\u0095~\3\2\2\2")
        buf.write("\u0095\u0081\3\2\2\2\u0095\u0083\3\2\2\2\u0095\u0088\3")
        buf.write("\2\2\2\u0095\u008d\3\2\2\2\u0095\u0094\3\2\2\2\u0096\r")
        buf.write("\3\2\2\2\u0097\u0098\7\34\2\2\u0098\u0099\5*\26\2\u0099")
        buf.write("\u009a\7\r\2\2\u009a\u009d\5.\30\2\u009b\u009c\7\31\2")
        buf.write("\2\u009c\u009e\5 \21\2\u009d\u009b\3\2\2\2\u009d\u009e")
        buf.write("\3\2\2\2\u009e\u00a2\3\2\2\2\u009f\u00a0\7\35\2\2\u00a0")
        buf.write("\u00a1\7\36\2\2\u00a1\u00a3\5$\23\2\u00a2\u009f\3\2\2")
        buf.write("\2\u00a2\u00a3\3\2\2\2\u00a3\u00aa\3\2\2\2\u00a4\u00a5")
        buf.write("\7\37\2\2\u00a5\u00a8\7G\2\2\u00a6\u00a7\7 \2\2\u00a7")
        buf.write("\u00a9\7G\2\2\u00a8\u00a6\3\2\2\2\u00a8\u00a9\3\2\2\2")
        buf.write("\u00a9\u00ab\3\2\2\2\u00aa\u00a4\3\2\2\2\u00aa\u00ab\3")
        buf.write("\2\2\2\u00ab\17\3\2\2\2\u00ac\u00ad\7\6\2\2\u00ad\u00ae")
        buf.write("\7!\2\2\u00ae\u00af\7F\2\2\u00af\u00b0\7\"\2\2\u00b0\u00b1")
        buf.write("\7F\2\2\u00b1\u00b2\7\22\2\2\u00b2\u00b3\5.\30\2\u00b3")
        buf.write("\u00b4\7\23\2\2\u00b4\u00c9\3\2\2\2\u00b5\u00b6\7\b\2")
        buf.write("\2\u00b6\u00b7\7!\2\2\u00b7\u00c9\7F\2\2\u00b8\u00b9\7")
        buf.write("#\2\2\u00b9\u00ba\7\20\2\2\u00ba\u00bb\7F\2\2\u00bb\u00bc")
        buf.write("\7$\2\2\u00bc\u00bd\7!\2\2\u00bd\u00be\7F\2\2\u00be\u00bf")
        buf.write("\7\22\2\2\u00bf\u00c0\5.\30\2\u00c0\u00c1\7\23\2\2\u00c1")
        buf.write("\u00c9\3\2\2\2\u00c2\u00c3\7#\2\2\u00c3\u00c4\7\20\2\2")
        buf.write("\u00c4\u00c5\7F\2\2\u00c5\u00c6\7\b\2\2\u00c6\u00c7\7")
        buf.write("!\2\2\u00c7\u00c9\7F\2\2\u00c8\u00ac\3\2\2\2\u00c8\u00b5")
        buf.write("\3\2\2\2\u00c8\u00b8\3\2\2\2\u00c8\u00c2\3\2\2\2\u00c9")
        buf.write("\21\3\2\2\2\u00ca\u00cb\7#\2\2\u00cb\u00cc\7\20\2\2\u00cc")
        buf.write("\u00cd\7F\2\2\u00cd\u00ce\7$\2\2\u00ce\u0117\5\26\f\2")
        buf.write("\u00cf\u00d0\7#\2\2\u00d0\u00d1\7\20\2\2\u00d1\u00d2\7")
        buf.write("F\2\2\u00d2\u00d3\7\b\2\2\u00d3\u0117\7F\2\2\u00d4\u00d5")
        buf.write("\7#\2\2\u00d5\u00d6\7\20\2\2\u00d6\u00d7\7F\2\2\u00d7")
        buf.write("\u00d8\7%\2\2\u00d8\u00d9\7F\2\2\u00d9\u0117\5\26\f\2")
        buf.write("\u00da\u00db\7#\2\2\u00db\u00dc\7\20\2\2\u00dc\u00dd\7")
        buf.write("F\2\2\u00dd\u00de\7&\2\2\u00de\u00df\7\17\2\2\u00df\u0117")
        buf.write("\7F\2\2\u00e0\u00e1\7#\2\2\u00e1\u00e2\7\20\2\2\u00e2")
        buf.write("\u00e3\7F\2\2\u00e3\u00e4\7\b\2\2\u00e4\u00e5\7\'\2\2")
        buf.write("\u00e5\u00e7\7(\2\2\u00e6\u00e8\7F\2\2\u00e7\u00e6\3\2")
        buf.write("\2\2\u00e7\u00e8\3\2\2\2\u00e8\u0117\3\2\2\2\u00e9\u00ea")
        buf.write("\7#\2\2\u00ea\u00eb\7\20\2\2\u00eb\u00ec\7F\2\2\u00ec")
        buf.write("\u00ed\7\b\2\2\u00ed\u00ee\7)\2\2\u00ee\u00ef\7(\2\2\u00ef")
        buf.write("\u0117\7F\2\2\u00f0\u00f1\7#\2\2\u00f1\u00f2\7\20\2\2")
        buf.write("\u00f2\u00f3\7F\2\2\u00f3\u00f4\7$\2\2\u00f4\u00f5\7*")
        buf.write("\2\2\u00f5\u00f6\7F\2\2\u00f6\u00f7\7\'\2\2\u00f7\u00f8")
        buf.write("\7(\2\2\u00f8\u00f9\7\22\2\2\u00f9\u00fa\5.\30\2\u00fa")
        buf.write("\u00fb\7\23\2\2\u00fb\u0117\3\2\2\2\u00fc\u00fd\7#\2\2")
        buf.write("\u00fd\u00fe\7\20\2\2\u00fe\u00ff\7F\2\2\u00ff\u0100\7")
        buf.write("$\2\2\u0100\u0101\7*\2\2\u0101\u0102\7F\2\2\u0102\u0103")
        buf.write("\7)\2\2\u0103\u0104\7(\2\2\u0104\u0105\7\22\2\2\u0105")
        buf.write("\u0106\5.\30\2\u0106\u0107\7\23\2\2\u0107\u0108\7+\2\2")
        buf.write("\u0108\u0109\7F\2\2\u0109\u010a\7\22\2\2\u010a\u010b\5")
        buf.write(".\30\2\u010b\u010c\7\23\2\2\u010c\u0117\3\2\2\2\u010d")
        buf.write("\u010e\7#\2\2\u010e\u010f\7\20\2\2\u010f\u0110\7F\2\2")
        buf.write("\u0110\u0111\7$\2\2\u0111\u0112\7,\2\2\u0112\u0113\7F")
        buf.write("\2\2\u0113\u0114\7\22\2\2\u0114\u0115\7F\2\2\u0115\u0117")
        buf.write("\7\23\2\2\u0116\u00ca\3\2\2\2\u0116\u00cf\3\2\2\2\u0116")
        buf.write("\u00d4\3\2\2\2\u0116\u00da\3\2\2\2\u0116\u00e0\3\2\2\2")
        buf.write("\u0116\u00e9\3\2\2\2\u0116\u00f0\3\2\2\2\u0116\u00fc\3")
        buf.write("\2\2\2\u0116\u010d\3\2\2\2\u0117\23\3\2\2\2\u0118\u011d")
        buf.write("\5\26\f\2\u0119\u011a\7-\2\2\u011a\u011c\5\26\f\2\u011b")
        buf.write("\u0119\3\2\2\2\u011c\u011f\3\2\2\2\u011d\u011b\3\2\2\2")
        buf.write("\u011d\u011e\3\2\2\2\u011e\25\3\2\2\2\u011f\u011d\3\2")
        buf.write("\2\2\u0120\u0121\7F\2\2\u0121\u0124\5\30\r\2\u0122\u0123")
        buf.write("\7.\2\2\u0123\u0125\7E\2\2\u0124\u0122\3\2\2\2\u0124\u0125")
        buf.write("\3\2\2\2\u0125\u0128\3\2\2\2\u0126\u0127\7/\2\2\u0127")
        buf.write("\u0129\5\36\20\2\u0128\u0126\3\2\2\2\u0128\u0129\3\2\2")
        buf.write("\2\u0129\u013b\3\2\2\2\u012a\u012b\7\'\2\2\u012b\u012c")
        buf.write("\7(\2\2\u012c\u012d\7\22\2\2\u012d\u012e\5.\30\2\u012e")
        buf.write("\u012f\7\23\2\2\u012f\u013b\3\2\2\2\u0130\u0131\7)\2\2")
        buf.write("\u0131\u0132\7(\2\2\u0132\u0133\7\22\2\2\u0133\u0134\7")
        buf.write("F\2\2\u0134\u0135\7\23\2\2\u0135\u0136\7+\2\2\u0136\u0137")
        buf.write("\7F\2\2\u0137\u0138\7\22\2\2\u0138\u0139\7F\2\2\u0139")
        buf.write("\u013b\7\23\2\2\u013a\u0120\3\2\2\2\u013a\u012a\3\2\2")
        buf.write("\2\u013a\u0130\3\2\2\2\u013b\27\3\2\2\2\u013c\u013d\7")
        buf.write("\60\2\2\u013d\u013e\7\22\2\2\u013e\u013f\7G\2\2\u013f")
        buf.write("\u0147\7\23\2\2\u0140\u0141\7\61\2\2\u0141\u0142\7\22")
        buf.write("\2\2\u0142\u0143\7G\2\2\u0143\u0147\7\23\2\2\u0144\u0147")
        buf.write("\7\62\2\2\u0145\u0147\7\63\2\2\u0146\u013c\3\2\2\2\u0146")
        buf.write("\u0140\3\2\2\2\u0146\u0144\3\2\2\2\u0146\u0145\3\2\2\2")
        buf.write("\u0147\31\3\2\2\2\u0148\u014d\5\34\17\2\u0149\u014a\7")
        buf.write("-\2\2\u014a\u014c\5\34\17\2\u014b\u0149\3\2\2\2\u014c")
        buf.write("\u014f\3\2\2\2\u014d\u014b\3\2\2\2\u014d\u014e\3\2\2\2")
        buf.write("\u014e\33\3\2\2\2\u014f\u014d\3\2\2\2\u0150\u0151\7\22")
        buf.write("\2\2\u0151\u0156\5\36\20\2\u0152\u0153\7-\2\2\u0153\u0155")
        buf.write("\5\36\20\2\u0154\u0152\3\2\2\2\u0155\u0158\3\2\2\2\u0156")
        buf.write("\u0154\3\2\2\2\u0156\u0157\3\2\2\2\u0157\u0159\3\2\2\2")
        buf.write("\u0158\u0156\3\2\2\2\u0159\u015a\7\23\2\2\u015a\35\3\2")
        buf.write("\2\2\u015b\u015c\t\2\2\2\u015c\37\3\2\2\2\u015d\u0162")
        buf.write("\5\"\22\2\u015e\u015f\7\64\2\2\u015f\u0161\5\"\22\2\u0160")
        buf.write("\u015e\3\2\2\2\u0161\u0164\3\2\2\2\u0162\u0160\3\2\2\2")
        buf.write("\u0162\u0163\3\2\2\2\u0163!\3\2\2\2\u0164\u0162\3\2\2")
        buf.write("\2\u0165\u0166\5$\23\2\u0166\u0167\5\60\31\2\u0167\u0168")
        buf.write("\5&\24\2\u0168\u0185\3\2\2\2\u0169\u016a\5$\23\2\u016a")
        buf.write("\u016b\5\60\31\2\u016b\u016c\7\22\2\2\u016c\u016d\5\16")
        buf.write("\b\2\u016d\u016e\7\23\2\2\u016e\u0185\3\2\2\2\u016f\u0170")
        buf.write("\5$\23\2\u0170\u0172\7\65\2\2\u0171\u0173\7.\2\2\u0172")
        buf.write("\u0171\3\2\2\2\u0172\u0173\3\2\2\2\u0173\u0174\3\2\2\2")
        buf.write("\u0174\u0175\7E\2\2\u0175\u0185\3\2\2\2\u0176\u0177\5")
        buf.write("$\23\2\u0177\u0178\7\66\2\2\u0178\u0179\5\34\17\2\u0179")
        buf.write("\u0185\3\2\2\2\u017a\u017b\5$\23\2\u017b\u017c\7\66\2")
        buf.write("\2\u017c\u017d\7\22\2\2\u017d\u017e\5\16\b\2\u017e\u017f")
        buf.write("\7\23\2\2\u017f\u0185\3\2\2\2\u0180\u0181\5$\23\2\u0181")
        buf.write("\u0182\7\67\2\2\u0182\u0183\7H\2\2\u0183\u0185\3\2\2\2")
        buf.write("\u0184\u0165\3\2\2\2\u0184\u0169\3\2\2\2\u0184\u016f\3")
        buf.write("\2\2\2\u0184\u0176\3\2\2\2\u0184\u017a\3\2\2\2\u0184\u0180")
        buf.write("\3\2\2\2\u0185#\3\2\2\2\u0186\u0187\7F\2\2\u0187\u0189")
        buf.write("\78\2\2\u0188\u0186\3\2\2\2\u0188\u0189\3\2\2\2\u0189")
        buf.write("\u018a\3\2\2\2\u018a\u018b\7F\2\2\u018b%\3\2\2\2\u018c")
        buf.write("\u018f\5\36\20\2\u018d\u018f\5$\23\2\u018e\u018c\3\2\2")
        buf.write("\2\u018e\u018d\3\2\2\2\u018f\'\3\2\2\2\u0190\u0191\7F")
        buf.write("\2\2\u0191\u0192\7:\2\2\u0192\u0199\5\36\20\2\u0193\u0194")
        buf.write("\7-\2\2\u0194\u0195\7F\2\2\u0195\u0196\7:\2\2\u0196\u0198")
        buf.write("\5\36\20\2\u0197\u0193\3\2\2\2\u0198\u019b\3\2\2\2\u0199")
        buf.write("\u0197\3\2\2\2\u0199\u019a\3\2\2\2\u019a)\3\2\2\2\u019b")
        buf.write("\u0199\3\2\2\2\u019c\u01a6\79\2\2\u019d\u01a2\5,\27\2")
        buf.write("\u019e\u019f\7-\2\2\u019f\u01a1\5,\27\2\u01a0\u019e\3")
        buf.write("\2\2\2\u01a1\u01a4\3\2\2\2\u01a2\u01a0\3\2\2\2\u01a2\u01a3")
        buf.write("\3\2\2\2\u01a3\u01a6\3\2\2\2\u01a4\u01a2\3\2\2\2\u01a5")
        buf.write("\u019c\3\2\2\2\u01a5\u019d\3\2\2\2\u01a6+\3\2\2\2\u01a7")
        buf.write("\u01b2\5$\23\2\u01a8\u01a9\5\62\32\2\u01a9\u01aa\7\22")
        buf.write("\2\2\u01aa\u01ab\5$\23\2\u01ab\u01ac\7\23\2\2\u01ac\u01b2")
        buf.write("\3\2\2\2\u01ad\u01ae\7@\2\2\u01ae\u01af\7\22\2\2\u01af")
        buf.write("\u01b0\79\2\2\u01b0\u01b2\7\23\2\2\u01b1\u01a7\3\2\2\2")
        buf.write("\u01b1\u01a8\3\2\2\2\u01b1\u01ad\3\2\2\2\u01b2-\3\2\2")
        buf.write("\2\u01b3\u01b8\7F\2\2\u01b4\u01b5\7-\2\2\u01b5\u01b7\7")
        buf.write("F\2\2\u01b6\u01b4\3\2\2\2\u01b7\u01ba\3\2\2\2\u01b8\u01b6")
        buf.write("\3\2\2\2\u01b8\u01b9\3\2\2\2\u01b9/\3\2\2\2\u01ba\u01b8")
        buf.write("\3\2\2\2\u01bb\u01bc\t\3\2\2\u01bc\61\3\2\2\2\u01bd\u01be")
        buf.write("\t\4\2\2\u01be\63\3\2\2\2\37\67Reu\u0095\u009d\u00a2\u00a8")
        buf.write("\u00aa\u00c8\u00e7\u0116\u011d\u0124\u0128\u013a\u0146")
        buf.write("\u014d\u0156\u0162\u0172\u0184\u0188\u018e\u0199\u01a2")
        buf.write("\u01a5\u01b1\u01b8")
        return buf.getvalue()


class SQLParser ( Parser ):

    grammarFileName = "SQL.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "';'", "'SHOW'", "'DATABASES'", "'CREATE'", 
                     "'DATABASE'", "'DROP'", "'USE'", "'TABLES'", "'INDEXES'", 
                     "'LOAD'", "'FROM'", "'FILE'", "'TO'", "'TABLE'", "'DUMP'", 
                     "'('", "')'", "'DESC'", "'INSERT'", "'INTO'", "'VALUES'", 
                     "'DELETE'", "'WHERE'", "'UPDATE'", "'SET'", "'SELECT'", 
                     "'GROUP'", "'BY'", "'LIMIT'", "'OFFSET'", "'INDEX'", 
                     "'ON'", "'ALTER'", "'ADD'", "'CHANGE'", "'RENAME'", 
                     "'PRIMARY'", "'KEY'", "'FOREIGN'", "'CONSTRAINT'", 
                     "'REFERENCES'", "'UNIQUE'", "','", "'NOT'", "'DEFAULT'", 
                     "'INT'", "'VARCHAR'", "'DATE'", "'FLOAT'", "'AND'", 
                     "'IS'", "'IN'", "'LIKE'", "'.'", "'*'", "'='", "'<'", 
                     "'<='", "'>'", "'>='", "'<>'", "'COUNT'", "'AVG'", 
                     "'MAX'", "'MIN'", "'SUM'", "'NULL'" ]

    symbolicNames = [ "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                      "EqualOrAssign", "Less", "LessEqual", "Greater", "GreaterEqual", 
                      "NotEqual", "Count", "Average", "Max", "Min", "Sum", 
                      "Null", "Identifier", "Integer", "String", "Float", 
                      "Whitespace", "Annotation" ]

    RULE_program = 0
    RULE_statement = 1
    RULE_system_statement = 2
    RULE_db_statement = 3
    RULE_io_statement = 4
    RULE_table_statement = 5
    RULE_select_table = 6
    RULE_index_statement = 7
    RULE_alter_statement = 8
    RULE_field_list = 9
    RULE_field = 10
    RULE_type_ = 11
    RULE_value_lists = 12
    RULE_value_list = 13
    RULE_value = 14
    RULE_where_and_clause = 15
    RULE_where_clause = 16
    RULE_column = 17
    RULE_expression = 18
    RULE_set_clause = 19
    RULE_selectors = 20
    RULE_selector = 21
    RULE_identifiers = 22
    RULE_operator = 23
    RULE_aggregator = 24

    ruleNames =  [ "program", "statement", "system_statement", "db_statement", 
                   "io_statement", "table_statement", "select_table", "index_statement", 
                   "alter_statement", "field_list", "field", "type_", "value_lists", 
                   "value_list", "value", "where_and_clause", "where_clause", 
                   "column", "expression", "set_clause", "selectors", "selector", 
                   "identifiers", "operator", "aggregator" ]

    EOF = Token.EOF
    T__0=1
    T__1=2
    T__2=3
    T__3=4
    T__4=5
    T__5=6
    T__6=7
    T__7=8
    T__8=9
    T__9=10
    T__10=11
    T__11=12
    T__12=13
    T__13=14
    T__14=15
    T__15=16
    T__16=17
    T__17=18
    T__18=19
    T__19=20
    T__20=21
    T__21=22
    T__22=23
    T__23=24
    T__24=25
    T__25=26
    T__26=27
    T__27=28
    T__28=29
    T__29=30
    T__30=31
    T__31=32
    T__32=33
    T__33=34
    T__34=35
    T__35=36
    T__36=37
    T__37=38
    T__38=39
    T__39=40
    T__40=41
    T__41=42
    T__42=43
    T__43=44
    T__44=45
    T__45=46
    T__46=47
    T__47=48
    T__48=49
    T__49=50
    T__50=51
    T__51=52
    T__52=53
    T__53=54
    T__54=55
    EqualOrAssign=56
    Less=57
    LessEqual=58
    Greater=59
    GreaterEqual=60
    NotEqual=61
    Count=62
    Average=63
    Max=64
    Min=65
    Sum=66
    Null=67
    Identifier=68
    Integer=69
    String=70
    Float=71
    Whitespace=72
    Annotation=73

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.9.2")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class ProgramContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EOF(self):
            return self.getToken(SQLParser.EOF, 0)

        def statement(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SQLParser.StatementContext)
            else:
                return self.getTypedRuleContext(SQLParser.StatementContext,i)


        def getRuleIndex(self):
            return SQLParser.RULE_program

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitProgram" ):
                return visitor.visitProgram(self)
            else:
                return visitor.visitChildren(self)




    def program(self):

        localctx = SQLParser.ProgramContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_program)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 53
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while (((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SQLParser.T__1) | (1 << SQLParser.T__3) | (1 << SQLParser.T__5) | (1 << SQLParser.T__6) | (1 << SQLParser.T__9) | (1 << SQLParser.T__14) | (1 << SQLParser.T__17) | (1 << SQLParser.T__18) | (1 << SQLParser.T__21) | (1 << SQLParser.T__23) | (1 << SQLParser.T__25) | (1 << SQLParser.T__32))) != 0) or _la==SQLParser.Null or _la==SQLParser.Annotation:
                self.state = 50
                self.statement()
                self.state = 55
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 56
            self.match(SQLParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class StatementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def system_statement(self):
            return self.getTypedRuleContext(SQLParser.System_statementContext,0)


        def db_statement(self):
            return self.getTypedRuleContext(SQLParser.Db_statementContext,0)


        def io_statement(self):
            return self.getTypedRuleContext(SQLParser.Io_statementContext,0)


        def table_statement(self):
            return self.getTypedRuleContext(SQLParser.Table_statementContext,0)


        def index_statement(self):
            return self.getTypedRuleContext(SQLParser.Index_statementContext,0)


        def alter_statement(self):
            return self.getTypedRuleContext(SQLParser.Alter_statementContext,0)


        def Annotation(self):
            return self.getToken(SQLParser.Annotation, 0)

        def Null(self):
            return self.getToken(SQLParser.Null, 0)

        def getRuleIndex(self):
            return SQLParser.RULE_statement

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStatement" ):
                return visitor.visitStatement(self)
            else:
                return visitor.visitChildren(self)




    def statement(self):

        localctx = SQLParser.StatementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 2, self.RULE_statement)
        try:
            self.state = 80
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 58
                self.system_statement()
                self.state = 59
                self.match(SQLParser.T__0)
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 61
                self.db_statement()
                self.state = 62
                self.match(SQLParser.T__0)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 64
                self.io_statement()
                self.state = 65
                self.match(SQLParser.T__0)
                pass

            elif la_ == 4:
                self.enterOuterAlt(localctx, 4)
                self.state = 67
                self.table_statement()
                self.state = 68
                self.match(SQLParser.T__0)
                pass

            elif la_ == 5:
                self.enterOuterAlt(localctx, 5)
                self.state = 70
                self.index_statement()
                self.state = 71
                self.match(SQLParser.T__0)
                pass

            elif la_ == 6:
                self.enterOuterAlt(localctx, 6)
                self.state = 73
                self.alter_statement()
                self.state = 74
                self.match(SQLParser.T__0)
                pass

            elif la_ == 7:
                self.enterOuterAlt(localctx, 7)
                self.state = 76
                self.match(SQLParser.Annotation)
                self.state = 77
                self.match(SQLParser.T__0)
                pass

            elif la_ == 8:
                self.enterOuterAlt(localctx, 8)
                self.state = 78
                self.match(SQLParser.Null)
                self.state = 79
                self.match(SQLParser.T__0)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class System_statementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return SQLParser.RULE_system_statement

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSystem_statement" ):
                return visitor.visitSystem_statement(self)
            else:
                return visitor.visitChildren(self)




    def system_statement(self):

        localctx = SQLParser.System_statementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 4, self.RULE_system_statement)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 82
            self.match(SQLParser.T__1)
            self.state = 83
            self.match(SQLParser.T__2)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Db_statementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return SQLParser.RULE_db_statement

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class Show_dbsContext(Db_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Db_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitShow_dbs" ):
                return visitor.visitShow_dbs(self)
            else:
                return visitor.visitChildren(self)


    class Drop_dbContext(Db_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Db_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDrop_db" ):
                return visitor.visitDrop_db(self)
            else:
                return visitor.visitChildren(self)


    class Show_tablesContext(Db_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Db_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitShow_tables" ):
                return visitor.visitShow_tables(self)
            else:
                return visitor.visitChildren(self)


    class Create_dbContext(Db_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Db_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCreate_db" ):
                return visitor.visitCreate_db(self)
            else:
                return visitor.visitChildren(self)


    class Use_dbContext(Db_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Db_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitUse_db" ):
                return visitor.visitUse_db(self)
            else:
                return visitor.visitChildren(self)


    class Show_indexesContext(Db_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Db_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitShow_indexes" ):
                return visitor.visitShow_indexes(self)
            else:
                return visitor.visitChildren(self)



    def db_statement(self):

        localctx = SQLParser.Db_statementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_db_statement)
        try:
            self.state = 99
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
            if la_ == 1:
                localctx = SQLParser.Create_dbContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 85
                self.match(SQLParser.T__3)
                self.state = 86
                self.match(SQLParser.T__4)
                self.state = 87
                self.match(SQLParser.Identifier)
                pass

            elif la_ == 2:
                localctx = SQLParser.Drop_dbContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 88
                self.match(SQLParser.T__5)
                self.state = 89
                self.match(SQLParser.T__4)
                self.state = 90
                self.match(SQLParser.Identifier)
                pass

            elif la_ == 3:
                localctx = SQLParser.Show_dbsContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 91
                self.match(SQLParser.T__1)
                self.state = 92
                self.match(SQLParser.T__2)
                pass

            elif la_ == 4:
                localctx = SQLParser.Use_dbContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 93
                self.match(SQLParser.T__6)
                self.state = 94
                self.match(SQLParser.Identifier)
                pass

            elif la_ == 5:
                localctx = SQLParser.Show_tablesContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 95
                self.match(SQLParser.T__1)
                self.state = 96
                self.match(SQLParser.T__7)
                pass

            elif la_ == 6:
                localctx = SQLParser.Show_indexesContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 97
                self.match(SQLParser.T__1)
                self.state = 98
                self.match(SQLParser.T__8)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Io_statementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return SQLParser.RULE_io_statement

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class Dump_dataContext(Io_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Io_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def String(self):
            return self.getToken(SQLParser.String, 0)
        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDump_data" ):
                return visitor.visitDump_data(self)
            else:
                return visitor.visitChildren(self)


    class Load_dataContext(Io_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Io_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def String(self):
            return self.getToken(SQLParser.String, 0)
        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitLoad_data" ):
                return visitor.visitLoad_data(self)
            else:
                return visitor.visitChildren(self)



    def io_statement(self):

        localctx = SQLParser.Io_statementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 8, self.RULE_io_statement)
        try:
            self.state = 115
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SQLParser.T__9]:
                localctx = SQLParser.Load_dataContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 101
                self.match(SQLParser.T__9)
                self.state = 102
                self.match(SQLParser.T__10)
                self.state = 103
                self.match(SQLParser.T__11)
                self.state = 104
                self.match(SQLParser.String)
                self.state = 105
                self.match(SQLParser.T__12)
                self.state = 106
                self.match(SQLParser.T__13)
                self.state = 107
                self.match(SQLParser.Identifier)
                pass
            elif token in [SQLParser.T__14]:
                localctx = SQLParser.Dump_dataContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 108
                self.match(SQLParser.T__14)
                self.state = 109
                self.match(SQLParser.T__12)
                self.state = 110
                self.match(SQLParser.T__11)
                self.state = 111
                self.match(SQLParser.String)
                self.state = 112
                self.match(SQLParser.T__10)
                self.state = 113
                self.match(SQLParser.T__13)
                self.state = 114
                self.match(SQLParser.Identifier)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Table_statementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return SQLParser.RULE_table_statement

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class Delete_from_tableContext(Table_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Table_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)
        def where_and_clause(self):
            return self.getTypedRuleContext(SQLParser.Where_and_clauseContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDelete_from_table" ):
                return visitor.visitDelete_from_table(self)
            else:
                return visitor.visitChildren(self)


    class Insert_into_tableContext(Table_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Table_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)
        def value_lists(self):
            return self.getTypedRuleContext(SQLParser.Value_listsContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitInsert_into_table" ):
                return visitor.visitInsert_into_table(self)
            else:
                return visitor.visitChildren(self)


    class Create_tableContext(Table_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Table_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)
        def field_list(self):
            return self.getTypedRuleContext(SQLParser.Field_listContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCreate_table" ):
                return visitor.visitCreate_table(self)
            else:
                return visitor.visitChildren(self)


    class Describe_tableContext(Table_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Table_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDescribe_table" ):
                return visitor.visitDescribe_table(self)
            else:
                return visitor.visitChildren(self)


    class Select_table_Context(Table_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Table_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def select_table(self):
            return self.getTypedRuleContext(SQLParser.Select_tableContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSelect_table_" ):
                return visitor.visitSelect_table_(self)
            else:
                return visitor.visitChildren(self)


    class Drop_tableContext(Table_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Table_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDrop_table" ):
                return visitor.visitDrop_table(self)
            else:
                return visitor.visitChildren(self)


    class Update_tableContext(Table_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Table_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)
        def set_clause(self):
            return self.getTypedRuleContext(SQLParser.Set_clauseContext,0)

        def where_and_clause(self):
            return self.getTypedRuleContext(SQLParser.Where_and_clauseContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitUpdate_table" ):
                return visitor.visitUpdate_table(self)
            else:
                return visitor.visitChildren(self)



    def table_statement(self):

        localctx = SQLParser.Table_statementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 10, self.RULE_table_statement)
        try:
            self.state = 147
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SQLParser.T__3]:
                localctx = SQLParser.Create_tableContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 117
                self.match(SQLParser.T__3)
                self.state = 118
                self.match(SQLParser.T__13)
                self.state = 119
                self.match(SQLParser.Identifier)
                self.state = 120
                self.match(SQLParser.T__15)
                self.state = 121
                self.field_list()
                self.state = 122
                self.match(SQLParser.T__16)
                pass
            elif token in [SQLParser.T__5]:
                localctx = SQLParser.Drop_tableContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 124
                self.match(SQLParser.T__5)
                self.state = 125
                self.match(SQLParser.T__13)
                self.state = 126
                self.match(SQLParser.Identifier)
                pass
            elif token in [SQLParser.T__17]:
                localctx = SQLParser.Describe_tableContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 127
                self.match(SQLParser.T__17)
                self.state = 128
                self.match(SQLParser.Identifier)
                pass
            elif token in [SQLParser.T__18]:
                localctx = SQLParser.Insert_into_tableContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 129
                self.match(SQLParser.T__18)
                self.state = 130
                self.match(SQLParser.T__19)
                self.state = 131
                self.match(SQLParser.Identifier)
                self.state = 132
                self.match(SQLParser.T__20)
                self.state = 133
                self.value_lists()
                pass
            elif token in [SQLParser.T__21]:
                localctx = SQLParser.Delete_from_tableContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 134
                self.match(SQLParser.T__21)
                self.state = 135
                self.match(SQLParser.T__10)
                self.state = 136
                self.match(SQLParser.Identifier)
                self.state = 137
                self.match(SQLParser.T__22)
                self.state = 138
                self.where_and_clause()
                pass
            elif token in [SQLParser.T__23]:
                localctx = SQLParser.Update_tableContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 139
                self.match(SQLParser.T__23)
                self.state = 140
                self.match(SQLParser.Identifier)
                self.state = 141
                self.match(SQLParser.T__24)
                self.state = 142
                self.set_clause()
                self.state = 143
                self.match(SQLParser.T__22)
                self.state = 144
                self.where_and_clause()
                pass
            elif token in [SQLParser.T__25]:
                localctx = SQLParser.Select_table_Context(self, localctx)
                self.enterOuterAlt(localctx, 7)
                self.state = 146
                self.select_table()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Select_tableContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def selectors(self):
            return self.getTypedRuleContext(SQLParser.SelectorsContext,0)


        def identifiers(self):
            return self.getTypedRuleContext(SQLParser.IdentifiersContext,0)


        def where_and_clause(self):
            return self.getTypedRuleContext(SQLParser.Where_and_clauseContext,0)


        def column(self):
            return self.getTypedRuleContext(SQLParser.ColumnContext,0)


        def Integer(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Integer)
            else:
                return self.getToken(SQLParser.Integer, i)

        def getRuleIndex(self):
            return SQLParser.RULE_select_table

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSelect_table" ):
                return visitor.visitSelect_table(self)
            else:
                return visitor.visitChildren(self)




    def select_table(self):

        localctx = SQLParser.Select_tableContext(self, self._ctx, self.state)
        self.enterRule(localctx, 12, self.RULE_select_table)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 149
            self.match(SQLParser.T__25)
            self.state = 150
            self.selectors()
            self.state = 151
            self.match(SQLParser.T__10)
            self.state = 152
            self.identifiers()
            self.state = 155
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==SQLParser.T__22:
                self.state = 153
                self.match(SQLParser.T__22)
                self.state = 154
                self.where_and_clause()


            self.state = 160
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==SQLParser.T__26:
                self.state = 157
                self.match(SQLParser.T__26)
                self.state = 158
                self.match(SQLParser.T__27)
                self.state = 159
                self.column()


            self.state = 168
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            if _la==SQLParser.T__28:
                self.state = 162
                self.match(SQLParser.T__28)
                self.state = 163
                self.match(SQLParser.Integer)
                self.state = 166
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==SQLParser.T__29:
                    self.state = 164
                    self.match(SQLParser.T__29)
                    self.state = 165
                    self.match(SQLParser.Integer)




        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Index_statementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return SQLParser.RULE_index_statement

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class Alter_drop_indexContext(Index_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Index_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_drop_index" ):
                return visitor.visitAlter_drop_index(self)
            else:
                return visitor.visitChildren(self)


    class Alter_add_indexContext(Index_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Index_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)
        def identifiers(self):
            return self.getTypedRuleContext(SQLParser.IdentifiersContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_add_index" ):
                return visitor.visitAlter_add_index(self)
            else:
                return visitor.visitChildren(self)


    class Create_indexContext(Index_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Index_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)
        def identifiers(self):
            return self.getTypedRuleContext(SQLParser.IdentifiersContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitCreate_index" ):
                return visitor.visitCreate_index(self)
            else:
                return visitor.visitChildren(self)


    class Drop_indexContext(Index_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Index_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitDrop_index" ):
                return visitor.visitDrop_index(self)
            else:
                return visitor.visitChildren(self)



    def index_statement(self):

        localctx = SQLParser.Index_statementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 14, self.RULE_index_statement)
        try:
            self.state = 198
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
            if la_ == 1:
                localctx = SQLParser.Create_indexContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 170
                self.match(SQLParser.T__3)
                self.state = 171
                self.match(SQLParser.T__30)
                self.state = 172
                self.match(SQLParser.Identifier)
                self.state = 173
                self.match(SQLParser.T__31)
                self.state = 174
                self.match(SQLParser.Identifier)
                self.state = 175
                self.match(SQLParser.T__15)
                self.state = 176
                self.identifiers()
                self.state = 177
                self.match(SQLParser.T__16)
                pass

            elif la_ == 2:
                localctx = SQLParser.Drop_indexContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 179
                self.match(SQLParser.T__5)
                self.state = 180
                self.match(SQLParser.T__30)
                self.state = 181
                self.match(SQLParser.Identifier)
                pass

            elif la_ == 3:
                localctx = SQLParser.Alter_add_indexContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 182
                self.match(SQLParser.T__32)
                self.state = 183
                self.match(SQLParser.T__13)
                self.state = 184
                self.match(SQLParser.Identifier)
                self.state = 185
                self.match(SQLParser.T__33)
                self.state = 186
                self.match(SQLParser.T__30)
                self.state = 187
                self.match(SQLParser.Identifier)
                self.state = 188
                self.match(SQLParser.T__15)
                self.state = 189
                self.identifiers()
                self.state = 190
                self.match(SQLParser.T__16)
                pass

            elif la_ == 4:
                localctx = SQLParser.Alter_drop_indexContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 192
                self.match(SQLParser.T__32)
                self.state = 193
                self.match(SQLParser.T__13)
                self.state = 194
                self.match(SQLParser.Identifier)
                self.state = 195
                self.match(SQLParser.T__5)
                self.state = 196
                self.match(SQLParser.T__30)
                self.state = 197
                self.match(SQLParser.Identifier)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Alter_statementContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return SQLParser.RULE_alter_statement

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class Alter_table_drop_pkContext(Alter_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Alter_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_table_drop_pk" ):
                return visitor.visitAlter_table_drop_pk(self)
            else:
                return visitor.visitChildren(self)


    class Alter_table_add_foreign_keyContext(Alter_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Alter_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)
        def identifiers(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SQLParser.IdentifiersContext)
            else:
                return self.getTypedRuleContext(SQLParser.IdentifiersContext,i)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_table_add_foreign_key" ):
                return visitor.visitAlter_table_add_foreign_key(self)
            else:
                return visitor.visitChildren(self)


    class Alter_table_add_uniqueContext(Alter_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Alter_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_table_add_unique" ):
                return visitor.visitAlter_table_add_unique(self)
            else:
                return visitor.visitChildren(self)


    class Alter_table_dropContext(Alter_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Alter_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_table_drop" ):
                return visitor.visitAlter_table_drop(self)
            else:
                return visitor.visitChildren(self)


    class Alter_table_addContext(Alter_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Alter_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)
        def field(self):
            return self.getTypedRuleContext(SQLParser.FieldContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_table_add" ):
                return visitor.visitAlter_table_add(self)
            else:
                return visitor.visitChildren(self)


    class Alter_table_changeContext(Alter_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Alter_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)
        def field(self):
            return self.getTypedRuleContext(SQLParser.FieldContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_table_change" ):
                return visitor.visitAlter_table_change(self)
            else:
                return visitor.visitChildren(self)


    class Alter_table_renameContext(Alter_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Alter_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_table_rename" ):
                return visitor.visitAlter_table_rename(self)
            else:
                return visitor.visitChildren(self)


    class Alter_table_drop_foreign_keyContext(Alter_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Alter_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_table_drop_foreign_key" ):
                return visitor.visitAlter_table_drop_foreign_key(self)
            else:
                return visitor.visitChildren(self)


    class Alter_table_add_pkContext(Alter_statementContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Alter_statementContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)
        def identifiers(self):
            return self.getTypedRuleContext(SQLParser.IdentifiersContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAlter_table_add_pk" ):
                return visitor.visitAlter_table_add_pk(self)
            else:
                return visitor.visitChildren(self)



    def alter_statement(self):

        localctx = SQLParser.Alter_statementContext(self, self._ctx, self.state)
        self.enterRule(localctx, 16, self.RULE_alter_statement)
        self._la = 0 # Token type
        try:
            self.state = 276
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,11,self._ctx)
            if la_ == 1:
                localctx = SQLParser.Alter_table_addContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 200
                self.match(SQLParser.T__32)
                self.state = 201
                self.match(SQLParser.T__13)
                self.state = 202
                self.match(SQLParser.Identifier)
                self.state = 203
                self.match(SQLParser.T__33)
                self.state = 204
                self.field()
                pass

            elif la_ == 2:
                localctx = SQLParser.Alter_table_dropContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 205
                self.match(SQLParser.T__32)
                self.state = 206
                self.match(SQLParser.T__13)
                self.state = 207
                self.match(SQLParser.Identifier)
                self.state = 208
                self.match(SQLParser.T__5)
                self.state = 209
                self.match(SQLParser.Identifier)
                pass

            elif la_ == 3:
                localctx = SQLParser.Alter_table_changeContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 210
                self.match(SQLParser.T__32)
                self.state = 211
                self.match(SQLParser.T__13)
                self.state = 212
                self.match(SQLParser.Identifier)
                self.state = 213
                self.match(SQLParser.T__34)
                self.state = 214
                self.match(SQLParser.Identifier)
                self.state = 215
                self.field()
                pass

            elif la_ == 4:
                localctx = SQLParser.Alter_table_renameContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 216
                self.match(SQLParser.T__32)
                self.state = 217
                self.match(SQLParser.T__13)
                self.state = 218
                self.match(SQLParser.Identifier)
                self.state = 219
                self.match(SQLParser.T__35)
                self.state = 220
                self.match(SQLParser.T__12)
                self.state = 221
                self.match(SQLParser.Identifier)
                pass

            elif la_ == 5:
                localctx = SQLParser.Alter_table_drop_pkContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 222
                self.match(SQLParser.T__32)
                self.state = 223
                self.match(SQLParser.T__13)
                self.state = 224
                self.match(SQLParser.Identifier)
                self.state = 225
                self.match(SQLParser.T__5)
                self.state = 226
                self.match(SQLParser.T__36)
                self.state = 227
                self.match(SQLParser.T__37)
                self.state = 229
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==SQLParser.Identifier:
                    self.state = 228
                    self.match(SQLParser.Identifier)


                pass

            elif la_ == 6:
                localctx = SQLParser.Alter_table_drop_foreign_keyContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 231
                self.match(SQLParser.T__32)
                self.state = 232
                self.match(SQLParser.T__13)
                self.state = 233
                self.match(SQLParser.Identifier)
                self.state = 234
                self.match(SQLParser.T__5)
                self.state = 235
                self.match(SQLParser.T__38)
                self.state = 236
                self.match(SQLParser.T__37)
                self.state = 237
                self.match(SQLParser.Identifier)
                pass

            elif la_ == 7:
                localctx = SQLParser.Alter_table_add_pkContext(self, localctx)
                self.enterOuterAlt(localctx, 7)
                self.state = 238
                self.match(SQLParser.T__32)
                self.state = 239
                self.match(SQLParser.T__13)
                self.state = 240
                self.match(SQLParser.Identifier)
                self.state = 241
                self.match(SQLParser.T__33)
                self.state = 242
                self.match(SQLParser.T__39)
                self.state = 243
                self.match(SQLParser.Identifier)
                self.state = 244
                self.match(SQLParser.T__36)
                self.state = 245
                self.match(SQLParser.T__37)
                self.state = 246
                self.match(SQLParser.T__15)
                self.state = 247
                self.identifiers()
                self.state = 248
                self.match(SQLParser.T__16)
                pass

            elif la_ == 8:
                localctx = SQLParser.Alter_table_add_foreign_keyContext(self, localctx)
                self.enterOuterAlt(localctx, 8)
                self.state = 250
                self.match(SQLParser.T__32)
                self.state = 251
                self.match(SQLParser.T__13)
                self.state = 252
                self.match(SQLParser.Identifier)
                self.state = 253
                self.match(SQLParser.T__33)
                self.state = 254
                self.match(SQLParser.T__39)
                self.state = 255
                self.match(SQLParser.Identifier)
                self.state = 256
                self.match(SQLParser.T__38)
                self.state = 257
                self.match(SQLParser.T__37)
                self.state = 258
                self.match(SQLParser.T__15)
                self.state = 259
                self.identifiers()
                self.state = 260
                self.match(SQLParser.T__16)
                self.state = 261
                self.match(SQLParser.T__40)
                self.state = 262
                self.match(SQLParser.Identifier)
                self.state = 263
                self.match(SQLParser.T__15)
                self.state = 264
                self.identifiers()
                self.state = 265
                self.match(SQLParser.T__16)
                pass

            elif la_ == 9:
                localctx = SQLParser.Alter_table_add_uniqueContext(self, localctx)
                self.enterOuterAlt(localctx, 9)
                self.state = 267
                self.match(SQLParser.T__32)
                self.state = 268
                self.match(SQLParser.T__13)
                self.state = 269
                self.match(SQLParser.Identifier)
                self.state = 270
                self.match(SQLParser.T__33)
                self.state = 271
                self.match(SQLParser.T__41)
                self.state = 272
                self.match(SQLParser.Identifier)
                self.state = 273
                self.match(SQLParser.T__15)
                self.state = 274
                self.match(SQLParser.Identifier)
                self.state = 275
                self.match(SQLParser.T__16)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Field_listContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def field(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SQLParser.FieldContext)
            else:
                return self.getTypedRuleContext(SQLParser.FieldContext,i)


        def getRuleIndex(self):
            return SQLParser.RULE_field_list

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitField_list" ):
                return visitor.visitField_list(self)
            else:
                return visitor.visitChildren(self)




    def field_list(self):

        localctx = SQLParser.Field_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 18, self.RULE_field_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 278
            self.field()
            self.state = 283
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==SQLParser.T__42:
                self.state = 279
                self.match(SQLParser.T__42)
                self.state = 280
                self.field()
                self.state = 285
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class FieldContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return SQLParser.RULE_field

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class Primary_key_fieldContext(FieldContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.FieldContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def identifiers(self):
            return self.getTypedRuleContext(SQLParser.IdentifiersContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPrimary_key_field" ):
                return visitor.visitPrimary_key_field(self)
            else:
                return visitor.visitChildren(self)


    class Foreign_key_fieldContext(FieldContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.FieldContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitForeign_key_field" ):
                return visitor.visitForeign_key_field(self)
            else:
                return visitor.visitChildren(self)


    class Normal_fieldContext(FieldContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.FieldContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def Identifier(self):
            return self.getToken(SQLParser.Identifier, 0)
        def type_(self):
            return self.getTypedRuleContext(SQLParser.Type_Context,0)

        def Null(self):
            return self.getToken(SQLParser.Null, 0)
        def value(self):
            return self.getTypedRuleContext(SQLParser.ValueContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitNormal_field" ):
                return visitor.visitNormal_field(self)
            else:
                return visitor.visitChildren(self)



    def field(self):

        localctx = SQLParser.FieldContext(self, self._ctx, self.state)
        self.enterRule(localctx, 20, self.RULE_field)
        self._la = 0 # Token type
        try:
            self.state = 312
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SQLParser.Identifier]:
                localctx = SQLParser.Normal_fieldContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 286
                self.match(SQLParser.Identifier)
                self.state = 287
                self.type_()
                self.state = 290
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==SQLParser.T__43:
                    self.state = 288
                    self.match(SQLParser.T__43)
                    self.state = 289
                    self.match(SQLParser.Null)


                self.state = 294
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==SQLParser.T__44:
                    self.state = 292
                    self.match(SQLParser.T__44)
                    self.state = 293
                    self.value()


                pass
            elif token in [SQLParser.T__36]:
                localctx = SQLParser.Primary_key_fieldContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 296
                self.match(SQLParser.T__36)
                self.state = 297
                self.match(SQLParser.T__37)
                self.state = 298
                self.match(SQLParser.T__15)
                self.state = 299
                self.identifiers()
                self.state = 300
                self.match(SQLParser.T__16)
                pass
            elif token in [SQLParser.T__38]:
                localctx = SQLParser.Foreign_key_fieldContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 302
                self.match(SQLParser.T__38)
                self.state = 303
                self.match(SQLParser.T__37)
                self.state = 304
                self.match(SQLParser.T__15)
                self.state = 305
                self.match(SQLParser.Identifier)
                self.state = 306
                self.match(SQLParser.T__16)
                self.state = 307
                self.match(SQLParser.T__40)
                self.state = 308
                self.match(SQLParser.Identifier)
                self.state = 309
                self.match(SQLParser.T__15)
                self.state = 310
                self.match(SQLParser.Identifier)
                self.state = 311
                self.match(SQLParser.T__16)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Type_Context(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Integer(self):
            return self.getToken(SQLParser.Integer, 0)

        def getRuleIndex(self):
            return SQLParser.RULE_type_

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitType_" ):
                return visitor.visitType_(self)
            else:
                return visitor.visitChildren(self)




    def type_(self):

        localctx = SQLParser.Type_Context(self, self._ctx, self.state)
        self.enterRule(localctx, 22, self.RULE_type_)
        try:
            self.state = 324
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SQLParser.T__45]:
                self.enterOuterAlt(localctx, 1)
                self.state = 314
                self.match(SQLParser.T__45)
                self.state = 315
                self.match(SQLParser.T__15)
                self.state = 316
                self.match(SQLParser.Integer)
                self.state = 317
                self.match(SQLParser.T__16)
                pass
            elif token in [SQLParser.T__46]:
                self.enterOuterAlt(localctx, 2)
                self.state = 318
                self.match(SQLParser.T__46)
                self.state = 319
                self.match(SQLParser.T__15)
                self.state = 320
                self.match(SQLParser.Integer)
                self.state = 321
                self.match(SQLParser.T__16)
                pass
            elif token in [SQLParser.T__47]:
                self.enterOuterAlt(localctx, 3)
                self.state = 322
                self.match(SQLParser.T__47)
                pass
            elif token in [SQLParser.T__48]:
                self.enterOuterAlt(localctx, 4)
                self.state = 323
                self.match(SQLParser.T__48)
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Value_listsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def value_list(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SQLParser.Value_listContext)
            else:
                return self.getTypedRuleContext(SQLParser.Value_listContext,i)


        def getRuleIndex(self):
            return SQLParser.RULE_value_lists

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitValue_lists" ):
                return visitor.visitValue_lists(self)
            else:
                return visitor.visitChildren(self)




    def value_lists(self):

        localctx = SQLParser.Value_listsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 24, self.RULE_value_lists)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 326
            self.value_list()
            self.state = 331
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==SQLParser.T__42:
                self.state = 327
                self.match(SQLParser.T__42)
                self.state = 328
                self.value_list()
                self.state = 333
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Value_listContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def value(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SQLParser.ValueContext)
            else:
                return self.getTypedRuleContext(SQLParser.ValueContext,i)


        def getRuleIndex(self):
            return SQLParser.RULE_value_list

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitValue_list" ):
                return visitor.visitValue_list(self)
            else:
                return visitor.visitChildren(self)




    def value_list(self):

        localctx = SQLParser.Value_listContext(self, self._ctx, self.state)
        self.enterRule(localctx, 26, self.RULE_value_list)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 334
            self.match(SQLParser.T__15)
            self.state = 335
            self.value()
            self.state = 340
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==SQLParser.T__42:
                self.state = 336
                self.match(SQLParser.T__42)
                self.state = 337
                self.value()
                self.state = 342
                self._errHandler.sync(self)
                _la = self._input.LA(1)

            self.state = 343
            self.match(SQLParser.T__16)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ValueContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Integer(self):
            return self.getToken(SQLParser.Integer, 0)

        def String(self):
            return self.getToken(SQLParser.String, 0)

        def Float(self):
            return self.getToken(SQLParser.Float, 0)

        def Null(self):
            return self.getToken(SQLParser.Null, 0)

        def getRuleIndex(self):
            return SQLParser.RULE_value

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitValue" ):
                return visitor.visitValue(self)
            else:
                return visitor.visitChildren(self)




    def value(self):

        localctx = SQLParser.ValueContext(self, self._ctx, self.state)
        self.enterRule(localctx, 28, self.RULE_value)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 345
            _la = self._input.LA(1)
            if not(((((_la - 67)) & ~0x3f) == 0 and ((1 << (_la - 67)) & ((1 << (SQLParser.Null - 67)) | (1 << (SQLParser.Integer - 67)) | (1 << (SQLParser.String - 67)) | (1 << (SQLParser.Float - 67)))) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Where_and_clauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def where_clause(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SQLParser.Where_clauseContext)
            else:
                return self.getTypedRuleContext(SQLParser.Where_clauseContext,i)


        def getRuleIndex(self):
            return SQLParser.RULE_where_and_clause

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWhere_and_clause" ):
                return visitor.visitWhere_and_clause(self)
            else:
                return visitor.visitChildren(self)




    def where_and_clause(self):

        localctx = SQLParser.Where_and_clauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 30, self.RULE_where_and_clause)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 347
            self.where_clause()
            self.state = 352
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==SQLParser.T__49:
                self.state = 348
                self.match(SQLParser.T__49)
                self.state = 349
                self.where_clause()
                self.state = 354
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Where_clauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return SQLParser.RULE_where_clause

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)



    class Where_in_listContext(Where_clauseContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Where_clauseContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def column(self):
            return self.getTypedRuleContext(SQLParser.ColumnContext,0)

        def value_list(self):
            return self.getTypedRuleContext(SQLParser.Value_listContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWhere_in_list" ):
                return visitor.visitWhere_in_list(self)
            else:
                return visitor.visitChildren(self)


    class Where_operator_selectContext(Where_clauseContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Where_clauseContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def column(self):
            return self.getTypedRuleContext(SQLParser.ColumnContext,0)

        def operator(self):
            return self.getTypedRuleContext(SQLParser.OperatorContext,0)

        def select_table(self):
            return self.getTypedRuleContext(SQLParser.Select_tableContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWhere_operator_select" ):
                return visitor.visitWhere_operator_select(self)
            else:
                return visitor.visitChildren(self)


    class Where_nullContext(Where_clauseContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Where_clauseContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def column(self):
            return self.getTypedRuleContext(SQLParser.ColumnContext,0)

        def Null(self):
            return self.getToken(SQLParser.Null, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWhere_null" ):
                return visitor.visitWhere_null(self)
            else:
                return visitor.visitChildren(self)


    class Where_operator_expressionContext(Where_clauseContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Where_clauseContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def column(self):
            return self.getTypedRuleContext(SQLParser.ColumnContext,0)

        def operator(self):
            return self.getTypedRuleContext(SQLParser.OperatorContext,0)

        def expression(self):
            return self.getTypedRuleContext(SQLParser.ExpressionContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWhere_operator_expression" ):
                return visitor.visitWhere_operator_expression(self)
            else:
                return visitor.visitChildren(self)


    class Where_in_selectContext(Where_clauseContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Where_clauseContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def column(self):
            return self.getTypedRuleContext(SQLParser.ColumnContext,0)

        def select_table(self):
            return self.getTypedRuleContext(SQLParser.Select_tableContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWhere_in_select" ):
                return visitor.visitWhere_in_select(self)
            else:
                return visitor.visitChildren(self)


    class Where_like_stringContext(Where_clauseContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a SQLParser.Where_clauseContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def column(self):
            return self.getTypedRuleContext(SQLParser.ColumnContext,0)

        def String(self):
            return self.getToken(SQLParser.String, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitWhere_like_string" ):
                return visitor.visitWhere_like_string(self)
            else:
                return visitor.visitChildren(self)



    def where_clause(self):

        localctx = SQLParser.Where_clauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 32, self.RULE_where_clause)
        self._la = 0 # Token type
        try:
            self.state = 386
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,21,self._ctx)
            if la_ == 1:
                localctx = SQLParser.Where_operator_expressionContext(self, localctx)
                self.enterOuterAlt(localctx, 1)
                self.state = 355
                self.column()
                self.state = 356
                self.operator()
                self.state = 357
                self.expression()
                pass

            elif la_ == 2:
                localctx = SQLParser.Where_operator_selectContext(self, localctx)
                self.enterOuterAlt(localctx, 2)
                self.state = 359
                self.column()
                self.state = 360
                self.operator()
                self.state = 361
                self.match(SQLParser.T__15)
                self.state = 362
                self.select_table()
                self.state = 363
                self.match(SQLParser.T__16)
                pass

            elif la_ == 3:
                localctx = SQLParser.Where_nullContext(self, localctx)
                self.enterOuterAlt(localctx, 3)
                self.state = 365
                self.column()
                self.state = 366
                self.match(SQLParser.T__50)
                self.state = 368
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                if _la==SQLParser.T__43:
                    self.state = 367
                    self.match(SQLParser.T__43)


                self.state = 370
                self.match(SQLParser.Null)
                pass

            elif la_ == 4:
                localctx = SQLParser.Where_in_listContext(self, localctx)
                self.enterOuterAlt(localctx, 4)
                self.state = 372
                self.column()
                self.state = 373
                self.match(SQLParser.T__51)
                self.state = 374
                self.value_list()
                pass

            elif la_ == 5:
                localctx = SQLParser.Where_in_selectContext(self, localctx)
                self.enterOuterAlt(localctx, 5)
                self.state = 376
                self.column()
                self.state = 377
                self.match(SQLParser.T__51)
                self.state = 378
                self.match(SQLParser.T__15)
                self.state = 379
                self.select_table()
                self.state = 380
                self.match(SQLParser.T__16)
                pass

            elif la_ == 6:
                localctx = SQLParser.Where_like_stringContext(self, localctx)
                self.enterOuterAlt(localctx, 6)
                self.state = 382
                self.column()
                self.state = 383
                self.match(SQLParser.T__52)
                self.state = 384
                self.match(SQLParser.String)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ColumnContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)

        def getRuleIndex(self):
            return SQLParser.RULE_column

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitColumn" ):
                return visitor.visitColumn(self)
            else:
                return visitor.visitChildren(self)




    def column(self):

        localctx = SQLParser.ColumnContext(self, self._ctx, self.state)
        self.enterRule(localctx, 34, self.RULE_column)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 390
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,22,self._ctx)
            if la_ == 1:
                self.state = 388
                self.match(SQLParser.Identifier)
                self.state = 389
                self.match(SQLParser.T__53)


            self.state = 392
            self.match(SQLParser.Identifier)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class ExpressionContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def value(self):
            return self.getTypedRuleContext(SQLParser.ValueContext,0)


        def column(self):
            return self.getTypedRuleContext(SQLParser.ColumnContext,0)


        def getRuleIndex(self):
            return SQLParser.RULE_expression

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitExpression" ):
                return visitor.visitExpression(self)
            else:
                return visitor.visitChildren(self)




    def expression(self):

        localctx = SQLParser.ExpressionContext(self, self._ctx, self.state)
        self.enterRule(localctx, 36, self.RULE_expression)
        try:
            self.state = 396
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SQLParser.Null, SQLParser.Integer, SQLParser.String, SQLParser.Float]:
                self.enterOuterAlt(localctx, 1)
                self.state = 394
                self.value()
                pass
            elif token in [SQLParser.Identifier]:
                self.enterOuterAlt(localctx, 2)
                self.state = 395
                self.column()
                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class Set_clauseContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)

        def EqualOrAssign(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.EqualOrAssign)
            else:
                return self.getToken(SQLParser.EqualOrAssign, i)

        def value(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SQLParser.ValueContext)
            else:
                return self.getTypedRuleContext(SQLParser.ValueContext,i)


        def getRuleIndex(self):
            return SQLParser.RULE_set_clause

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSet_clause" ):
                return visitor.visitSet_clause(self)
            else:
                return visitor.visitChildren(self)




    def set_clause(self):

        localctx = SQLParser.Set_clauseContext(self, self._ctx, self.state)
        self.enterRule(localctx, 38, self.RULE_set_clause)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 398
            self.match(SQLParser.Identifier)
            self.state = 399
            self.match(SQLParser.EqualOrAssign)
            self.state = 400
            self.value()
            self.state = 407
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==SQLParser.T__42:
                self.state = 401
                self.match(SQLParser.T__42)
                self.state = 402
                self.match(SQLParser.Identifier)
                self.state = 403
                self.match(SQLParser.EqualOrAssign)
                self.state = 404
                self.value()
                self.state = 409
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SelectorsContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def selector(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(SQLParser.SelectorContext)
            else:
                return self.getTypedRuleContext(SQLParser.SelectorContext,i)


        def getRuleIndex(self):
            return SQLParser.RULE_selectors

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSelectors" ):
                return visitor.visitSelectors(self)
            else:
                return visitor.visitChildren(self)




    def selectors(self):

        localctx = SQLParser.SelectorsContext(self, self._ctx, self.state)
        self.enterRule(localctx, 40, self.RULE_selectors)
        self._la = 0 # Token type
        try:
            self.state = 419
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [SQLParser.T__54]:
                self.enterOuterAlt(localctx, 1)
                self.state = 410
                self.match(SQLParser.T__54)
                pass
            elif token in [SQLParser.Count, SQLParser.Average, SQLParser.Max, SQLParser.Min, SQLParser.Sum, SQLParser.Identifier]:
                self.enterOuterAlt(localctx, 2)
                self.state = 411
                self.selector()
                self.state = 416
                self._errHandler.sync(self)
                _la = self._input.LA(1)
                while _la==SQLParser.T__42:
                    self.state = 412
                    self.match(SQLParser.T__42)
                    self.state = 413
                    self.selector()
                    self.state = 418
                    self._errHandler.sync(self)
                    _la = self._input.LA(1)

                pass
            else:
                raise NoViableAltException(self)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class SelectorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def column(self):
            return self.getTypedRuleContext(SQLParser.ColumnContext,0)


        def aggregator(self):
            return self.getTypedRuleContext(SQLParser.AggregatorContext,0)


        def Count(self):
            return self.getToken(SQLParser.Count, 0)

        def getRuleIndex(self):
            return SQLParser.RULE_selector

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSelector" ):
                return visitor.visitSelector(self)
            else:
                return visitor.visitChildren(self)




    def selector(self):

        localctx = SQLParser.SelectorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 42, self.RULE_selector)
        try:
            self.state = 431
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,27,self._ctx)
            if la_ == 1:
                self.enterOuterAlt(localctx, 1)
                self.state = 421
                self.column()
                pass

            elif la_ == 2:
                self.enterOuterAlt(localctx, 2)
                self.state = 422
                self.aggregator()
                self.state = 423
                self.match(SQLParser.T__15)
                self.state = 424
                self.column()
                self.state = 425
                self.match(SQLParser.T__16)
                pass

            elif la_ == 3:
                self.enterOuterAlt(localctx, 3)
                self.state = 427
                self.match(SQLParser.Count)
                self.state = 428
                self.match(SQLParser.T__15)
                self.state = 429
                self.match(SQLParser.T__54)
                self.state = 430
                self.match(SQLParser.T__16)
                pass


        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class IdentifiersContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Identifier(self, i:int=None):
            if i is None:
                return self.getTokens(SQLParser.Identifier)
            else:
                return self.getToken(SQLParser.Identifier, i)

        def getRuleIndex(self):
            return SQLParser.RULE_identifiers

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitIdentifiers" ):
                return visitor.visitIdentifiers(self)
            else:
                return visitor.visitChildren(self)




    def identifiers(self):

        localctx = SQLParser.IdentifiersContext(self, self._ctx, self.state)
        self.enterRule(localctx, 44, self.RULE_identifiers)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 433
            self.match(SQLParser.Identifier)
            self.state = 438
            self._errHandler.sync(self)
            _la = self._input.LA(1)
            while _la==SQLParser.T__42:
                self.state = 434
                self.match(SQLParser.T__42)
                self.state = 435
                self.match(SQLParser.Identifier)
                self.state = 440
                self._errHandler.sync(self)
                _la = self._input.LA(1)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class OperatorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def EqualOrAssign(self):
            return self.getToken(SQLParser.EqualOrAssign, 0)

        def Less(self):
            return self.getToken(SQLParser.Less, 0)

        def LessEqual(self):
            return self.getToken(SQLParser.LessEqual, 0)

        def Greater(self):
            return self.getToken(SQLParser.Greater, 0)

        def GreaterEqual(self):
            return self.getToken(SQLParser.GreaterEqual, 0)

        def NotEqual(self):
            return self.getToken(SQLParser.NotEqual, 0)

        def getRuleIndex(self):
            return SQLParser.RULE_operator

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOperator" ):
                return visitor.visitOperator(self)
            else:
                return visitor.visitChildren(self)




    def operator(self):

        localctx = SQLParser.OperatorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 46, self.RULE_operator)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 441
            _la = self._input.LA(1)
            if not((((_la) & ~0x3f) == 0 and ((1 << _la) & ((1 << SQLParser.EqualOrAssign) | (1 << SQLParser.Less) | (1 << SQLParser.LessEqual) | (1 << SQLParser.Greater) | (1 << SQLParser.GreaterEqual) | (1 << SQLParser.NotEqual))) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class AggregatorContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def Count(self):
            return self.getToken(SQLParser.Count, 0)

        def Average(self):
            return self.getToken(SQLParser.Average, 0)

        def Max(self):
            return self.getToken(SQLParser.Max, 0)

        def Min(self):
            return self.getToken(SQLParser.Min, 0)

        def Sum(self):
            return self.getToken(SQLParser.Sum, 0)

        def getRuleIndex(self):
            return SQLParser.RULE_aggregator

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitAggregator" ):
                return visitor.visitAggregator(self)
            else:
                return visitor.visitChildren(self)




    def aggregator(self):

        localctx = SQLParser.AggregatorContext(self, self._ctx, self.state)
        self.enterRule(localctx, 48, self.RULE_aggregator)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 443
            _la = self._input.LA(1)
            if not(((((_la - 62)) & ~0x3f) == 0 and ((1 << (_la - 62)) & ((1 << (SQLParser.Count - 62)) | (1 << (SQLParser.Average - 62)) | (1 << (SQLParser.Max - 62)) | (1 << (SQLParser.Min - 62)) | (1 << (SQLParser.Sum - 62)))) != 0)):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx





T__0=1
T__1=2
T__2=3
T__3=4
T__4=5
T__5=6
T__6=7
T__7=8
T__8=9
T__9=10
T__10=11
T__11=12
T__12=13
T__13=14
T__14=15
T__15=16
T__16=17
T__17=18
T__18=19
T__19=20
T__20=21
T__21=22
T__22=23
T__23=24
T__24=25
T__25=26
T__26=27
T__27=28
T__28=29
T__29=30
T__30=31
T__31=32
T__32=33
T__33=34
T__34=35
T__35=36
T__36=37
T__37=38
T__38=39
T__39=40
T__40=41
T__41=42
T__42=43
T__43=44
T__44=45
T__45=46
T__46=47
T__47=48
T__48=49
T__49=50
T__50=51
T__51=52
T__52=53
T__53=54
T__54=55
EqualOrAssign=56
Less=57
LessEqual=58
Greater=59
GreaterEqual=60
NotEqual=61
Count=62
Average=63
Max=64
Min=65
Sum=66
Null=67
Identifier=68
Integer=69
String=70
Float=71
Whitespace=72
Annotation=73
';'=1
'SHOW'=2
'DATABASES'=3
'CREATE'=4
'DATABASE'=5
'DROP'=6
'USE'=7
'TABLES'=8
'INDEXES'=9
'LOAD'=10
'FROM'=11
'FILE'=12
'TO'=13
'TABLE'=14
'DUMP'=15
'('=16
')'=17
'DESC'=18
'INSERT'=19
'INTO'=20
'VALUES'=21
'DELETE'=22
'WHERE'=23
'UPDATE'=24
'SET'=25
'SELECT'=26
'GROUP'=27
'BY'=28
'LIMIT'=29
'OFFSET'=30
'INDEX'=31
'ON'=32
'ALTER'=33
'ADD'=34
'CHANGE'=35
'RENAME'=36
'PRIMARY'=37
'KEY'=38
'FOREIGN'=39
'CONSTRAINT'=40
'REFERENCES'=41
'UNIQUE'=42
','=43
'NOT'=44
'DEFAULT'=45
'INT'=46
'VARCHAR'=47
'DATE'=48
'FLOAT'=49
'AND'=50
'IS'=51
'IN'=52
'LIKE'=53
'.'=54
'*'=55
'='=56
'<'=57
'<='=58
'>'=59
'>='=60
'<>'=61
'COUNT'=62
'AVG'=63
'MAX'=64
'MIN'=65
'SUM'=66
'NULL'=67
token literal names:
null
';'
'SHOW'
'DATABASES'
'CREATE'
'DATABASE'
'DROP'
'USE'
'TABLES'
'INDEXES'
'LOAD'
'FROM'
'FILE'
'TO'
'TABLE'
'DUMP'
'('
')'
'DESC'
'INSERT'
'INTO'
'VALUES'
'DELETE'
'WHERE'
'UPDATE'
'SET'
'SELECT'
'GROUP'
'BY'
'LIMIT'
'OFFSET'
'INDEX'
'ON'
'ALTER'
'ADD'
'CHANGE'
'RENAME'
'PRIMARY'
'KEY'
'FOREIGN'
'CONSTRAINT'
'REFERENCES'
'UNIQUE'
','
'NOT'
'DEFAULT'
'INT'
'VARCHAR'
'DATE'
'FLOAT'
'AND'
'IS'
'IN'
'LIKE'
'.'
'*'
'='
'<'
'<='
'>'
'>='
'<>'
'COUNT'
'AVG'
'MAX'
'MIN'
'SUM'
'NULL'
null
null
null
null
null
null

token symbolic names:
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
null
EqualOrAssign
Less
LessEqual
Greater
GreaterEqual
NotEqual
Count
Average
Max
Min
Sum
Null
Identifier
Integer
String
Float
Whitespace
Annotation

rule names:
program
statement
system_statement
db_statement
io_statement
table_statement
select_table
index_statement
alter_statement
field_list
field
type_
value_lists
value_list
value
where_and_clause
where_clause
column
expression
set_clause
selectors
selector
identifiers
operator
aggregator


atn:
[3, 24715, 42794, 33075, 47597, 16764, 15335, 30598, 22884, 3, 75, 448, 4, 2, 9, 2, 4, 3, 9, 3, 4, 4, 9, 4, 4, 5, 9, 5, 4, 6, 9, 6, 4, 7, 9, 7, 4, 8, 9, 8, 4, 9, 9, 9, 4, 10, 9, 10, 4, 11, 9, 11, 4, 12, 9, 12, 4, 13, 9, 13, 4, 14, 9, 14, 4, 15, 9, 15, 4, 16, 9, 16, 4, 17, 9, 17, 4, 18, 9, 18, 4, 19, 9, 19, 4, 20, 9, 20, 4, 21, 9, 21, 4, 22, 9, 22, 4, 23, 9, 23, 4, 24, 9, 24, 4, 25, 9, 25, 4, 26, 9, 26, 3, 2, 7, 2, 54, 10, 2, 12, 2, 14, 2, 57, 11, 2, 3, 2, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 5, 3, 83, 10, 3, 3, 4, 3, 4, 3, 4, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 3, 5, 5, 5, 102, 10, 5, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 3, 6, 5, 6, 118, 10, 6, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 3, 7, 5, 7, 150, 10, 7, 3, 8, 3, 8, 3, 8, 3, 8, 3, 8, 3, 8, 5, 8, 158, 10, 8, 3, 8, 3, 8, 3, 8, 5, 8, 163, 10, 8, 3, 8, 3, 8, 3, 8, 3, 8, 5, 8, 169, 10, 8, 5, 8, 171, 10, 8, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 3, 9, 5, 9, 201, 10, 9, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 5, 10, 232, 10, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 3, 10, 5, 10, 279, 10, 10, 3, 11, 3, 11, 3, 11, 7, 11, 284, 10, 11, 12, 11, 14, 11, 287, 11, 11, 3, 12, 3, 12, 3, 12, 3, 12, 5, 12, 293, 10, 12, 3, 12, 3, 12, 5, 12, 297, 10, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 3, 12, 5, 12, 315, 10, 12, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 3, 13, 5, 13, 327, 10, 13, 3, 14, 3, 14, 3, 14, 7, 14, 332, 10, 14, 12, 14, 14, 14, 335, 11, 14, 3, 15, 3, 15, 3, 15, 3, 15, 7, 15, 341, 10, 15, 12, 15, 14, 15, 344, 11, 15, 3, 15, 3, 15, 3, 16, 3, 16, 3, 17, 3, 17, 3, 17, 7, 17, 353, 10, 17, 12, 17, 14, 17, 356, 11, 17, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 5, 18, 371, 10, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 3, 18, 5, 18, 389, 10, 18, 3, 19, 3, 19, 5, 19, 393, 10, 19, 3, 19, 3, 19, 3, 20, 3, 20, 5, 20, 399, 10, 20, 3, 21, 3, 21, 3, 21, 3, 21, 3, 21, 3, 21, 3, 21, 7, 21, 408, 10, 21, 12, 21, 14, 21, 411, 11, 21, 3, 22, 3, 22, 3, 22, 3, 22, 7, 22, 417, 10, 22, 12, 22, 14, 22, 420, 11, 22, 5, 22, 422, 10, 22, 3, 23, 3, 23, 3, 23, 3, 23, 3, 23, 3, 23, 3, 23, 3, 23, 3, 23, 3, 23, 5, 23, 434, 10, 23, 3, 24, 3, 24, 3, 24, 7, 24, 439, 10, 24, 12, 24, 14, 24, 442, 11, 24, 3, 25, 3, 25, 3, 26, 3, 26, 3, 26, 2, 2, 27, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 2, 5, 4, 2, 69, 69, 71, 73, 3, 2, 58, 63, 3, 2, 64, 68, 2, 483, 2, 55, 3, 2, 2, 2, 4, 82, 3, 2, 2, 2, 6, 84, 3, 2, 2, 2, 8, 101, 3, 2, 2, 2, 10, 117, 3, 2, 2, 2, 12, 149, 3, 2, 2, 2, 14, 151, 3, 2, 2, 2, 16, 200, 3, 2, 2, 2, 18, 278, 3, 2, 2, 2, 20, 280, 3, 2, 2, 2, 22, 314, 3, 2, 2, 2, 24, 326, 3, 2, 2, 2, 26, 328, 3, 2, 2, 2, 28, 336, 3, 2, 2, 2, 30, 347, 3, 2, 2, 2, 32, 349, 3, 2, 2, 2, 34, 388, 3, 2, 2, 2, 36, 392, 3, 2, 2, 2, 38, 398, 3, 2, 2, 2, 40, 400, 3, 2, 2, 2, 42, 421, 3, 2, 2, 2, 44, 433, 3, 2, 2, 2, 46, 435, 3, 2, 2, 2, 48, 443, 3, 2, 2, 2, 50, 445, 3, 2, 2, 2, 52, 54, 5, 4, 3, 2, 53, 52, 3, 2, 2, 2, 54, 57, 3, 2, 2, 2, 55, 53, 3, 2, 2, 2, 55, 56, 3, 2, 2, 2, 56, 58, 3, 2, 2, 2, 57, 55, 3, 2, 2, 2, 58, 59, 7, 2, 2, 3, 59, 3, 3, 2, 2, 2, 60, 61, 5, 6, 4, 2, 61, 62, 7, 3, 2, 2, 62, 83, 3, 2, 2, 2, 63, 64, 5, 8, 5, 2, 64, 65, 7, 3, 2, 2, 65, 83, 3, 2, 2, 2, 66, 67, 5, 10, 6, 2, 67, 68, 7, 3, 2, 2, 68, 83, 3, 2, 2, 2, 69, 70, 5, 12, 7, 2, 70, 71, 7, 3, 2, 2, 71, 83, 3, 2, 2, 2, 72, 73, 5, 16, 9, 2, 73, 74, 7, 3, 2, 2, 74, 83, 3, 2, 2, 2, 75, 76, 5, 18, 10, 2, 76, 77, 7, 3, 2, 2, 77, 83, 3, 2, 2, 2, 78, 79, 7, 75, 2, 2, 79, 83, 7, 3, 2, 2, 80, 81, 7, 69, 2, 2, 81, 83, 7, 3, 2, 2, 82, 60, 3, 2, 2, 2, 82, 63, 3, 2, 2, 2, 82, 66, 3, 2, 2, 2, 82, 69, 3, 2, 2, 2, 82, 72, 3, 2, 2, 2, 82, 75, 3, 2, 2, 2, 82, 78, 3, 2, 2, 2, 82, 80, 3, 2, 2, 2, 83, 5, 3, 2, 2, 2, 84, 85, 7, 4, 2, 2, 85, 86, 7, 5, 2, 2, 86, 7, 3, 2, 2, 2, 87, 88, 7, 6, 2, 2, 88, 89, 7, 7, 2, 2, 89, 102, 7, 70, 2, 2, 90, 91, 7, 8, 2, 2, 91, 92, 7, 7, 2, 2, 92, 102, 7, 70, 2, 2, 93, 94, 7, 4, 2, 2, 94, 102, 7, 5, 2, 2, 95, 96, 7, 9, 2, 2, 96, 102, 7, 70, 2, 2, 97, 98, 7, 4, 2, 2, 98, 102, 7, 10, 2, 2, 99, 100, 7, 4, 2, 2, 100, 102, 7, 11, 2, 2, 101, 87, 3, 2, 2, 2, 101, 90, 3, 2, 2, 2, 101, 93, 3, 2, 2, 2, 101, 95, 3, 2, 2, 2, 101, 97, 3, 2, 2, 2, 101, 99, 3, 2, 2, 2, 102, 9, 3, 2, 2, 2, 103, 104, 7, 12, 2, 2, 104, 105, 7, 13, 2, 2, 105, 106, 7, 14, 2, 2, 106, 107, 7, 72, 2, 2, 107, 108, 7, 15, 2, 2, 108, 109, 7, 16, 2, 2, 109, 118, 7, 70, 2, 2, 110, 111, 7, 17, 2, 2, 111, 112, 7, 15, 2, 2, 112, 113, 7, 14, 2, 2, 113, 114, 7, 72, 2, 2, 114, 115, 7, 13, 2, 2, 115, 116, 7, 16, 2, 2, 116, 118, 7, 70, 2, 2, 117, 103, 3, 2, 2, 2, 117, 110, 3, 2, 2, 2, 118, 11, 3, 2, 2, 2, 119, 120, 7, 6, 2, 2, 120, 121, 7, 16, 2, 2, 121, 122, 7, 70, 2, 2, 122, 123, 7, 18, 2, 2, 123, 124, 5, 20, 11, 2, 124, 125, 7, 19, 2, 2, 125, 150, 3, 2, 2, 2, 126, 127, 7, 8, 2, 2, 127, 128, 7, 16, 2, 2, 128, 150, 7, 70, 2, 2, 129, 130, 7, 20, 2, 2, 130, 150, 7, 70, 2, 2, 131, 132, 7, 21, 2, 2, 132, 133, 7, 22, 2, 2, 133, 134, 7, 70, 2, 2, 134, 135, 7, 23, 2, 2, 135, 150, 5, 26, 14, 2, 136, 137, 7, 24, 2, 2, 137, 138, 7, 13, 2, 2, 138, 139, 7, 70, 2, 2, 139, 140, 7, 25, 2, 2, 140, 150, 5, 32, 17, 2, 141, 142, 7, 26, 2, 2, 142, 143, 7, 70, 2, 2, 143, 144, 7, 27, 2, 2, 144, 145, 5, 40, 21, 2, 145, 146, 7, 25, 2, 2, 146, 147, 5, 32, 17, 2, 147, 150, 3, 2, 2, 2, 148, 150, 5, 14, 8, 2, 149, 119, 3, 2, 2, 2, 149, 126, 3, 2, 2, 2, 149, 129, 3, 2, 2, 2, 149, 131, 3, 2, 2, 2, 149, 136, 3, 2, 2, 2, 149, 141, 3, 2, 2, 2, 149, 148, 3, 2, 2, 2, 150, 13, 3, 2, 2, 2, 151, 152, 7, 28, 2, 2, 152, 153, 5, 42, 22, 2, 153, 154, 7, 13, 2, 2, 154, 157, 5, 46, 24, 2, 155, 156, 7, 25, 2, 2, 156, 158, 5, 32, 17, 2, 157, 155, 3, 2, 2, 2, 157, 158, 3, 2, 2, 2, 158, 162, 3, 2, 2, 2, 159, 160, 7, 29, 2, 2, 160, 161, 7, 30, 2, 2, 161, 163, 5, 36, 19, 2, 162, 159, 3, 2, 2, 2, 162, 163, 3, 2, 2, 2, 163, 170, 3, 2, 2, 2, 164, 165, 7, 31, 2, 2, 165, 168, 7, 71, 2, 2, 166, 167, 7, 32, 2, 2, 167, 169, 7, 71, 2, 2, 168, 166, 3, 2, 2, 2, 168, 169, 3, 2, 2, 2, 169, 171, 3, 2, 2, 2, 170, 164, 3, 2, 2, 2, 170, 171, 3, 2, 2, 2, 171, 15, 3, 2, 2, 2, 172, 173, 7, 6, 2, 2, 173, 174, 7, 33, 2, 2, 174, 175, 7, 70, 2, 2, 175, 176, 7, 34, 2, 2, 176, 177, 7, 70, 2, 2, 177, 178, 7, 18, 2, 2, 178, 179, 5, 46, 24, 2, 179, 180, 7, 19, 2, 2, 180, 201, 3, 2, 2, 2, 181, 182, 7, 8, 2, 2, 182, 183, 7, 33, 2, 2, 183, 201, 7, 70, 2, 2, 184, 185, 7, 35, 2, 2, 185, 186, 7, 16, 2, 2, 186, 187, 7, 70, 2, 2, 187, 188, 7, 36, 2, 2, 188, 189, 7, 33, 2, 2, 189, 190, 7, 70, 2, 2, 190, 191, 7, 18, 2, 2, 191, 192, 5, 46, 24, 2, 192, 193, 7, 19, 2, 2, 193, 201, 3, 2, 2, 2, 194, 195, 7, 35, 2, 2, 195, 196, 7, 16, 2, 2, 196, 197, 7, 70, 2, 2, 197, 198, 7, 8, 2, 2, 198, 199, 7, 33, 2, 2, 199, 201, 7, 70, 2, 2, 200, 172, 3, 2, 2, 2, 200, 181, 3, 2, 2, 2, 200, 184, 3, 2, 2, 2, 200, 194, 3, 2, 2, 2, 201, 17, 3, 2, 2, 2, 202, 203, 7, 35, 2, 2, 203, 204, 7, 16, 2, 2, 204, 205, 7, 70, 2, 2, 205, 206, 7, 36, 2, 2, 206, 279, 5, 22, 12, 2, 207, 208, 7, 35, 2, 2, 208, 209, 7, 16, 2, 2, 209, 210, 7, 70, 2, 2, 210, 211, 7, 8, 2, 2, 211, 279, 7, 70, 2, 2, 212, 213, 7, 35, 2, 2, 213, 214, 7, 16, 2, 2, 214, 215, 7, 70, 2, 2, 215, 216, 7, 37, 2, 2, 216, 217, 7, 70, 2, 2, 217, 279, 5, 22, 12, 2, 218, 219, 7, 35, 2, 2, 219, 220, 7, 16, 2, 2, 220, 221, 7, 70, 2, 2, 221, 222, 7, 38, 2, 2, 222, 223, 7, 15, 2, 2, 223, 279, 7, 70, 2, 2, 224, 225, 7, 35, 2, 2, 225, 226, 7, 16, 2, 2, 226, 227, 7, 70, 2, 2, 227, 228, 7, 8, 2, 2, 228, 229, 7, 39, 2, 2, 229, 231, 7, 40, 2, 2, 230, 232, 7, 70, 2, 2, 231, 230, 3, 2, 2, 2, 231, 232, 3, 2, 2, 2, 232, 279, 3, 2, 2, 2, 233, 234, 7, 35, 2, 2, 234, 235, 7, 16, 2, 2, 235, 236, 7, 70, 2, 2, 236, 237, 7, 8, 2, 2, 237, 238, 7, 41, 2, 2, 238, 239, 7, 40, 2, 2, 239, 279, 7, 70, 2, 2, 240, 241, 7, 35, 2, 2, 241, 242, 7, 16, 2, 2, 242, 243, 7, 70, 2, 2, 243, 244, 7, 36, 2, 2, 244, 245, 7, 42, 2, 2, 245, 246, 7, 70, 2, 2, 246, 247, 7, 39, 2, 2, 247, 248, 7, 40, 2, 2, 248, 249, 7, 18, 2, 2, 249, 250, 5, 46, 24, 2, 250, 251, 7, 19, 2, 2, 251, 279, 3, 2, 2, 2, 252, 253, 7, 35, 2, 2, 253, 254, 7, 16, 2, 2, 254, 255, 7, 70, 2, 2, 255, 256, 7, 36, 2, 2, 256, 257, 7, 42, 2, 2, 257, 258, 7, 70, 2, 2, 258, 259, 7, 41, 2, 2, 259, 260, 7, 40, 2, 2, 260, 261, 7, 18, 2, 2, 261, 262, 5, 46, 24, 2, 262, 263, 7, 19, 2, 2, 263, 264, 7, 43, 2, 2, 264, 265, 7, 70, 2, 2, 265, 266, 7, 18, 2, 2, 266, 267, 5, 46, 24, 2, 267, 268, 7, 19, 2, 2, 268, 279, 3, 2, 2, 2, 269, 270, 7, 35, 2, 2, 270, 271, 7, 16, 2, 2, 271, 272, 7, 70, 2, 2, 272, 273, 7, 36, 2, 2, 273, 274, 7, 44, 2, 2, 274, 275, 7, 70, 2, 2, 275, 276, 7, 18, 2, 2, 276, 277, 7, 70, 2, 2, 277, 279, 7, 19, 2, 2, 278, 202, 3, 2, 2, 2, 278, 207, 3, 2, 2, 2, 278, 212, 3, 2, 2, 2, 278, 218, 3, 2, 2, 2, 278, 224, 3, 2, 2, 2, 278, 233, 3, 2, 2, 2, 278, 240, 3, 2, 2, 2, 278, 252, 3, 2, 2, 2, 278, 269, 3, 2, 2, 2, 279, 19, 3, 2, 2, 2, 280, 285, 5, 22, 12, 2, 281, 282, 7, 45, 2, 2, 282, 284, 5, 22, 12, 2, 283, 281, 3, 2, 2, 2, 284, 287, 3, 2, 2, 2, 285, 283, 3, 2, 2, 2, 285, 286, 3, 2, 2, 2, 286, 21, 3, 2, 2, 2, 287, 285, 3, 2, 2, 2, 288, 289, 7, 70, 2, 2, 289, 292, 5, 24, 13, 2, 290, 291, 7, 46, 2, 2, 291, 293, 7, 69, 2, 2, 292, 290, 3, 2, 2, 2, 292, 293, 3, 2, 2, 2, 293, 296, 3, 2, 2, 2, 294, 295, 7, 47, 2, 2, 295, 297, 5, 30, 16, 2, 296, 294, 3, 2, 2, 2, 296, 297, 3, 2, 2, 2, 297, 315, 3, 2, 2, 2, 298, 299, 7, 39, 2, 2, 299, 300, 7, 40, 2, 2, 300, 301, 7, 18, 2, 2, 301, 302, 5, 46, 24, 2, 302, 303, 7, 19, 2, 2, 303, 315, 3, 2, 2, 2, 304, 305, 7, 41, 2, 2, 305, 306, 7, 40, 2, 2, 306, 307, 7, 18, 2, 2, 307, 308, 7, 70, 2, 2, 308, 309, 7, 19, 2, 2, 309, 310, 7, 43, 2, 2, 310, 311, 7, 70, 2, 2, 311, 312, 7, 18, 2, 2, 312, 313, 7, 70, 2, 2, 313, 315, 7, 19, 2, 2, 314, 288, 3, 2, 2, 2, 314, 298, 3, 2, 2, 2, 314, 304, 3, 2, 2, 2, 315, 23, 3, 2, 2, 2, 316, 317, 7, 48, 2, 2, 317, 318, 7, 18, 2, 2, 318, 319, 7, 71, 2, 2, 319, 327, 7, 19, 2, 2, 320, 321, 7, 49, 2, 2, 321, 322, 7, 18, 2, 2, 322, 323, 7, 71, 2, 2, 323, 327, 7, 19, 2, 2, 324, 327, 7, 50, 2, 2, 325, 327, 7, 51, 2, 2, 326, 316, 3, 2, 2, 2, 326, 320, 3, 2, 2, 2, 326, 324, 3, 2, 2, 2, 326, 325, 3, 2, 2, 2, 327, 25, 3, 2, 2, 2, 328, 333, 5, 28, 15, 2, 329, 330, 7, 45, 2, 2, 330, 332, 5, 28, 15, 2, 331, 329, 3, 2, 2, 2, 332, 335, 3, 2, 2, 2, 333, 331, 3, 2, 2, 2, 333, 334, 3, 2, 2, 2, 334, 27, 3, 2, 2, 2, 335, 333, 3, 2, 2, 2, 336, 337, 7, 18, 2, 2, 337, 342, 5, 30, 16, 2, 338, 339, 7, 45, 2, 2, 339, 341, 5, 30, 16, 2, 340, 338, 3, 2, 2, 2, 341, 344, 3, 2, 2, 2, 342, 340, 3, 2, 2, 2, 342, 343, 3, 2, 2, 2, 343, 345, 3, 2, 2, 2, 344, 342, 3, 2, 2, 2, 345, 346, 7, 19, 2, 2, 346, 29, 3, 2, 2, 2, 347, 348, 9, 2, 2, 2, 348, 31, 3, 2, 2, 2, 349, 354, 5, 34, 18, 2, 350, 351, 7, 52, 2, 2, 351, 353, 5, 34, 18, 2, 352, 350, 3, 2, 2, 2, 353, 356, 3, 2, 2, 2, 354, 352, 3, 2, 2, 2, 354, 355, 3, 2, 2, 2, 355, 33, 3, 2, 2, 2, 356, 354, 3, 2, 2, 2, 357, 358, 5, 36, 19, 2, 358, 359, 5, 48, 25, 2, 359, 360, 5, 38, 20, 2, 360, 389, 3, 2, 2, 2, 361, 362, 5, 36, 19, 2, 362, 363, 5, 48, 25, 2, 363, 364, 7, 18, 2, 2, 364, 365, 5, 14, 8, 2, 365, 366, 7, 19, 2, 2, 366, 389, 3, 2, 2, 2, 367, 368, 5, 36, 19, 2, 368, 370, 7, 53, 2, 2, 369, 371, 7, 46, 2, 2, 370, 369, 3, 2, 2, 2, 370, 371, 3, 2, 2, 2, 371, 372, 3, 2, 2, 2, 372, 373, 7, 69, 2, 2, 373, 389, 3, 2, 2, 2, 374, 375, 5, 36, 19, 2, 375, 376, 7, 54, 2, 2, 376, 377, 5, 28, 15, 2, 377, 389, 3, 2, 2, 2, 378, 379, 5, 36, 19, 2, 379, 380, 7, 54, 2, 2, 380, 381, 7, 18, 2, 2, 381, 382, 5, 14, 8, 2, 382, 383, 7, 19, 2, 2, 383, 389, 3, 2, 2, 2, 384, 385, 5, 36, 19, 2, 385, 386, 7, 55, 2, 2, 386, 387, 7, 72, 2, 2, 387, 389, 3, 2, 2, 2, 388, 357, 3, 2, 2, 2, 388, 361, 3, 2, 2, 2, 388, 367, 3, 2, 2, 2, 388, 374, 3, 2, 2, 2, 388, 378, 3, 2, 2, 2, 388, 384, 3, 2, 2, 2, 389, 35, 3, 2, 2, 2, 390, 391, 7, 70, 2, 2, 391, 393, 7, 56, 2, 2, 392, 390, 3, 2, 2, 2, 392, 393, 3, 2, 2, 2, 393, 394, 3, 2, 2, 2, 394, 395, 7, 70, 2, 2, 395, 37, 3, 2, 2, 2, 396, 399, 5, 30, 16, 2, 397, 399, 5, 36, 19, 2, 398, 396, 3, 2, 2, 2, 398, 397, 3, 2, 2, 2, 399, 39, 3, 2, 2, 2, 400, 401, 7, 70, 2, 2, 401, 402, 7, 58, 2, 2, 402, 409, 5, 30, 16, 2, 403, 404, 7, 45, 2, 2, 404, 405, 7, 70, 2, 2, 405, 406, 7, 58, 2, 2, 406, 408, 5, 30, 16, 2, 407, 403, 3, 2, 2, 2, 408, 411, 3, 2, 2, 2, 409, 407, 3, 2, 2, 2, 409, 410, 3, 2, 2, 2, 410, 41, 3, 2, 2, 2, 411, 409, 3, 2, 2, 2, 412, 422, 7, 57, 2, 2, 413, 418, 5, 44, 23, 2, 414, 415, 7, 45, 2, 2, 415, 417, 5, 44, 23, 2, 416, 414, 3, 2, 2, 2, 417, 420, 3, 2, 2, 2, 418, 416, 3, 2, 2, 2, 418, 419, 3, 2, 2, 2, 419, 422, 3, 2, 2, 2, 420, 418, 3, 2, 2, 2, 421, 412, 3, 2, 2, 2, 421, 413, 3, 2, 2, 2, 422, 43, 3, 2, 2, 2, 423, 434, 5, 36, 19, 2, 424, 425, 5, 50, 26, 2, 425, 426, 7, 18, 2, 2, 426, 427, 5, 36, 19, 2, 427, 428, 7, 19, 2, 2, 428, 434, 3, 2, 2, 2, 429, 430, 7, 64, 2, 2, 430, 431, 7, 18, 2, 2, 431, 432, 7, 57, 2, 2, 432, 434, 7, 19, 2, 2, 433, 423, 3, 2, 2, 2, 433, 424, 3, 2, 2, 2, 433, 429, 3, 2, 2, 2, 434, 45, 3, 2, 2, 2, 435, 440, 7, 70, 2, 2, 436, 437, 7, 45, 2, 2, 437, 439, 7, 70, 2, 2, 438, 436, 3, 2, 2, 2, 439, 442, 3, 2, 2, 2, 440, 438, 3, 2, 2, 2, 440, 441, 3, 2, 2, 2, 441, 47, 3, 2, 2, 2, 442, 440, 3, 2, 2, 2, 443, 444, 9, 3, 2, 2, 444, 49, 3, 2, 2, 2, 445, 446, 9, 4, 2, 2, 446, 51, 3, 2, 2, 2, 31, 55, 82, 101, 117, 149, 157, 162, 168, 170, 200, 231, 278, 285, 292, 296, 314, 326, 333, 342, 354, 370, 388, 392, 398, 409, 418, 421, 433, 440]
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
