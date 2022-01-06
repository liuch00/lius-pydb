
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
        self.index2FPID = np.zeros(CAP)
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
        self.index2FPID[index] = -1
        fpID = self.index2FPID[index]
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
                fID = self.split_FPID(fpID)[0]
                pID = self.split_FPID(fpID)[1]
                self.FM.writePage(fID, pID, self.addr[index])
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
            self._root_node._father = new_root_father.page()
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
            return None


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
        lower = self.lower_bound(key=lo)
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
        index_file_path = home_directory + '/' + database_name + '/' + index_file_name
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
from index_handler import IndexHandler
from file_index import FileIndex
from typing import Dict
from ..FileSystem import BufManager
from FileIndexID import FileIndexID


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
            for ID in self._started_file_index:
                file_index = self._started_file_index.get(ID)
                if file_index.handler is index_handler:
                    # shut index
                    if ID in self._started_file_index:
                        tmp_file_index = self._started_file_index.pop(ID)
                        if tmp_file_index.is_modified:
                            tmp_file_index.pour()
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
        for db_name in self._started_index_handler:
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
            if self._child_list == value:
                cursor = index
                break
        if cursor != upper:
            self._child_key_list.pop(cursor)
            self._child_list.pop(cursor)
            if len_key_list > 0:
                if cursor == 0:
                    return self._child_list[0]
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

from lookup_element import Reducer, Term, LookupOutput, Join


class Join:
    def __init__(self, res_map: dict, term, union=None):
        self.res_map = map
        self._cond = term
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
from MetaSystem.info import TableInfo, ColumnInfo
from SQL_Parser.SQLVisitor import SQLVisitor
from SQL_Parser.SQLParser import SQLParser
from SQL_Parser.SQLLexer import SQLLexer
from antlr4 import ParserRuleContext
import time
from system_manager import SystemManger
from lookup_element import Reducer, Term, LookupOutput, Join

# todo:move to SQL_parser
class SystemVisitor(SQLVisitor):
    def __init__(self, system_manager=None):
        super(SQLVisitor, self).__init__()
        self.system_manager: SystemManger = system_manager
        self.time_begin = None

    def to_str(self, context):
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

    def to_float(self,context):
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

    def aggregate_result(self, aggregate, next_result):
        if next_result is None:
            return aggregate
        else:
            return next_result

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
        pass
    ## todo:fix
    # columns, foreign_keys, primary = ctx.field_list().accept(self)
    # table_name = self.to_str(ctx.Identifier())
    # res = self.system_manager.createTable(TableInfo(table_name, columns))
    # for col in foreign_keys:
    #     self.system_manager.addForeign(table_name, col, foreign_keys[col])
    # self.system_manager.setPrimary(table_name, primary)
    # return res

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
            self.system_manager.addRecord(self.to_str(ctx.getChild(2)), item)
        return LookupOutput('inserted_items', (len(data),))

    # Visit a parse tree produced by SQLParser#delete_from_table.
    def visitDelete_from_table(self, ctx: SQLParser.Delete_from_tableContext):
        return self.system_manager.removeRecord(self.to_str(ctx.Identifier()), ctx.where_and_clause().accept(self))

    # Visit a parse tree produced by SQLParser#update_table.
    def visitUpdate_table(self, ctx: SQLParser.Update_tableContext):
        return self.system_manager.updateRecord(self.to_str(ctx.Identifier()), ctx.where_and_clause().accept(self),
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
        return self.system_manager.drop_index(self.to_str(ctx.Identifier()))

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
        self.system_manager.addColumn(self.to_str(ctx.Identifier()), col)

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
        self.system_manager.removeForeign(None, None, self.to_str(ctx.Identifier(1)))

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
                name_to_column[name] = ColumnInfo(type_, name, size)
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
        value = self.system_manager.resultToValue(result, False)
        return Term(1, table_name, column_name, operator, value=value)


    # Visit a parse tree produced by SQLParser#where_null.
    def visitWhere_null(self, ctx: SQLParser.Where_nullContext):
        table_name, col_name = ctx.column().accept(self)
        is_null = ctx.getChild(2) != "NOT"
        return Term(0, table_name, col_name, is_null)

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


from pathlib import Path
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.Errors import ParseCancellationException
from antlr4.error.ErrorListener import ErrorListener
from copy import deepcopy
from .system_visitor import SystemVisitor
from FileSystem.FileManager import FileManager
from FileSystem.BufManager import BufManager
from RecordSystem.RecordManager import RecordManager
from RecordSystem.FileScan import FileScan
from RecordSystem.FileHandler import FileHandler
from RecordSystem.record import Record
from RecordSystem.rid import RID
from IndexSystem.index_manager import IndexManager
from MetaSystem.MetaHandler import MetaHandler
from Exceptions.exception import *
from .lookup_element import LookupOutput
from SQL_Parser.SQLLexer import SQLLexer
from SQL_Parser.SQLParser import SQLParser
from MetaSystem.info import TableInfo, ColumnInfo


class SystemManger:
    def __init__(self, visitor: SystemVisitor, syspath: Path, bm: BufManager, rm: RecordManager, im: IndexManager):
        self.visitor = visitor
        self.systemPath = syspath
        self.BM = bm
        self.IM = im
        self.RM = rm
        self.metaHandlers = {}
        self.databaselist = []
        for dir in syspath.iterdir():
            self.databaselist.append(dir.name)
        self.inUse = None
        self.visitor.manager = self

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
        tableInfo.index[col] = indexFile._root

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
        metaHandler.collectTableInfo(table).indexes.pop(col)
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
        tableInfo.addForeign(col, foreign)
        metaHandler.shutdown()
        if forName:
            if forName not in metaHandler.databaseInfo.indexMap:
                self.createIndex(forName, foreign[0], foreign[1])
        else:
            indexName = foreign[0] + "." + foreign[1]
            if indexName not in metaHandler.databaseInfo.indexMap:
                self.createIndex(indexName, foreign[0], foreign[1])
        return

    def removeForeign(self, table: str, col: str, forName=None):
        metaHandler, tableInfo = self.collectTableinfo(table)
        if forName:
            if metaHandler.databaseInfo.indexMap.get(forName):
                self.removeIndex(forName)
        else:
            if tableInfo.foreign.get(col) is not None:
                foreign = tableInfo.foreign[col][0] + "." + tableInfo.foreign[col][1]
                if metaHandler.databaseInfo.indexMap.get(foreign):
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
        metaHandler, tableInfo = self.collectTableinfo(table)
        if tableInfo.primary:
            for column in tableInfo.primary:
                indexName = table + "." + column
                if indexName in metaHandler.databaseInfo.indexMap:
                    self.removeIndex(indexName)
        return

    def addColumn(self, table: str, col: ColumnInfo):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if tableInfo.getColumnIndex(col.name):
            print("OH NO")
            raise ColumnAlreadyExist(col.name + " exists")
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
        tableInfo = metaHandler.collectTableInfo(table)
        if col not in tableInfo.columnIndex:
            print("OH NO")
            raise ColumnNotExist(col + " doesn't exist")
        oldTableInfo : TableInfo = deepcopy(tableInfo)
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
        self.handleInsertIndex(tableInfo, self.inUse, valTuple, rid)
        return

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

    def to_string(self, prefix=True):
        # todo:
        pass

    def select(self, data: tuple):
        # todo:
        pass


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
        pass


class Join:
    def __init__(self):
        pass

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
                colInfo = tableInfo.columnMap.get(col)
                if result.get(col) is None:
                    result[col] = [colInfo]
                else:
                    result[col].append(colInfo)
        return result

from datetime import date
from Exceptions.exception import *
from .macro import *
import numpy as np
import struct
from numbers import Number
from RecordSystem.record import Record


class ColumnInfo:
    def __init__(self, type: str, name: str, size: int, default=None):
        self.type = type
        self.name = name
        self.size = size
        self.default = default

    def getSize(self):
        if self.type == "VARCHAR":
            return self.size + 1
        return 8

    def getDESC(self):
        """name, type, null, keytype, default, extra"""
        return [self.name, self.type, "N", "", self.default, ""]


class TableInfo:
    def __init__(self, name: str, contents: list):
        self.contents = contents
        self.name = name
        self.primary = None

        self.columnMap = {col.name: col for col in self.contents}
        self.columnType = [col.getSize() for col in self.contents]
        self.columnSize = [col.type for col in self.contents]
        self.foreign = {}
        self.index = {}
        self.rowSize = sum(self.columnSize)
        self.unique = {}
        self.columnIndex = {self.contents[i].name: i for i in range(len(self.contents))}

    def describe(self):
        desc = {col.name: col.getDESC() for col in self.contents}
        for name in self.primary:
            desc[name][3] = 'primary'
        for name in self.foreign:
            if desc[name][3] is not None:
                desc[name][3] = 'multi'
            else:
                desc[name][3] = 'foreign'
        for name in self.unique:
            if desc[name][3] is "":
                desc[name][3] = 'unique'
        return tuple(desc.values())

    def insertColumn(self, col: ColumnInfo):
        if col.name not in self.columnMap:
            self.contents.append(col)
            self.columnMap = {col.name: col for col in self.contents}
            self.columnType = [col.getSize() for col in self.contents]
            self.columnSize = [col.type for col in self.contents]
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
            self.columnType = [col.getSize() for col in self.contents]
            self.columnSize = [col.type for col in self.contents]
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

    def addUnique(self, column: str, uniq: str):
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
                    byte = (0, ) + tuple(value.encode())
                    if len(byte) > size:
                        print("OH NO")
                        raise VarcharTooLong("too long. max size is " + str(size - 1))
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
                val = val.replace("/", "-")
                vals = val.split("-")
                d = date(*map(int, vals))
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
                    val = data.tobytes()[1:0].rstrip(b'\x00').decode('utf-8')
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

from FileSystem.BufManager import BufManager

class MetaManager:
    def __init__(self, bm: BufManager, syspath: str):
        self.BM = bm
        self.systemPath = syspath
        self.handlers = {}

    def shutHandler(self, database: str):
        if self.handlers.get(database) is not None:
            self.handlers.pop(database).shutdown()
        return

    def shutdown(self):
        namelist = self.handlers.keys()
        for name in namelist:
            self.shutHandler(name)
        return
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

from .RecordManager import RecordManager
from .macro import *
from .rid import RID
from .record import Record

class FileHandler:

    def __init__(self, rm: RecordManager, fid: int, name: str):
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
        if bitmap[rid.slot] is False:
            bitmap[rid.slot] = True
        self.headChanged = True

        bitmap = np.packbits(bitmap)
        page[RECORD_PAGE_FIXED_HEADER: RECORD_PAGE_FIXED_HEADER + bitmapLen] = bitmap
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
            page = self.handler.RM.BM.getPage(pID)
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

        self.BM.FM.newPage(fileID, self.toSerial(self.getHead(recordLen)))
        self.BM.closeFile(fileID)
        return

    def getHead(self, recordLen: int):
        recordNum = self.getRecordNum(recordLen)
        bitmapLen = self.getBitmapLen(recordNum)
        return {'RecordLen': recordLen, 'RecordNum': recordNum, 'PageNum': 1,
                'AllRecord': 0, 'NextAvai': 0, 'BitmapLen': bitmapLen}

    def openFile(self, name: str):
        if name in self.opened:
            handler = self.opened[name]
            return handler
        fID = self.BM.openFile(name)
        self.opened[name] = FileHandler(self, fID, name)
        return self.opened[name]

    def destroyFile(self, name: str):
        self.BM.FM.destroyFile(name)
        self.opened.pop(name)
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
    def getBitmapLen(recordNum):
        length = (recordNum + 7) / 8
        return length


    def replaceFile(self, src: str, dst: str):
        if self.opened.get(src) is not None:
            self.closeFile(src)
        if self.opened.get(dst) is not None:
            self.closeFile(dst)
        self.destroyFile(dst)
        self.BM.FM.renameFile(src, dst)
        return

    def shutdown(self):
        for name in self.opened.keys():
            self.closeFile(name)

    @staticmethod
    def toSerial(d: dict):
        serial = dumps(d, ensure_ascii=False).encode('utf-8')
        empty = np.zeros(PAGE_SIZE, dtype=np.unit8)
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
        self.index2FPID = np.zeros(CAP)
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
        self.index2FPID[index] = -1
        fpID = self.index2FPID[index]
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
                fID = self.split_FPID(fpID)[0]
                pID = self.split_FPID(fpID)[1]
                self.FM.writePage(fID, pID, self.addr[index])
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
            self._root_node._father = new_root_father.page()
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
            return None


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
        lower = self.lower_bound(key=lo)
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
        index_file_path = home_directory + '/' + database_name + '/' + index_file_name
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
from index_handler import IndexHandler
from file_index import FileIndex
from typing import Dict
from ..FileSystem import BufManager
from FileIndexID import FileIndexID


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
            for ID in self._started_file_index:
                file_index = self._started_file_index.get(ID)
                if file_index.handler is index_handler:
                    # shut index
                    if ID in self._started_file_index:
                        tmp_file_index = self._started_file_index.pop(ID)
                        if tmp_file_index.is_modified:
                            tmp_file_index.pour()
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
        for db_name in self._started_index_handler:
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
            if self._child_list == value:
                cursor = index
                break
        if cursor != upper:
            self._child_key_list.pop(cursor)
            self._child_list.pop(cursor)
            if len_key_list > 0:
                if cursor == 0:
                    return self._child_list[0]
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

from lookup_element import Reducer, Term, LookupOutput, Join


class Join:
    def __init__(self, res_map: dict, term, union=None):
        self.res_map = map
        self._cond = term
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

from pathlib import Path
from antlr4 import InputStream, CommonTokenStream
from antlr4.error.Errors import ParseCancellationException
from antlr4.error.ErrorListener import ErrorListener
from copy import deepcopy
from SQL_Parser.system_visitor import SystemVisitor
from FileSystem.BufManager import BufManager
from RecordSystem.RecordManager import RecordManager
from RecordSystem.FileScan import FileScan
from RecordSystem.FileHandler import FileHandler
from RecordSystem.record import Record
from RecordSystem.rid import RID
from IndexSystem.index_manager import IndexManager
from MetaSystem.MetaHandler import MetaHandler
from Exceptions.exception import *
from .lookup_element import LookupOutput
from SQL_Parser.SQLLexer import SQLLexer
from SQL_Parser.SQLParser import SQLParser
from MetaSystem.info import TableInfo, ColumnInfo


class SystemManger:
    def __init__(self, visitor: SystemVisitor, syspath: Path, bm: BufManager, rm: RecordManager, im: IndexManager):
        self.visitor = visitor
        self.systemPath = syspath
        self.BM = bm
        self.IM = im
        self.RM = rm
        self.metaHandlers = {}
        self.databaselist = []
        for dir in syspath.iterdir():
            self.databaselist.append(dir.name)
        self.inUse = None
        self.visitor.manager = self

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
        tableInfo.index[col] = indexFile._root

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
        metaHandler.collectTableInfo(table).indexes.pop(col)
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
        tableInfo.addForeign(col, foreign)
        metaHandler.shutdown()
        if forName:
            if forName not in metaHandler.databaseInfo.indexMap:
                self.createIndex(forName, foreign[0], foreign[1])
        else:
            indexName = foreign[0] + "." + foreign[1]
            if indexName not in metaHandler.databaseInfo.indexMap:
                self.createIndex(indexName, foreign[0], foreign[1])
        return

    def removeForeign(self, table: str, col: str, forName=None):
        metaHandler, tableInfo = self.collectTableinfo(table)
        if forName:
            if metaHandler.databaseInfo.indexMap.get(forName):
                self.removeIndex(forName)
        else:
            if tableInfo.foreign.get(col) is not None:
                foreign = tableInfo.foreign[col][0] + "." + tableInfo.foreign[col][1]
                if metaHandler.databaseInfo.indexMap.get(foreign):
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
        metaHandler, tableInfo = self.collectTableinfo(table)
        if tableInfo.primary:
            for column in tableInfo.primary:
                indexName = table + "." + column
                if indexName in metaHandler.databaseInfo.indexMap:
                    self.removeIndex(indexName)
        return

    def addColumn(self, table: str, col: ColumnInfo):
        self.checkInUse()
        metaHandler = self.fetchMetaHandler()
        tableInfo = metaHandler.collectTableInfo(table)
        if tableInfo.getColumnIndex(col.name):
            print("OH NO")
            raise ColumnAlreadyExist(col.name + " exists")
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
        tableInfo = metaHandler.collectTableInfo(table)
        if col not in tableInfo.columnIndex:
            print("OH NO")
            raise ColumnNotExist(col + " doesn't exist")
        oldTableInfo : TableInfo = deepcopy(tableInfo)
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
        self.handleInsertIndex(tableInfo, self.inUse, valTuple, rid)
        return

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

    def to_string(self, prefix=True):
        # todo:
        pass

    def select(self, data: tuple):
        # todo:
        pass


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
        pass


class Join:
    def __init__(self):
        pass

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
                colInfo = tableInfo.columnMap.get(col)
                if result.get(col) is None:
                    result[col] = [colInfo]
                else:
                    result[col].append(colInfo)
        return result

from datetime import date
from Exceptions.exception import *
from .macro import *
import numpy as np
import struct
from numbers import Number
from RecordSystem.record import Record


class ColumnInfo:
    def __init__(self, type: str, name: str, size: int, default=None):
        self.type = type
        self.name = name
        self.size = size
        self.default = default

    def getSize(self):
        if self.type == "VARCHAR":
            return self.size + 1
        return 8

    def getDESC(self):
        """name, type, null, keytype, default, extra"""
        return [self.name, self.type, "N", "", self.default, ""]


class TableInfo:
    def __init__(self, name: str, contents: list):
        self.contents = contents
        self.name = name
        self.primary = None

        self.columnMap = {col.name: col for col in self.contents}
        self.columnType = [col.getSize() for col in self.contents]
        self.columnSize = [col.type for col in self.contents]
        self.foreign = {}
        self.index = {}
        self.rowSize = sum(self.columnSize)
        self.unique = {}
        self.columnIndex = {self.contents[i].name: i for i in range(len(self.contents))}

    def describe(self):
        desc = {col.name: col.getDESC() for col in self.contents}
        for name in self.primary:
            desc[name][3] = 'primary'
        for name in self.foreign:
            if desc[name][3] is not None:
                desc[name][3] = 'multi'
            else:
                desc[name][3] = 'foreign'
        for name in self.unique:
            if desc[name][3] is "":
                desc[name][3] = 'unique'
        return tuple(desc.values())

    def insertColumn(self, col: ColumnInfo):
        if col.name not in self.columnMap:
            self.contents.append(col)
            self.columnMap = {col.name: col for col in self.contents}
            self.columnType = [col.getSize() for col in self.contents]
            self.columnSize = [col.type for col in self.contents]
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
            self.columnType = [col.getSize() for col in self.contents]
            self.columnSize = [col.type for col in self.contents]
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

    def addUnique(self, column: str, uniq: str):
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
                    byte = (0, ) + tuple(value.encode())
                    if len(byte) > size:
                        print("OH NO")
                        raise VarcharTooLong("too long. max size is " + str(size - 1))
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
                val = val.replace("/", "-")
                vals = val.split("-")
                d = date(*map(int, vals))
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
                    val = data.tobytes()[1:0].rstrip(b'\x00').decode('utf-8')
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

from FileSystem.BufManager import BufManager

class MetaManager:
    def __init__(self, bm: BufManager, syspath: str):
        self.BM = bm
        self.systemPath = syspath
        self.handlers = {}

    def shutHandler(self, database: str):
        if self.handlers.get(database) is not None:
            self.handlers.pop(database).shutdown()
        return

    def shutdown(self):
        namelist = self.handlers.keys()
        for name in namelist:
            self.shutHandler(name)
        return
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

from .RecordManager import RecordManager
from .macro import *
from .rid import RID
from .record import Record

class FileHandler:

    def __init__(self, rm: RecordManager, fid: int, name: str):
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
        if bitmap[rid.slot] is False:
            bitmap[rid.slot] = True
        self.headChanged = True

        bitmap = np.packbits(bitmap)
        page[RECORD_PAGE_FIXED_HEADER: RECORD_PAGE_FIXED_HEADER + bitmapLen] = bitmap
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
            page = self.handler.RM.BM.getPage(pID)
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

        self.BM.FM.newPage(fileID, self.toSerial(self.getHead(recordLen)))
        self.BM.closeFile(fileID)
        return

    def getHead(self, recordLen: int):
        recordNum = self.getRecordNum(recordLen)
        bitmapLen = self.getBitmapLen(recordNum)
        return {'RecordLen': recordLen, 'RecordNum': recordNum, 'PageNum': 1,
                'AllRecord': 0, 'NextAvai': 0, 'BitmapLen': bitmapLen}

    def openFile(self, name: str):
        if name in self.opened:
            handler = self.opened[name]
            return handler
        fID = self.BM.openFile(name)
        self.opened[name] = FileHandler(self, fID, name)
        return self.opened[name]

    def destroyFile(self, name: str):
        self.BM.FM.destroyFile(name)
        self.opened.pop(name)
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
    def getBitmapLen(recordNum):
        length = (recordNum + 7) / 8
        return length


    def replaceFile(self, src: str, dst: str):
        if self.opened.get(src) is not None:
            self.closeFile(src)
        if self.opened.get(dst) is not None:
            self.closeFile(dst)
        self.destroyFile(dst)
        self.BM.FM.renameFile(src, dst)
        return

    def shutdown(self):
        for name in self.opened.keys():
            self.closeFile(name)

    @staticmethod
    def toSerial(d: dict):
        serial = dumps(d, ensure_ascii=False).encode('utf-8')
        empty = np.zeros(PAGE_SIZE, dtype=np.unit8)
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
