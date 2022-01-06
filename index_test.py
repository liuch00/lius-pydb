from .RecordSystem.rid import RID
from .FileSystem import BufManager
from .FileSystem import FileManager
from .IndexSystem.index_handler import IndexHandler
from .IndexSystem.file_index import FileIndex


def insert_test(handler, page_id):
    print("Init Test...")
    indexer = FileIndex(handler, page_id)
    for i in range(4096):
        indexer.insert(i, RID(0, i))
    for i in range(4090, 4100):
        res = indexer.search(i)
        print("Search %d:" % i, None if res == None else res.slot_id)
    print("Range [100, 110]:")
    for i in indexer.range(100, 110):
        print(i)
    indexer.pour()
    print("Test End")


def load_test(handler, page_id):
    print("Load Test...")
    indexer = FileIndex(handler, page_id)
    indexer.take()
    for i in range(4090, 4100):
        res = indexer.search(i)
        print("Search %d:" % i, None if res == None else res.slot_id)
    print("Test End")


def test1():
    # now just do some test
    file_manager = FileManager()
    manager = BufManager(file_manager)
    handler = IndexHandler(buf_manager=manager)
    print("Test start.")
    page_id = handler.new_page()
    print(page_id)
    insert_test(handler, page_id)
    load_test(handler, page_id)

def test2():
    file_manager=FileManager()
    manager = BufManager(file_manager)
    handler = IndexHandler(buf_manager=manager)
    page_id = 0
    load_test(handler, page_id)


if __name__ == "__main__":
    test2()
