from pathlib import Path
from argparse import ArgumentParser, Namespace
import os, stat, sys
from ManageSystem.printers import TablePrinter, CSVPrinter
from ManageSystem.system_visitor import SystemVisitor
from ManageSystem.system_manager import SystemManger
from FileSystem.FileManager import FileManager
from FileSystem.BufManager import BufManager
from RecordSystem.RecordManager import RecordManager
from IndexSystem.index_manager import IndexManager
from ManageSystem.Executor import Executor

def getParser():
    parser = ArgumentParser()
    choices = {'table': TablePrinter, 'csv': CSVPrinter}
    parser.add_argument('-p', '--printer', type=str, choices=choices, default='table')
    parser.add_argument('-b', '--base', type=Path, default='data')
    parser.add_argument('-f', '--file', type=Path)
    parser.add_argument('-t', '--table', type=str)
    parser.add_argument('database', nargs='?', type=str)
    return parser

if __name__ == '__main__':
    args = getParser().parse_args()
    printer = None
    if args.printer == 'table':
        printer = TablePrinter()
    elif args.printer == 'csv':
        printer = CSVPrinter()
    visitor = SystemVisitor()
    syspath = Path(args.base)
    FM = FileManager()
    BM = BufManager(FM)
    RM = RecordManager(BM)
    IM = IndexManager(BM, syspath)
    manager = SystemManger(visitor, syspath, BM, RM, IM)
    if args.database:
        manager.useDatabase(args.database)
    if args.file:
        executor = Executor()
        result = executor.execute(manager, args.file, args.database, args.table)
        printer.print(results=result)
    else:
        sql = ''
        mode = os.fstat(0).st_mode
        while True:
            if not stat.S_ISREG(mode):
                prefix = f'liubase({printer.inUse})'
                if sql:
                    pre = '-'.rjust(len(prefix))
                    print(pre + '> ', end='')
                else:
                    print(prefix + '> ', end='')
            try:
                sql += ' ' + input()
            except (KeyboardInterrupt, EOFError):
                break
            if sql.strip().lower() in ('exit', 'quit', '.exit', '.quit'):
                break
            if sql.endswith(';'):
                result = manager.execute(sql)

    manager.shutdown()
