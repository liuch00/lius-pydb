
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