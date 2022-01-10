
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
                return