from SQL_Parser.SQLVisitor import SQLVisitor
from SQL_Parser.SQLParser import SQLParser
from SQL_Parser.SQLLexer import SQLLexer
import time
from system_manager import SystemManger
from lookup_element import Reducer, Term, LookupOutput, Join


class SystemVisitor(SQLVisitor):
    def __init__(self, system_manager=None):
        super(SQLVisitor, self).__init__()
        self.system_manager = system_manager
        self.time_begin = None

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

    def visit_program(self, ctx: SQLParser.ProgramContext):
        res = []
        for item in ctx.statement():
            try:
                output: LookupOutput = item.accept(self)
                if output is not None:
                    output._cost=self.spend_time()
                    output.simplify()
                    res.append(output)
                    