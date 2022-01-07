from MetaSystem.info import TableInfo, ColumnInfo
from SQL_Parser.SQLVisitor import SQLVisitor
from SQL_Parser.SQLParser import SQLParser
from antlr4 import ParserRuleContext
import time
from system_manager import SystemManger
from lookup_element import Reducer, Term, LookupOutput


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
        value = self.system_manager.resultToValue(result=result, is_in=False)
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
