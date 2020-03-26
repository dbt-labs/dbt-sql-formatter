# encoding: utf-8
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this file,
# You can obtain one at http://mozilla.org/MPL/2.0/.
#
# Author: Beto Dealmeida (beto@dealmeida.net)
#

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import re
import pdb

from contextlib import contextmanager

VALID = re.compile(r'[a-zA-Z_]\w*')


class Token(object):
    def __init__(self, token):
        self.token = token

class Whitespace(object):
    def __init__(self, count, char):
        self.count = count
        self.char = char

class SQLDocument(object):
    def __init__(self):
        self.tokens = []
        self.indent = 0
        self.indent_char = ' '

        self._newline = False
        self.commas = 'back'

    def add(self, token):
        if hasattr(token, 'asList'):
            token = token.asList()
        if type(token) not in (list, tuple):
            token = [token]
        if self._newline:
            self.tokens.append(self.indent_char * self.indent)
            self._newline = False
        self.tokens.extend(list(token))

    def newline(self):
        self.tokens.append('\n')
        self._newline = True

    @contextmanager
    def indented(self, count=4):
        self.indent += count
        yield self
        self.indent -= count

    def pprint(self):
        print("".join(self.tokens))

def Operator(op, parentheses=False):
    op = ' {0} '.format(op)
    def func(self, json):
        out = op.join(self.dispatch(v) for v in json)
        if parentheses:
            out = '({0})'.format(out)
        return out
    return func


class Formatter:

    clauses = [
        'ctes',
        'columns',
        'from_',
        'where',
        'group_by_terms',
        'having_expr',
        'order_by_terms',
        'limit',
        # 'offset', # not supported TODO
    ]

    # simple operators
    _concat = Operator('||')
    _mul = Operator('*')
    _div = Operator('/', parentheses=True)
    _add = Operator('+')
    _sub = Operator('-', parentheses=True)
    _neq = Operator('<>')
    _gt = Operator('>')
    _lt = Operator('<')
    _gte = Operator('>=')
    _lte = Operator('<=')
    _eq = Operator('=')
    _or = Operator('OR')
    _and = Operator('AND')

    def __init__(self):
        self.document = SQLDocument()

    def format(self, json):
        if 'union' in json:
            return self.union(json['union'])
        else:
            return self.query(json)

    def dispatch(self, json):
        if isinstance(json, list):
            return self.delimited_list(json)
        if isinstance(json, dict):
            if len(json) == 0:
                return ''
            elif 'value' in json:
                return self.value(json)
            elif 'from' in json:
                # Nested queries
                return '({})'.format(self.format(json))
            elif 'select' in json:
                # Nested queries
                return '({})'.format(self.format(json))
            else:
                return self.op(json)
        if isinstance(json, string_types):
            return json

        return text(json)

    def delimited_list(self, json):
        return ', '.join(self.dispatch(element) for element in json)

    def value(self, json):
        parts = [self.dispatch(json['value'])]
        if 'name' in json:
            parts.extend(['AS', self.dispatch(json['name'])])
        return ' '.join(parts)

    def op(self, json):
        if 'on' in json:
            return self._on(json)

        if len(json) > 1:
            raise Exception('Operators should have only one key!')
        key, value = list(json.items())[0]

        # check if the attribute exists, and call the corresponding method;
        # note that we disallow keys that start with `_` to avoid giving access
        # to magic methods
        attr = '_{0}'.format(key)
        if hasattr(self, attr) and not key.startswith('_'):
            method = getattr(self, attr)
            return method(value)

        # treat as regular function call
        if isinstance(value, dict) and len(value) == 0:
            return key.upper() + "()"  # NOT SURE IF AN EMPTY dict SHOULD BE DELT WITH HERE, OR IN self.dispatch()
        else:
            return '{0}({1})'.format(key.upper(), self.dispatch(value))

    def _exists(self, value):
        return '{0} IS NOT NULL'.format(self.dispatch(value))

    def _missing(self, value):
        return '{0} IS NULL'.format(self.dispatch(value))

    def _like(self, pair):
        return '{0} LIKE {1}'.format(self.dispatch(pair[0]), self.dispatch(pair[1]))

    def _nlike(self, pair):
        return '{0} NOT LIKE {1}'.format(self.dispatch(pair[0]), self.dispatch(pair[1]))

    def _is(self, pair):
        return '{0} IS {1}'.format(self.dispatch(pair[0]), self.dispatch(pair[1]))

    def _in(self, json):
        valid = self.dispatch(json[1])
        # `(10, 11, 12)` does not get parsed as literal, so it's formatted as
        # `10, 11, 12`. This fixes it.
        if not valid.startswith('('):
            valid = '({0})'.format(valid)

        return '{0} IN {1}'.format(json[0], valid)

    def _nin(self, json):
        valid = self.dispatch(json[1])
        # `(10, 11, 12)` does not get parsed as literal, so it's formatted as
        # `10, 11, 12`. This fixes it.
        if not valid.startswith('('):
            valid = '({0})'.format(valid)

        return '{0} NOT IN {1}'.format(json[0], valid)

    def _case(self, checks):
        parts = ['CASE']
        for check in checks:
            if isinstance(check, dict):
                parts.extend(['WHEN', self.dispatch(check['when'])])
                parts.extend(['THEN', self.dispatch(check['then'])])
            else:
                parts.extend(['ELSE', self.dispatch(check)])
        parts.append('END')
        return ' '.join(parts)

    def _literal(self, json):
        if isinstance(json, list):
            return '({0})'.format(', '.join(self._literal(v) for v in json))
        elif isinstance(json, string_types):
            return "'{0}'".format(json.replace("'", "''"))
        else:
            return str(json)

    def _on(self, json):
        detected_join = join_keywords & set(json.keys())
        if len(detected_join) == 0:
            raise Exception(
                'Fail to detect join type! Detected: "{}" Except one of: "{}"'.format(
                    [on_keyword for on_keyword in json if on_keyword != 'on'][0],
                    '", "'.join(join_keywords)
                )
            )

        join_keyword = detected_join.pop()

        return '{0} {1} ON {2}'.format(
            join_keyword.upper(), self.dispatch(json[join_keyword]), self.dispatch(json['on'])
        )

    def union(self, json):
        return ' UNION '.join(self.query(query) for query in json)

    def query(self, json):
        for clause in self.clauses:
            func = getattr(self, clause, None)
            if func:
                func(json)

    def add_expr_list(self, expr):
        if isinstance(expr, (str, int, float)):
            self.add_expr(expr)
        else:
            for i, field in enumerate(expr):
                self.add_expr(field)
                if i != len(expr) - 1:
                    self.document.add(', ')

    def add_expr(self, expr):
        doc = self.document

        if hasattr(expr, 'getName') and expr.getName() == 'operator':
            if expr.assoc == 'unary':
                doc.add(str(expr.op.match))
                if len(expr.op.match) > 1:
                    doc.add(' ')
                self.add_expr(expr.tokens)
            elif expr.assoc == 'binary':
                self.add_expr(expr.tokens[0])
                doc.add(' ')
                doc.add(str(expr.op.match))
                doc.add(' ')
                self.add_expr(expr.tokens[1])
            elif expr.assoc == 'ternary':
                import ipdb; ipdb.set_trace()
            else:
                import ipdb; ipdb.set_trace()
        elif hasattr(expr, 'getName') and expr.getName() == 'function':
            doc.add(str(expr.func))
            doc.add('(')
            self.add_expr_list(expr.tokens[0])
            doc.add(')')
        elif hasattr(expr, 'getName') and expr.getName() == 'window function':
            doc.add(str(expr.func))
            doc.add('(')
            self.add_expr_list(expr.func_args)
            doc.add(')')
            doc.add(' ')
            doc.add('over')
            doc.add(' ')
            doc.add('(')
            if expr.partition_args:
                # TODO: Method for printing list
                doc.add('partition by ')
                self.add_expr_list(expr.partition_args)
            if expr.order_args:
                if expr.partition_args:
                    doc.add(' ')
                doc.add('order by ')
                self.add_expr_list(expr.order_args)
            for arg in expr.window_args:
                doc.add(' ')
                doc.add(arg)
            doc.add(')')
        elif hasattr(expr, 'getName') and expr.getName() == 'case':
            doc.add('case')
            doc.newline()
            with doc.indented():
                for when in expr.whens:
                    doc.add('when')
                    doc.add(' ')
                    self.add_expr(when['when'])
                    doc.add(' ')
                    doc.add('then')
                    doc.add(' ')
                    self.add_expr(when['then'])
                    doc.newline()
                if expr._else:
                    doc.add('else')
                    doc.add(' ')
                    self.add_expr(expr._else)
                    doc.newline()
            doc.add('end')
        elif hasattr(expr, 'getName') and expr.getName() == 'select':
            import ipdb; ipdb.set_trace()
        #elif hasattr(expr, 'getName') and expr.getName() != 'quoted_expr':
        #    import ipdb; ipdb.set_trace()

        elif isinstance(expr, (str, int)):
            doc.add(expr)
        else:
            for el in expr:
                self.add_expr(el)

    def add_column(self, column):
        if type(column) == str:
            self.document.add(column)
        elif column.select:
            self.document.add('(')
            self.document.newline()
            with self.document.indented():
                self.query(column)
            self.document.newline()
            self.document.add(')')

        else:
            self.add_expr(column.quoted_expr)

        if type(column) != str and column.alias:
            self.document.add(' ')
            self.document.add('as')
            self.document.add(' ')
            self.document.add(column.alias)

    def add_cte(self, cte):
        self.document.add(cte.cte_name)
        self.document.add(' ')
        self.document.add('as')
        self.document.add(' ')
        self.document.add('(')
        self.document.newline()
        self.document.newline()
        with self.document.indented():
            self.query(cte)
        self.document.newline()
        self.document.newline()
        self.document.add(')')

    def ctes(self, json):
        if len(json.ctes) == 0:
            return

        self.document.add('with ')
        for i, cte in enumerate(json.ctes):
            self.add_cte(cte)
            if i != len(json.ctes) - 1:
                self.document.add(',')
            self.document.newline()
            self.document.newline()


    def columns(self, json):
        self.document.add('select')

        if len(json.columns) == 1 and json.columns[0] == '*':
            self.document.add(' * ')
        else:
            self.document.newline()
            with self.document.indented():
                for i, column in enumerate(json.columns):
                    if self.document.commas == 'front' and i != 0:
                        self.document.add(', ')

                    self.add_column(column)

                    if self.document.commas == 'back' and i != len(json.columns)-1:
                        self.document.add(',')
                    self.document.newline()

    # This ain't it :/
    def add_from(self, from_):
        self.document.newline()
        self.document.add('from')
        self.document.add(' ')
        if type(from_) == str:
            self.document.add(from_)
        else:
            self.document.add(from_.table.asList()) # TODO: SBQ

            for join in from_.joins:
                if join.join_op[0] == ',':
                    self.document.add(',')
                    self.document.newline()
                    self.document.add(join.table)
                else:
                    self.document.newline()
                    self.add_expr(join.asList())

    def from_(self, json):
        if 'from' not in json:
            return
        from_ = json['from']
        if 'union' in from_:
            return self.union(from_['union'])

        self.add_from(from_[0])

    def where(self, json):
        if 'where' not in json:
            return

        self.document.newline()
        self.document.add('where ')
        self.add_expr(json['where'])

    def group_by_terms(self, json):
        if 'group_by_terms' not in json:
            return

    def having(self, json):
        import ipdb; ipdb.set_trace()
        if 'having' in json:
            return 'HAVING {0}'.format(self.dispatch(json['having']))

    def orderby(self, json):
        import ipdb; ipdb.set_trace()
        if 'orderby' in json:
            orderby = json['orderby']
            if isinstance(orderby, dict):
                orderby = [orderby]
            return 'ORDER BY {0}'.format(','.join([
                '{0} {1}'.format(self.dispatch(o), o.get('sort', '').upper()).strip()
                for o in orderby
            ]))

    def limit(self, json):
        if 'limit' in json:
            if json['limit']:
                return 'LIMIT {0}'.format(self.dispatch(json['limit']))

    def offset(self, json):
        if 'offset' in json:
            return 'OFFSET {0}'.format(self.dispatch(json['offset']))

