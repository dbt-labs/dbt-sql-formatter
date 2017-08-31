
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Andi Albrecht, albrecht.andi@gmail.com
#
# This module is part of python-sqlparse and is released under
# the BSD License: https://opensource.org/licenses/BSD-3-Clause

from sqlparse import sql, tokens as T
from sqlparse.compat import text_type
from sqlparse.utils import offset, indent
from sqlparse import engine, filters

import sql_options


class DbtReindentFilter(object):
    def __init__(self, width=2, char=' ', wrap_after=0, n='\n',
                 comma_first=False):
        self.n = n
        self.width = width
        self.char = char
        self.indent = 0
        self.offset = 0
        self.wrap_after = wrap_after
        self.comma_first = comma_first
        self._curr_stmt = None
        self._last_stmt = None

    def _flatten_up_to_token(self, token):
        """Yields all tokens up to token but excluding current."""
        if token.is_group:
            token = next(token.flatten())

        for t in self._curr_stmt.flatten():
            if t == token:
                break
            yield t

    @property
    def leading_ws(self):
        return self.offset + self.indent * self.width

    def _get_offset(self, token):
        raw = u''.join(map(text_type, self._flatten_up_to_token(token)))
        line = (raw or '\n').splitlines()[-1]
        # Now take current offset into account and return relative offset.
        res = len(line) - len(self.char * self.leading_ws)
        return res

    def nl(self, offset=0, blank=False):
        if blank:
            line = ''
        else:
            line = self.char * max(0, self.leading_ws + offset)

        return sql.Token(T.Whitespace, self.n + line)

    def _next_token(self, tlist, idx=-1):
        split_words = ('FROM', 'STRAIGHT_JOIN$', 'JOIN$', 'AND', 'OR',
                       'GROUP', 'ORDER', 'UNION', 'VALUES', 'ROWS', # DB: Added 'rows', 'with'
                       'SET', 'BETWEEN', 'EXCEPT', 'HAVING', 'LIMIT')
        m_split = T.Keyword, split_words, True
        tidx, token = tlist.token_next_by(m=m_split, idx=idx)

        if token and token.normalized == 'BETWEEN':
            tidx, token = self._next_token(tlist, tidx)

            if token and token.normalized == 'AND':
                tidx, token = self._next_token(tlist, tidx)

        return tidx, token

    def _split_kwds(self, tlist):
        tidx, token = self._next_token(tlist)
        while token:
            pidx, prev_ = tlist.token_prev(tidx, skip_ws=False)
            uprev = text_type(prev_)

            if prev_ and prev_.is_whitespace:
                del tlist.tokens[pidx]
                tidx -= 1

            if not (uprev.endswith('\n') or uprev.endswith('\r')):
                #tlist.insert_before(tidx, self.nl(offset=-7))

                # db (for window funcs):
                if tlist.within(sql.Function):
                    tlist.insert_before(tidx, self.nl(offset=self.width * 2))
                else:
                    tlist.insert_before(tidx, self.nl(offset=0))
                tidx += 1

            tidx, token = self._next_token(tlist, tidx)

    def _split_statements(self, tlist):
        ttypes = T.Keyword.DML, T.Keyword.DDL
        tidx, token = tlist.token_next_by(t=ttypes)
        while token:
            pidx, prev_ = tlist.token_prev(tidx, skip_ws=False)
            if prev_ and prev_.is_whitespace:
                del tlist.tokens[pidx]
                tidx -= 1
            # only break if it's not the first token
            if prev_:
                tlist.insert_before(tidx, self.nl())
                tlist.insert_before(tidx, self.nl())
                tidx += 2
            tidx, token = tlist.token_next_by(t=ttypes, idx=tidx)

    def _process(self, tlist):
        func_name = '_process_{cls}'.format(cls=type(tlist).__name__)
        #print("PROCESSING:" , type(tlist).__name__, tlist)
        func = getattr(self, func_name.lower(), self._process_default)
        func(tlist)

    def _process_where(self, tlist):
        tidx, token = tlist.token_next_by(m=(T.Keyword, 'WHERE'))
        # issue121, errors in statement fixed??
        tlist.insert_before(tidx, self.nl())

        with indent(self):
            self._process_default(tlist)

    def _process_parenthesis(self, tlist):
        ttypes = T.Keyword.DML, T.Keyword.DDL
        _, is_dml_dll = tlist.token_next_by(t=ttypes)
        fidx, first = tlist.token_next_by(m=sql.Parenthesis.M_OPEN)
        
        with indent(self, 1 if is_dml_dll else 0):
            if is_dml_dll:
                tlist.tokens.insert(fidx + 1, self.nl(blank=True))
                tlist.tokens.insert(fidx + 2, self.nl())

            self._process_default(tlist, not is_dml_dll)
            #with offset(self, self.width):
            #    self._process_default(tlist, not is_dml_dll)

        if is_dml_dll:
            last_index = len(tlist.tokens)
            tlist.tokens.insert(last_index - 1, self.nl(blank=True))
            tlist.tokens.insert(last_index , self.nl())
                
    def _process_identifierlist(self, tlist):
        identifiers = list(tlist.get_identifiers())
        
        #first = next(identifiers.pop(0).flatten())
        #num_offset = 1 if self.char == '\t' else self._get_offset(first)

        num_offset = self.width
        
        if not tlist.within(sql.Function):
            with offset(self, num_offset):
                position = 0
                for i, token in enumerate(identifiers):
                    # Add 1 for the "," separator
                    #position += len(token.value) + 1
                    
                    idx = tlist.token_index(token)
                    prev_idx, prev_tok = tlist.token_prev(idx, skip_ws=True, skip_cm=True)
                    if position > (self.wrap_after - self.offset):
                        adjust = 0
                        #tlist.insert_before(token, self.nl(offset=-3))
                        
                        #is_cte = 'Parenthesis' in [type(t).__name__ for t in token.get_sublists()]
                        #print(token, token.is_group)
                        #print(token.get_sublists())
                        #print([t for t in token.get_sublists()])
                        
                        # dumb hack:
                        if hasattr(token, 'tokens') and len(token.tokens) > 2:
                            is_cte = (token.tokens[2].value == 'as')
                        else:
                            is_cte = False
                        
                        if token.is_group and i == 0 and is_cte:
                            pass # first cte
                        elif token.is_group and i > 0 and is_cte:
                            #print("CTE", type(token).__name__)
                            tlist.insert_before(token, self.nl(blank=True))
                            tlist.insert_before(token, self.nl(blank=True))  
                            position = 0
                        else:
                            tlist.insert_before(token, self.nl(offset=adjust))
                            position = 0

        self._process_default(tlist)


    def _process_case(self, tlist):
        iterable = iter(tlist.get_cases())
        cond, _ = next(iterable)
        first = next(cond[0].flatten())

        with offset(self, self._get_offset(tlist[0])):
            with offset(self, self._get_offset(first)):
                for cond, value in iterable:
                    token = value[0] if cond is None else cond[0]
                    tlist.insert_before(token, self.nl())

                # Line breaks on group level are done. let's add an offset of
                # len "when ", "then ", "else "
                with offset(self, len("WHEN ")):
                    self._process_default(tlist)
            end_idx, end = tlist.token_next_by(m=sql.Case.M_CLOSE)
            if end_idx is not None:
                tlist.insert_before(end_idx, self.nl())

    def _process_default(self, tlist, stmts=True):
        self._split_statements(tlist) if stmts else None
        self._split_kwds(tlist)
        for sgroup in tlist.get_sublists():
            self._process(sgroup)

    def process(self, stmt):
        self._curr_stmt = stmt
        self._process(stmt)

        if self._last_stmt is not None:
            nl = '\n' if text_type(self._last_stmt).endswith('\n') else '\n\n'
            stmt.tokens.insert(0, sql.Token(T.Whitespace, nl))

        self._last_stmt = stmt
        return stmt

def build_filter_stack(stack, options):
    """Setup and return a filter stack.
    Args:
      stack: :class:`~sqlparse.filters.FilterStack` instance
      options: Dictionary with options validated by validate_options.
    """
    # Token filter
    if options.get('strip_comments'):
        stack.enable_grouping()
        stack.stmtprocess.append(filters.StripCommentsFilter())

    if options.get('strip_whitespace') or options.get('reindent'):
        stack.enable_grouping()
        stack.stmtprocess.append(filters.StripWhitespaceFilter())

    if options.get('reindent'):
        stack.enable_grouping()
        stack.stmtprocess.append(
            DbtReindentFilter(char=options['indent_char'],
                                   width=options['indent_width'],
                                   wrap_after=options['wrap_after'],
                                   comma_first=options['comma_first']))
    return stack

def sql_format(sql, options={}, encoding=None):
    stack = engine.FilterStack()
    options = sql_options.validate_options(options)
    stack = build_filter_stack(stack, options)
    stack.postprocess.append(filters.SerializerUnicode())
    return u''.join(stack.run(sql, encoding))
