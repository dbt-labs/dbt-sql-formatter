# bigquery_view_parser.py
#
# A parser to extract table names from BigQuery view definitions.
# This is based on the `select_parser.py` sample in pyparsing:
# https://github.com/pyparsing/pyparsing/blob/master/examples/select_parser.py
#
# Michael Smedberg
#

from pyparsing import ParserElement, Suppress, Forward, CaselessKeyword
from pyparsing import MatchFirst, alphas, alphanums, Combine, Word, Literal, White, Empty
from pyparsing import QuotedString, CharsNotIn, Optional, Group, ZeroOrMore, NoMatch
from pyparsing import oneOf, delimitedList, restOfLine, cStyleComment
from pyparsing import infixNotation, opAssoc, OneOrMore, Regex, nums

def debug(s, i, toks):
    if len(toks) > 0:
        #import ipdb; ipdb.set_trace()
        pass


class SemanticToken(object):
    def __iter__(self):
        return (i for i in self.tokens)

    def __len__(self):
        return len(self.tokens)




class BigQueryViewParser:
    """Parser to extract table info from BigQuery view definitions"""

    _parser = None
    _table_identifiers = set()
    _with_aliases = set()

    def get_table_names(self, sql_stmt):
        table_identifiers, with_aliases = self._parse(sql_stmt)

        # Table names and alias names might differ by case, but that's not
        # relevant- aliases are not case sensitive
        lower_aliases = BigQueryViewParser.lowercase_set_of_tuples(with_aliases)
        tables = {
            x
            for x in table_identifiers
            if not BigQueryViewParser.lowercase_of_tuple(x) in lower_aliases
        }

        # Table names ARE case sensitive as described at
        # https://cloud.google.com/bigquery/docs/reference/standard-sql/lexical#case_sensitivity
        # return tables
        return table_identifiers, with_aliases

    def _parse(self, sql_stmt):
        BigQueryViewParser._table_identifiers.clear()
        BigQueryViewParser._with_aliases.clear()
        res = BigQueryViewParser._get_parser().parseString(sql_stmt, parseAll=True)

        def kk(v):
            print(list(v.keys()))

        return res

    @classmethod
    def lowercase_of_tuple(cls, tuple_to_lowercase):
        return tuple(x.lower() if x else None for x in tuple_to_lowercase)

    @classmethod
    def lowercase_set_of_tuples(cls, set_of_tuples):
        return {BigQueryViewParser.lowercase_of_tuple(x) for x in set_of_tuples}

    @classmethod
    def _get_parser(cls):
        if cls._parser is not None:
            return cls._parser

        ParserElement.enablePackrat()

        LPAR, RPAR, COMMA, LBRACKET, RBRACKET, LT, GT = map(Literal, "(),[]<>")
        ungrouped_select_stmt = Forward().setName("select statement")

        # keywords
        (
            UNION,
            ALL,
            AND,
            INTERSECT,
            EXCEPT,
            COLLATE,
            ASC,
            DESC,
            ON,
            USING,
            NATURAL,
            INNER,
            CROSS,
            LEFT,
            RIGHT,
            OUTER,
            FULL,
            JOIN,
            AS,
            INDEXED,
            NOT,
            SELECT,
            DISTINCT,
            FROM,
            WHERE,
            GROUP,
            BY,
            HAVING,
            ORDER,
            BY,
            LIMIT,
            OFFSET,
            OR,
            CAST,
            ISNULL,
            NOTNULL,
            NULL,
            IS,
            BETWEEN,
            ELSE,
            END,
            CASE,
            WHEN,
            THEN,
            EXISTS,
            COLLATE,
            IN,
            LIKE,
            GLOB,
            REGEXP,
            MATCH,
            ESCAPE,
            CURRENT_TIME,
            CURRENT_DATE,
            CURRENT_TIMESTAMP,
            WITH,
            EXTRACT,
            PARTITION,
            ROWS,
            RANGE,
            UNBOUNDED,
            PRECEDING,
            CURRENT,
            ROW,
            FOLLOWING,
            OVER,
            INTERVAL,
            DATE_ADD,
            DATE_SUB,
            ADDDATE,
            SUBDATE,
            REGEXP_EXTRACT,
            SPLIT,
            ORDINAL,
            FIRST_VALUE,
            LAST_VALUE,
            NTH_VALUE,
            LEAD,
            LAG,
            PERCENTILE_CONT,
            PRECENTILE_DISC,
            RANK,
            DENSE_RANK,
            PERCENT_RANK,
            CUME_DIST,
            NTILE,
            ROW_NUMBER,
            DATE,
            TIME,
            DATETIME,
            TIMESTAMP,
            UNNEST,
            INT64,
            NUMERIC,
            FLOAT64,
            BOOL,
            BYTES,
            GEOGRAPHY,
            ARRAY,
            STRUCT,
            SAFE_CAST,
            ANY_VALUE,
            ARRAY_AGG,
            ARRAY_CONCAT_AGG,
            AVG,
            BIT_AND,
            BIT_OR,
            BIT_XOR,
            COUNT,
            COUNTIF,
            LOGICAL_AND,
            LOGICAL_OR,
            MAX,
            MIN,
            STRING_AGG,
            SUM,
            CORR,
            COVAR_POP,
            COVAR_SAMP,
            STDDEV_POP,
            STDDEV_SAMP,
            STDDEV,
            VAR_POP,
            VAR_SAMP,
            VARIANCE,
            TIMESTAMP_ADD,
            TIMESTAMP_SUB,
            GENERATE_ARRAY,
            GENERATE_DATE_ARRAY,
            GENERATE_TIMESTAMP_ARRAY,
            FOR,
            SYSTEMTIME,
            AS,
            OF,
            WINDOW,
            RESPECT,
            IGNORE,
            NULLS,
        ) = map(
            CaselessKeyword,
            """
            UNION, ALL, AND, INTERSECT, EXCEPT, COLLATE, ASC, DESC, ON, USING,
            NATURAL, INNER, CROSS, LEFT, RIGHT, OUTER, FULL, JOIN, AS, INDEXED,
            NOT, SELECT, DISTINCT, FROM, WHERE, GROUP, BY, HAVING, ORDER, BY,
            LIMIT, OFFSET, OR, CAST, ISNULL, NOTNULL, NULL, IS, BETWEEN, ELSE,
            END, CASE, WHEN, THEN, EXISTS, COLLATE, IN, LIKE, GLOB, REGEXP,
            MATCH, ESCAPE, CURRENT_TIME, CURRENT_DATE, CURRENT_TIMESTAMP, WITH,
            EXTRACT, PARTITION, ROWS, RANGE, UNBOUNDED, PRECEDING, CURRENT,
            ROW, FOLLOWING, OVER, INTERVAL, DATE_ADD, DATE_SUB, ADDDATE,
            SUBDATE, REGEXP_EXTRACT, SPLIT, ORDINAL, FIRST_VALUE, LAST_VALUE,
            NTH_VALUE, LEAD, LAG, PERCENTILE_CONT, PRECENTILE_DISC, RANK,
            DENSE_RANK, PERCENT_RANK, CUME_DIST, NTILE, ROW_NUMBER, DATE, TIME,
            DATETIME, TIMESTAMP, UNNEST, INT64, NUMERIC, FLOAT64, BOOL, BYTES,
            GEOGRAPHY, ARRAY, STRUCT, SAFE_CAST, ANY_VALUE, ARRAY_AGG,
            ARRAY_CONCAT_AGG, AVG, BIT_AND, BIT_OR, BIT_XOR, COUNT, COUNTIF,
            LOGICAL_AND, LOGICAL_OR, MAX, MIN, STRING_AGG, SUM, CORR,
            COVAR_POP, COVAR_SAMP, STDDEV_POP, STDDEV_SAMP, STDDEV, VAR_POP,
            VAR_SAMP, VARIANCE, TIMESTAMP_ADD, TIMESTAMP_SUB, GENERATE_ARRAY,
            GENERATE_DATE_ARRAY, GENERATE_TIMESTAMP_ARRAY, FOR, SYSTEMTIME, AS,
            OF, WINDOW, RESPECT, IGNORE, NULLS
                 """.replace(
                ",", ""
            ).split(),
        )

        keyword_nonfunctions = MatchFirst(
            (
                UNION,
                ALL,
                INTERSECT,
                EXCEPT,
                COLLATE,
                ASC,
                DESC,
                ON,
                USING,
                NATURAL,
                INNER,
                CROSS,
                LEFT,
                RIGHT,
                OUTER,
                FULL,
                JOIN,
                AS,
                INDEXED,
                NOT,
                SELECT,
                DISTINCT,
                FROM,
                WHERE,
                GROUP,
                BY,
                HAVING,
                ORDER,
                BY,
                LIMIT,
                OFFSET,
                CAST,
                ISNULL,
                NOTNULL,
                NULL,
                IS,
                BETWEEN,
                ELSE,
                END,
                CASE,
                WHEN,
                THEN,
                EXISTS,
                COLLATE,
                IN,
                LIKE,
                GLOB,
                REGEXP,
                MATCH,
                STRUCT,
                WINDOW,
            )
        )

        keyword = keyword_nonfunctions | MatchFirst(
            (
                ESCAPE,
                CURRENT_TIME,
                CURRENT_DATE,
                CURRENT_TIMESTAMP,
                DATE_ADD,
                DATE_SUB,
                ADDDATE,
                SUBDATE,
                INTERVAL,
                STRING_AGG,
                REGEXP_EXTRACT,
                SPLIT,
                ORDINAL,
                UNNEST,
                SAFE_CAST,
                PARTITION,
                TIMESTAMP_ADD,
                TIMESTAMP_SUB,
                ARRAY,
                GENERATE_ARRAY,
                GENERATE_DATE_ARRAY,
                GENERATE_TIMESTAMP_ARRAY,
            )
        )

        identifier_word = Word(alphas + "_@#", alphanums + "@$#_")
        identifier = ~keyword + identifier_word.copy()
        collation_name = identifier.copy()
        # NOTE: Column names can be keywords.  Doc says they cannot, but in practice it seems to work.
        column_name = identifier.copy()
        cast_to = identifier.copy()
        qualified_column_name = Group(
            delimitedList(column_name, delim=".")
            + Optional(
                Suppress("::") 
                + delimitedList(cast_to("cast"), delim="::")
            )
        )
        # NOTE: As with column names, column aliases can be keywords, e.g. functions like `current_time`.  Other
        # keywords, e.g. `from` make parsing pretty difficult (e.g. "SELECT a from from b" is confusing.)
        column_alias = ~keyword_nonfunctions + column_name.copy()
        table_name = identifier.copy()
        table_alias = identifier.copy()
        index_name = identifier.copy()
        function_name = identifier.copy()
        parameter_name = identifier.copy()
        # NOTE: The expression in a CASE statement can be an integer.  E.g. this is valid SQL:
        # select CASE 1 WHEN 1 THEN -1 ELSE -2 END from test_table
        unquoted_case_identifier = ~keyword + Word(alphanums + "$_")
        quoted_case_identifier = ~keyword + (
            QuotedString('"') ^ Suppress("`") + CharsNotIn("`") + Suppress("`")
        )
        case_identifier = quoted_case_identifier | unquoted_case_identifier
        case_expr = (
            Optional(case_identifier + Suppress("."))
            + Optional(case_identifier + Suppress("."))
            + case_identifier
        )

        # expression
        expr = Forward().setName("expression")

        integer = Regex(r"[+-]?\d+")
        numeric_literal = Regex(r"[+-]?\d*\.?\d+([eE][+-]?\d+)?")
        string_literal = QuotedString("'") | QuotedString('"') | QuotedString("`")
        regex_literal = "r" + string_literal
        blob_literal = Regex(r"[xX]'[0-9A-Fa-f]+'")
        date_or_time_literal = (DATE | TIME | DATETIME | TIMESTAMP) + string_literal
        literal_value = (
            numeric_literal
            | string_literal
            | regex_literal
            | blob_literal
            | date_or_time_literal
            | NULL
            | CURRENT_TIME + Optional(LPAR + Optional(string_literal) + RPAR)
            | CURRENT_DATE + Optional(LPAR + Optional(string_literal) + RPAR)
            | CURRENT_TIMESTAMP + Optional(LPAR + Optional(string_literal) + RPAR)
        )
        bind_parameter = Word("?", nums) | Combine(oneOf(": @ $") + parameter_name)
        type_name = oneOf(
            """TEXT REAL INTEGER BLOB NULL TIMESTAMP STRING DATE
            INT64 NUMERIC FLOAT64 BOOL BYTES DATETIME GEOGRAPHY TIME ARRAY
            STRUCT""",
            caseless=True,
        )
        date_part = oneOf(
            """DAY DAY_HOUR DAY_MICROSECOND DAY_MINUTE DAY_SECOND
            HOUR HOUR_MICROSECOND HOUR_MINUTE HOUR_SECOND MICROSECOND MINUTE
            MINUTE_MICROSECOND MINUTE_SECOND MONTH QUARTER SECOND
            SECOND_MICROSECOND WEEK YEAR YEAR_MONTH""",
            caseless=True,
        )
        datetime_operators = (
            DATE_ADD | DATE_SUB | ADDDATE | SUBDATE | TIMESTAMP_ADD | TIMESTAMP_SUB
        )

        def invalid_date_add(s, loc, tokens):
            prev_newline = s[:loc].rfind('\n')
            prev_prev_newline = s[:prev_newline].rfind('\n')
            if '--ignore' in s[prev_prev_newline:prev_newline]:
                pass
            else:
                raise RuntimeError("{} is not valid, did you mean 'date_add'".format(tokens[0]))

        #bad_datetime_operators = (
        #    CaselessKeyword('dateadd').setParseAction(invalid_date_add)
        #)

        grouping_term = expr.copy()
        ordering_term = Group(
            expr("order_key")
            + Optional(COLLATE + collation_name("collate"))
            + Optional(ASC | DESC)("direction")
        )("ordering_term")

        function_arg = expr.copy()("function_arg")
        function_args = Optional(
            "*"
            | Optional(DISTINCT)
            + delimitedList(function_arg)
            + Optional((RESPECT | IGNORE) + NULLS)
        )("function_args")
        function_call = (
            (function_name | keyword)("function_name")
            + LPAR
            + Group(function_args)("function_args_group")
            + RPAR
        )('function')

        navigation_function_name = (
            FIRST_VALUE
            | LAST_VALUE
            | NTH_VALUE
            | LEAD
            | LAG
            | PERCENTILE_CONT
            | PRECENTILE_DISC
        )
        aggregate_function_name = (
            ANY_VALUE
            | ARRAY_AGG
            | ARRAY_CONCAT_AGG
            | AVG
            | BIT_AND
            | BIT_OR
            | BIT_XOR
            | COUNT
            | COUNTIF
            | LOGICAL_AND
            | LOGICAL_OR
            | MAX
            | MIN
            | STRING_AGG
            | SUM
        )
        statistical_aggregate_function_name = (
            CORR
            | COVAR_POP
            | COVAR_SAMP
            | STDDEV_POP
            | STDDEV_SAMP
            | STDDEV
            | VAR_POP
            | VAR_SAMP
            | VARIANCE
        )
        numbering_function_name = (
            RANK | DENSE_RANK | PERCENT_RANK | CUME_DIST | NTILE | ROW_NUMBER
        )
        analytic_function_name = (
            navigation_function_name
            | aggregate_function_name
            | statistical_aggregate_function_name
            | numbering_function_name
        )("analytic_function_name")
        partition_expression_list = delimitedList(grouping_term)(
            "partition_expression_list"
        )
        window_frame_boundary_start = (
            UNBOUNDED + PRECEDING
            | numeric_literal + (PRECEDING | FOLLOWING)
            | CURRENT + ROW
        )
        window_frame_boundary_end = (
            UNBOUNDED + FOLLOWING
            | numeric_literal + (PRECEDING | FOLLOWING)
            | CURRENT + ROW
        )
        window_frame_clause = (ROWS | RANGE) + (
            ((UNBOUNDED + PRECEDING) | (numeric_literal + PRECEDING) | (CURRENT + ROW))
            | (BETWEEN + window_frame_boundary_start + AND + window_frame_boundary_end)
        )
        window_name = identifier.copy()("window_name")
        window_specification = (
            Optional(window_name)
            + Optional(PARTITION + BY + partition_expression_list)
            + Optional(ORDER + BY + delimitedList(ordering_term))
            + Optional(window_frame_clause)("window_specification")
        )
        analytic_function = (
            analytic_function_name
            + LPAR
            + function_args.setParseAction(debug)
            + RPAR
            + OVER
            + (window_name | LPAR + Optional(window_specification)('window') + RPAR)
        )("analytic_function")

        string_agg_term = (
            STRING_AGG
            + LPAR
            + Optional(DISTINCT)('has_distinct')
            + expr('string_agg_expr')
            + Optional(COMMA + string_literal('delimiter'))
            + Optional(
                ORDER + BY + expr + Optional(ASC | DESC) + Optional(LIMIT + integer)
            )
            + RPAR
        )("string_agg")
        array_literal = (
            Optional(ARRAY + Optional(LT + delimitedList(type_name) + GT))
            + LBRACKET
            + delimitedList(expr)
            + RBRACKET
        )
        interval = INTERVAL + expr + date_part
        array_generator = (
            GENERATE_ARRAY
            + LPAR
            + numeric_literal
            + COMMA
            + numeric_literal
            + COMMA
            + numeric_literal
            + RPAR
        )
        date_array_generator = (
            (GENERATE_DATE_ARRAY | GENERATE_TIMESTAMP_ARRAY)
            + LPAR
            + expr("start_date")
            + COMMA
            + expr("end_date")
            + Optional(COMMA + interval)
            + RPAR
        )

        explicit_struct = (
            STRUCT
            + Optional(LT + delimitedList(type_name) + GT)
            + LPAR
            + Optional(delimitedList(expr + Optional(AS + identifier)))
            + RPAR
        )

        case_when = WHEN + expr.copy()("when")
        case_then = THEN + expr.copy()("then")
        case_clauses = Group(ZeroOrMore(case_when + case_then))
        case_else = ELSE + expr.copy()("_else")
        case_stmt = (
            CASE
            + Optional(case_expr.copy())
            + case_clauses("case_clauses")
            + Optional(case_else)
            + END
        )("case")

        class SelectStatement(SemanticToken):
            def __init__(self, tokens):
                self.tokens = tokens

            def getName(self):
                return 'select'

            @classmethod
            def parse(cls, tokens):
                return SelectStatement(tokens)

        class Function(SemanticToken):
            def __init__(self, func, tokens):
                self.func = func
                self.tokens = tokens

            def getName(self):
                return 'function'

            @classmethod
            def parse(cls, tokens):
                method = tokens[0]
                args = tokens[2:-1]
                return Function(method, args)

            def __repr__(self):
                return "func:{}({})".format(self.func, self.tokens)


        class WindowFunction(Function):
            def __init__(self, func, tokens, func_args, partition_args, order_args, window_args):
                self.func = func
                self.tokens = tokens
                self.func_args = func_args
                self.partition_args = partition_args
                self.order_args = order_args
                self.window_args = window_args

            def getName(self):
                return 'window function'

            @classmethod
            def parse(cls, tokens):
                return WindowFunction(
                    tokens.analytic_function_name,
                    tokens,
                    tokens.function_args,
                    tokens.partition_expression_list,
                    tokens.ordering_term,
                    tokens.window_specification
                )

            def __repr__(self):
                return "window:{}({})over({}, {}, {})".format(self.func, self.func_args, self.partition_args, self.order_args, self.window_args)

        class CaseStatement(SemanticToken):
            def __init__(self, tokens, whens, _else):
                self.tokens = tokens
                self.whens = whens
                self._else = _else

            def getName(self):
                return 'case'

            @classmethod
            def parse_whens(self, tokens):
                whens = []
                while len(tokens) > 0:
                    _, when, _, then, *tokens = tokens
                    whens.append({"when": when, "then": then})
                return whens

            @classmethod
            def parse(cls, tokens):
                whens = tokens[1]
                _else = tokens[3]
                return CaseStatement(
                    tokens,
                    cls.parse_whens(whens),
                    _else
                )

            def __repr__(self):
                return "<case statement ({}, {})>".format(len(self.whens), self._else)

        expr_term = (
            (analytic_function)("analytic_function").setParseAction(WindowFunction.parse)
            | (CAST + LPAR + expr + AS + type_name + RPAR)("cast")
            | (SAFE_CAST + LPAR + expr + AS + type_name + RPAR)("safe_cast")
            | (Optional(EXISTS) + LPAR + ungrouped_select_stmt + RPAR)("subselect")
            | (literal_value)("literal")
            | (bind_parameter)("bind_parameter")
            | (EXTRACT + LPAR + expr + FROM + expr + RPAR)("extract")
            | case_stmt.setParseAction(CaseStatement.parse)
            | (datetime_operators + LPAR + expr + COMMA + interval + RPAR)(
                "date_operation"
            )
            #| (bad_datetime_operators + LPAR + expr + COMMA + interval + RPAR)
            | string_agg_term("string_agg_term")
            | array_literal("array_literal")
            | array_generator("array_generator")
            | date_array_generator("date_array_generator")
            | explicit_struct("explicit_struct")
            | function_call("function_call").setParseAction(Function.parse)
            | qualified_column_name("column").setParseAction(lambda x: ".".join([str(i) for i in x[0]]))
        ).setParseAction(debug) + Optional(LBRACKET + (OFFSET | ORDINAL) + LPAR + expr + RPAR + RBRACKET)(
            "offset_ordinal"
        )

        struct_term = (LPAR + delimitedList(expr_term) + RPAR)

        KNOWN_OPS = [
            (BETWEEN, AND),
            Literal("||").setName("concat"),
            Literal("*").setName("mul"),
            Literal("/").setName("div"),
            Literal("+").setName("add"),
            Literal("-").setName("sub"),
            Literal("<>").setName("neq"),
            Literal(">").setName("gt"),
            Literal("<").setName("lt"),
            Literal(">=").setName("gte"),
            Literal("<=").setName("lte"),
            Literal("=").setName("eq"),
            Literal("==").setName("eq"),
            Literal("!=").setName("neq"),
            IN.setName("in"),
            IS.setName("is"),
            LIKE.setName("like"),
            OR.setName("or"),
            AND.setName("and"),

            NOT.setName('not')
        ]

        class Operator(SemanticToken):
            def __init__(self, op, assoc, name, tokens):
                self.op = op
                self.assoc = assoc
                self.name = name
                self.tokens = tokens

            def getName(self):
                return 'operator'

            @classmethod
            def parse(cls, tokens):
                # ARRANGE INTO {op: params} FORMAT
                toks = tokens[0]
                if toks[1] in KNOWN_OPS:
                    op = KNOWN_OPS[KNOWN_OPS.index(toks[1])]
                    if toks.subselect:
                        import ipdb; ipdb.set_trace()
                    return Operator(op, 'binary', op.name, [toks[0], toks[2:]])
                else:
                    import ipdb; ipdb.set_trace()
                    return tokens

            @classmethod
            def parse_unary(cls, tokens):
                toks = tokens[0]
                if toks[0] in KNOWN_OPS:
                    op = KNOWN_OPS[KNOWN_OPS.index(toks[0])]
                else:
                    import ipdb; ipdb.set_trace()
                return Operator(op, 'unary', op.name, [toks[1:]])

            @classmethod
            def parse_ternary(cls, tokens):
                import ipdb; ipdb.set_trace()

            def __repr__(self):
                return "<operator({}, {}, {})>".format(self.op, self.assoc, self.tokens)

        UNARY, BINARY, TERNARY = 1, 2, 3
        expr << infixNotation(
            (expr_term | struct_term),
            [
                (oneOf("- + ~") | NOT, UNARY, opAssoc.RIGHT, Operator.parse_unary),
                (ISNULL | NOTNULL | NOT + NULL, UNARY, opAssoc.LEFT, Operator.parse_unary),
                ("||", BINARY, opAssoc.LEFT, Operator.parse),
                (oneOf("* / %"), BINARY, opAssoc.LEFT, Operator.parse),
                (oneOf("+ -"), BINARY, opAssoc.LEFT, Operator.parse),
                (oneOf("<< >> & |"), BINARY, opAssoc.LEFT, Operator.parse),
                (oneOf("= > < >= <= <> != !< !>"), BINARY, opAssoc.LEFT, Operator.parse),
                (
                    IS + Optional(NOT)
                    | Optional(NOT) + IN
                    | Optional(NOT) + LIKE
                    | GLOB
                    | MATCH
                    | REGEXP,
                    BINARY,
                    opAssoc.LEFT,
                    Operator.parse
                ),
                ((BETWEEN, AND), TERNARY, opAssoc.LEFT, Operator.parse_ternary),
                (
                    Optional(NOT)
                    + IN
                    + LPAR
                    + Group(ungrouped_select_stmt | delimitedList(expr))
                    + RPAR,
                    UNARY,
                    opAssoc.LEFT,
                    Operator.parse_unary
                ),
                (AND, BINARY, opAssoc.LEFT, Operator.parse),
                (OR, BINARY, opAssoc.LEFT, Operator.parse),
            ],
            lpar=Literal('('),
            rpar=Literal(')'),
        )
        quoted_expr = (
            expr
            ^ Suppress('"') + expr + Suppress('"')
            ^ Suppress("'") + expr + Suppress("'")
            ^ Suppress("`") + expr + Suppress("`")
        )("quoted_expr")

        compound_operator = (
            UNION + Optional(ALL | DISTINCT)
            | INTERSECT + DISTINCT
            | EXCEPT + DISTINCT
            | INTERSECT
            | EXCEPT
        )("compound_operator")

        join_constraint = Group(
            Optional(
                ON + expr
                | USING + LPAR + Group(delimitedList(qualified_column_name)) + RPAR
            )
        )("join_constraint")

        join_op = (
            COMMA
            | Group(
                Optional(NATURAL)
                + Optional(
                    INNER
                    | CROSS
                    | LEFT + OUTER
                    | LEFT
                    | RIGHT + OUTER
                    | RIGHT
                    | FULL + OUTER
                    | OUTER
                    | FULL
                )
                + JOIN
            )
        )("join_op")

        join_source = Forward()

        # We support three kinds of table identifiers.
        #
        # First, dot delimited info like project.dataset.table, where
        # each component follows the rules described in the BigQuery
        # docs, namely:
        #  Contain letters (upper or lower case), numbers, and underscores
        #
        # Second, a dot delimited quoted string.  Since it's quoted, we'll be
        # liberal w.r.t. what characters we allow.  E.g.:
        #  `project.dataset.name-with-dashes`
        #
        # Third, a series of quoted strings, delimited by dots, e.g.:
        #  `project`.`dataset`.`name-with-dashes`
        #
        # We won't attempt to support combinations, like:
        #  project.dataset.`name-with-dashes`
        #  `project`.`dataset.name-with-dashes`

        def record_table_identifier(t):
            identifier_list = t.asList()
            padded_list = [None] * (3 - len(identifier_list)) + identifier_list
            cls._table_identifiers.add(tuple(padded_list))

        standard_table_part = ~keyword + Word(alphanums + "_")
        standard_table_identifier = (
            Optional(standard_table_part("project") + Suppress("."))
            + Optional(standard_table_part("dataset") + Suppress("."))
            + standard_table_part("table")
        ).setParseAction(lambda t: record_table_identifier(t))

        quoted_project_part = (
            Suppress('"') + CharsNotIn('"') + Suppress('"')
            | Suppress("'") + CharsNotIn("'") + Suppress("'")
            | Suppress("`") + CharsNotIn("`") + Suppress("`")
        )
        quoted_table_part = (
            Suppress('"') + CharsNotIn('".') + Suppress('"')
            | Suppress("'") + CharsNotIn("'.") + Suppress("'")
            | Suppress("`") + CharsNotIn("`.") + Suppress("`")
        )
        quoted_table_parts_identifier = (
            Optional(quoted_project_part("project") + Suppress("."))
            + Optional(quoted_table_part("dataset") + Suppress("."))
            + quoted_table_part("table")
        ).setParseAction(lambda t: record_table_identifier(t))

        def record_quoted_table_identifier(t):
            identifier_list = t.asList()[0].split(".")
            first = ".".join(identifier_list[0:-2]) or None
            second = identifier_list[-2]
            third = identifier_list[-1]
            identifier_list = [first, second, third]
            padded_list = [None] * (3 - len(identifier_list)) + identifier_list
            cls._table_identifiers.add(tuple(padded_list))

        quotable_table_parts_identifier = (
            Suppress('"') + CharsNotIn('"') + Suppress('"')
            | Suppress("'") + CharsNotIn("'") + Suppress("'")
            | Suppress("`") + CharsNotIn("`") + Suppress("`")
        ).setParseAction(lambda t: record_quoted_table_identifier(t))

        table_identifier = (
            standard_table_identifier
            | quoted_table_parts_identifier
            | quotable_table_parts_identifier
        )

        def record_ref(t):
            lol = [t.op] + t.ref_target.asList()
            cls._with_aliases.add(tuple(lol))
            cls._table_identifiers.add(tuple(lol))

        ref_target = identifier.copy()
        single_source = (
            # ref + source statements
            (
                (
                    Suppress('{{')
                    + (CaselessKeyword('ref') | CaselessKeyword("source"))("op")
                    + LPAR
                    + delimitedList(
                        (Suppress("'") | Suppress('"'))
                        + ref_target
                        + (Suppress("'") | Suppress('"'))
                    )("ref_target")
                    + RPAR
                    + Suppress("}}")
                ).setParseAction(record_ref)
                | table_identifier
            )
            + Optional(Optional(AS) + table_alias("table_alias*"))
            + Optional(FOR + SYSTEMTIME + AS + OF + string_literal)
            + Optional(INDEXED + BY + index_name("name") | NOT + INDEXED)("index")
            | (
                LPAR
                + ungrouped_select_stmt
                + RPAR
                + Optional(Optional(AS) + table_alias)
            )('subquery')
            | (LPAR + join_source + RPAR)
            | (UNNEST + LPAR + expr + RPAR) + Optional(Optional(AS) + column_alias)
        )

        join_source << (
            Group(single_source + OneOrMore(Group(join_op + single_source + join_constraint)('joins*')))
            | single_source
        )('sources*')

        over_partition = (PARTITION + BY + delimitedList(partition_expression_list))(
            "over_partition"
        )
        over_order = ORDER + BY + delimitedList(ordering_term)
        over_unsigned_value_specification = expr
        over_window_frame_preceding = (
            UNBOUNDED + PRECEDING
            | over_unsigned_value_specification + PRECEDING
            | CURRENT + ROW
        )
        over_window_frame_following = (
            UNBOUNDED + FOLLOWING
            | over_unsigned_value_specification + FOLLOWING
            | CURRENT + ROW
        )
        over_window_frame_bound = (
            over_window_frame_preceding | over_window_frame_following
        )
        over_window_frame_between = (
            BETWEEN + over_window_frame_bound + AND + over_window_frame_bound
        )
        over_window_frame_extent = (
            over_window_frame_preceding | over_window_frame_between
        )
        over_row_or_range = (ROWS | RANGE) + over_window_frame_extent
        over = (
            OVER
            + LPAR
            + Optional(over_partition)
            + Optional(over_order)
            + Optional(over_row_or_range)
            + RPAR
        )("over")


        result_column = (
            Optional(table_name + ".")
            + "*"
            + Optional(
                EXCEPT
                + LPAR
                + delimitedList(column_name)
                + RPAR
            ) | Group(quoted_expr + Optional(over) + Optional(Optional(AS) + column_alias('alias')))
        )

        window_select_clause = (
            WINDOW + identifier + AS + LPAR + window_specification + RPAR
        )

        select_core = (
            SELECT
            + Optional(DISTINCT | ALL)
            + Group(delimitedList(result_column))("columns")
            + Optional(FROM - join_source("from*"))
            + Optional(WHERE + expr('where'))
            + Optional(
                GROUP + BY + Group(delimitedList(grouping_term))("group_by_terms")
            )
            + Optional(HAVING + expr("having_expr"))
            + Optional(
                ORDER + BY + Group(delimitedList(ordering_term))("order_by_terms")
            )
            + Optional(delimitedList(window_select_clause))
        )
        grouped_select_core = select_core | (LPAR + select_core + RPAR)

        ungrouped_select_stmt << (
            grouped_select_core
            + ZeroOrMore(compound_operator + grouped_select_core)
            + Optional(
                LIMIT
                + (Group(expr + OFFSET + expr) | Group(expr + COMMA + expr) | expr)(
                    "limit"
                )
            )
        )("select")
        select_stmt = ungrouped_select_stmt | (LPAR + ungrouped_select_stmt + RPAR)

        # define comment format, and ignore them
        sql_comment = oneOf("-- #") + restOfLine | cStyleComment
        select_stmt.ignore(sql_comment)

        def record_with_alias(t):
            identifier_list = t.asList()
            padded_list = [None] * (3 - len(identifier_list)) + identifier_list
            cls._with_aliases.add(tuple(padded_list))

        with_stmt = Forward().setName("with statement")
        with_clause = Group(
            identifier.setParseAction(lambda t: record_with_alias(t))('cte_name')
            - AS
            - LPAR
            + (select_stmt | with_stmt)
            - RPAR
        )
        with_core = WITH + delimitedList(with_clause)('ctes')
        with_stmt << (with_core - ~Literal(',') + ungrouped_select_stmt)
        with_stmt.ignore(sql_comment)

        select_or_with = select_stmt | with_stmt
        select_or_with_parens = LPAR + select_or_with - RPAR

        cls._parser = select_or_with | select_or_with_parens
        return cls._parser


