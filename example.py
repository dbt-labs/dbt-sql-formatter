
import sql_parser
from format import Formatter

parser = sql_parser.BigQueryViewParser()
sql = """
with my_cte as (select sum(case when a=1 then 1 else 0 end) as pivoted from table) select * from my_cte
"""

ast = parser._parse(sql)
f = Formatter()
f.format(ast)
f.document.pprint()
