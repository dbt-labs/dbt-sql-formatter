A validating SQL parser that can lint select statements

Lots left to do here:
1. Finish formatting logic (only partially implemented)
2. Cross-db support (Snowflake, Redshift, BigQuery, Postgres, etc)
3. Handle more jinja blocks (config, ref, source, etc)
4. Add validating to formatting code (make sure we don't drop tokens)
5. Make this distributable (editor plugin at a POC?)
6. Lots and lots of testing!

### Requirements
```
pip install pyparsing
```


### Example
```
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
```

Output:

```
with my_cte as (

    select
        sum(case
            when a = 1 then 1
            else 0
        end) as pivoted

    from table

)

select
    *

from my_cte
```

### Thanks

Heavily inspired by (and partially copied from) code in:
 - https://github.com/mozilla/moz-sql-parser
 - https://github.com/pyparsing/pyparsing/blob/master/examples/bigquery_view_parser.py
