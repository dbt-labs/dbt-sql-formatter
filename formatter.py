import sqlparse
from sql_formatter import sql_format

raw_sql = """
with wrapper as (
    with data as (
    select field1, count(*) from "dbt_dbanin"."test2" group by 1 order by 2 desc





          ), idk as (

        select distinct a,b,case when d = 'ok' then 2 else 15 end as whatever,

        row_number() over (partition by a_thing order by d rows between unbounded preceding and current row)
            from
    something
    )

    select *, row_number() over (partition by a_thing order by d rows between unbounded preceding and current row) from data
)

select
*,
row_number() over (partition by a_thing order by d rows between unbounded preceding and current row)
from             wrapper
"""

# raw_sql = """ with a as ( select a, sum(b) from zz group by 1 ), b as ( select * from zz ) select * from b """ 


parsed = sqlparse.parse(raw_sql)[0]



char = '·'
#char = ' '
print(sql_format(raw_sql, {"reindent": True, 'indent_char': char, 'indent_width': 4}))
