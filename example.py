
import sql_parser
from format import Formatter
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("filepath")
args = argparser.parse_args()

parser = sql_parser.BigQueryViewParser()

with open(args.filepath, 'r') as f:
    sql = f.read()

ast = parser._parse(sql)

f = Formatter()
f.format(ast)
f.document.pprint()
