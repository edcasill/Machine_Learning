import os
import sqlite3
import pandas as pd
#from test_db-master import employees_partitioned.sql


path = "employee_db-master/employees.db"
conn = sqlite3.connect(path)

with open('consulta.sql', 'r') as file:
    query = file.read()

df = pd.read_sql_query(query, conn)