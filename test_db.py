import duckdb
db_path = 'upbit_data.db'
print(f"Connecting to {db_path}...")
with duckdb.connect(db_path) as con:
    print("Connected.")
    tables = con.execute("SELECT table_name FROM information_schema.tables").fetchall()
    print(f"Tables: {tables}")
