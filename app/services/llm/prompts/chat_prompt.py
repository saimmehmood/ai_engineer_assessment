from app.services.llm.prompts import prompt

def get_duckdb_schema_summary():
    from app.services.datastore.duckdb_datastore import DuckDBDatastore
    db = DuckDBDatastore(database="app/data/data.db")
    tables = db.connection.execute("SHOW TABLES").fetchall()
    schema_lines = []
    for (table_name,) in tables:
        cols = db.connection.execute(f"DESCRIBE {table_name}").fetchall()
        col_names = [col[0] for col in cols]
        schema_lines.append(f"- {table_name} ({', '.join(col_names)})")
    return "Tables and columns:\n" + "\n".join(schema_lines)

@prompt()
def chat_prompt(**kwargs) -> str:
    schema_summary = get_duckdb_schema_summary()
    return [
        {
            "role": "system",
            "content": f"""
You are an expert financial data analyst assistant.

{schema_summary}

IMPORTANT:
- Only use the tables and columns listed above.
- Table and column names are case-sensitive and must be used exactly as shown.
- Do NOT invent tables or columns (e.g., do NOT use 'sales_data').
- If unsure, ask the user for clarification.
- After executing the SQL, explain the results in clear business language.
"""
        }
    ]
