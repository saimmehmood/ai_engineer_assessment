from langfuse.decorators import observe
from vaul import tool_call

from app import logger
from app.services.datastore.duckdb_datastore import DuckDBDatastore
from app.services.llm.structured_outputs.text_to_sql import SqlQuery
import re


def ensure_limit(sql, default_limit=20):
    # If there's already a LIMIT, do nothing
    if re.search(r'\bLIMIT\b', sql, re.IGNORECASE):
        return sql
    # Otherwise, add LIMIT at the end
    return sql.rstrip(';') + f' LIMIT {default_limit};'

@tool_call
@observe
def text_to_sql(query: str) -> str:
    """A tool for converting natural language queries to SQL queries and executing them against the financial database."""

    logger.info(f"Converting natural language query to SQL query: {query}")

    # Initialize the DuckDB datastore
    datastore = DuckDBDatastore(
        database="app/data/data.db"
    )

    # First, generate the SQL query using structured output
    from app.services.llm.session import LLMSession
    from app.services.llm.structured_outputs.text_to_sql import SqlQuery
    
    # Create LLMSession with default models to avoid app context issues
    try:
        from flask import current_app
        llm_session = LLMSession(
            chat_model=current_app.config.get("CHAT_MODEL", "gpt-4o-mini"),
            embedding_model=current_app.config.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        )
    except RuntimeError:
        # Fallback when outside app context
        llm_session = LLMSession(
            chat_model="gpt-4o-mini",
            embedding_model="text-embedding-3-small",
        )
    
    # Create a prompt for SQL generation with database schema context
    sql_generation_prompt = [
        {
            "role": "system",
            "content": """You are an expert SQL developer specializing in financial data analysis. Your task is to convert natural language questions into accurate SQL queries. For all queries, limit the number of rows returned to 20 unless the user specifies otherwise.

DATABASE SCHEMA:
- account: Financial accounts (Key, ParentId, Name, AccountType, CalculationMethod, DebitCredit, NumericFormat)
- customer: Customer data (Key, ParentId, Name, Channel, Industry, Location, Sales Manager, Salesperson)
- product: Product catalog (Key, ParentId, Name, Product Line)
- time: Time dimensions (Name, Month, Year, Quarter, FiscalQuarterNumber, FiscalMonthNumber)
- time_perspective: Time perspectives (Key, Name, CalculationMethod, MemberType)

SQL GUIDELINES:
1. Use appropriate JOINs when combining data from multiple tables
2. Apply proper aggregations (SUM, COUNT, AVG, etc.) for financial metrics
3. Include WHERE clauses for filtering when relevant
4. Use ORDER BY for ranking and sorting
5. Limit results when appropriate (LIMIT clause)
6. Handle NULL values appropriately
7. Use proper financial terminology in column aliases

Generate only the SQL query without any explanation."""
        },
        {
            "role": "user",
            "content": f"Convert this question to SQL: {query}"
        }
    ]
    
    try:
        # Generate SQL using structured output
        sql_result = llm_session.get_structured_output(sql_generation_prompt, SqlQuery)
        generated_sql = sql_result.query
        generated_sql = ensure_limit(generated_sql, default_limit=20)
        logger.info(f"Generated SQL: {generated_sql}")
        
        # Execute the generated SQL
        result = datastore.execute(generated_sql)
        
        # Return the result as plain markdown (not JSON-encoded)
        return result.to_markdown(index=False, floatfmt=".2f") if result is not None else "No data found."
        
    except Exception as e:
        logger.error(f"Error in text_to_sql: {e}")
        return f"Error generating or executing SQL: {str(e)}"