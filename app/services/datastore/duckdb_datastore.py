import duckdb
import pandas as pd
from typing import Optional, Dict, Any



class DuckDBDatastore:
    """
    A datastore implementation for DuckDB.
    """

    def __init__(self, database: Optional[str] = None) -> None:
        """
        Initialize the DuckDBDataStore.

        Args:
            database (str, optional): Path to the DuckDB database file.
                                      If None, an in-memory database is used.
        """
        if database is None:
            database = ':memory:'
        self.connection = duckdb.connect(database=database)

    def execute(
        self, query: str, parameters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Execute a SQL query and return the result as a DataFrame.

        Args:
            query (str): The SQL query to execute.
            parameters (Dict[str, Any], optional): Parameters to include in the query.

        Returns:
            pd.DataFrame: The query result.
        """
        if parameters:
            return self.connection.execute(query, parameters).df()
        else:
            return self.connection.execute(query).df()
        

    def get_columns(
        self, table_name: str, schema_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve column information for a specific table.

        Args:
            table_name (str): Name of the table.
            schema_name (str, optional): Schema name.

        Returns:
            pd.DataFrame: DataFrame with column information.
        """
        schema_filter = f"AND table_schema = '{schema_name}'" if schema_name else ""
        query = f"""
        SELECT column_name, data_type, is_nullable, character_maximum_length
        FROM information_schema.columns
        WHERE table_name = '{table_name}' {schema_filter}
        """
        return self.execute(query)

    def get_sample_data(
        self, table_name: str, limit: int = 5, schema_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Retrieve a sample of data from a specific table.

        Args:
            table_name (str): Name of the table.
            limit (int, optional): Number of rows to retrieve. Defaults to 5.
            schema_name (str, optional): Schema name.

        Returns:
            pd.DataFrame: DataFrame with sample data.
        """
        schema_prefix = f"{schema_name}." if schema_name else ""
        query = f"""
        SELECT *
        FROM {schema_prefix}{table_name}
        ORDER BY RANDOM()
        LIMIT {limit}
        """
        return self.execute(query)
