from pydantic import Field
from vaul import StructuredOutput


class SqlQuery(StructuredOutput):
    """A SQL query for retrieving data from a given table."""
    query: str = Field(..., title="A generated SQL query for retrieving data from the table.")