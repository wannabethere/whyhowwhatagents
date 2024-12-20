from typing import AsyncGenerator

from bs4 import BeautifulSoup

from swarm.models.document import DataType
from swarm.base.base_parser import AsyncParser




class HTMLParser(AsyncParser[DataType]):
    """A parser for HTML data."""

    async def ingest(self, data: DataType) -> AsyncGenerator[str, None]:
        """Ingest HTML data and yield text."""
        soup = BeautifulSoup(data, "html.parser")
        yield soup.get_text()
