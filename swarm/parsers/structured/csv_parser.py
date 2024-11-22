from typing import AsyncGenerator, Union

from swarm.models.document import DataType
from swarm.base.base_parser import AsyncParser




class CSVParser(AsyncParser[DataType]):
    """A parser for CSV data."""

    def __init__(self):
        import csv
        from io import StringIO

        self.csv = csv
        self.StringIO = StringIO

    async def ingest(
        self, data: Union[str, bytes]
    ) -> AsyncGenerator[str, None]:
        """Ingest CSV data and yield text from each row."""
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        csv_reader = self.csv.reader(self.StringIO(data))
        for row in csv_reader:
            yield ", ".join(row)
