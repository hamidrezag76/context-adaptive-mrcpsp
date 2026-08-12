from __future__ import annotations

from src.models.project import Project
from src.parser.psplib_reader import PSPLIBReader


class ProjectInformationParser:

    def __init__(
        self,
        reader: PSPLIBReader,
    ):

        self.reader = reader

    # -----------------------------------------------------

    def parse(
        self,
        project: Project,
    ) -> None:

        self.reader.goto(
            "PROJECT INFORMATION"
        )

        # عنوان بخش
        self.reader.next_line()

        # هدر جدول
        self.reader.next_line()

        line = self.reader.next_line().strip()

        tokens = line.split()

        project.horizon = int(tokens[3])

        while not self.reader.eof():

            line = self.reader.next_line().strip()

            if line.startswith("***"):
                break

            if not line:
                continue

            tokens = line.split()

            if len(tokens) >= 6:

                project.horizon = int(tokens[3])

                break