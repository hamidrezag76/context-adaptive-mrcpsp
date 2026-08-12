from __future__ import annotations

from src.models.project import Project
from src.parser.psplib_reader import PSPLIBReader


class PrecedenceParser:

    def __init__(
        self,
        reader: PSPLIBReader,
    ):

        self.reader = reader

    # ------------------------------------------------------

    def parse(
        self,
        project: Project,
    ) -> None:

        self.reader.goto(
            "PRECEDENCE RELATIONS"
        )

        # header
        self.reader.next_line()
        self.reader.next_line()

        while not self.reader.eof():

            line = self.reader.next_line().strip()

            if line.startswith("***"):
                break

            if line == "":
                continue

            tokens = line.split()

            try:

                activity = int(tokens[0])

                n_successors = int(tokens[2])

            except Exception:

                continue

            successors = [
                int(x)
                for x in tokens[3:3 + n_successors]
            ]

            for successor in successors:

                project.add_precedence(
                    activity,
                    successor,
                )