from __future__ import annotations

from pathlib import Path


class PSPLIBReader:
    """
    Low-level reader for PSPLIB benchmark files.

    Responsibilities
    ----------------
    - Load file
    - Strip comments
    - Preserve line order
    - Provide sequential access
    """

    def __init__(self, filepath: str | Path):

        self.filepath = Path(filepath)

        self.lines: list[str] = []
        
        self.index = 0

    def read(self) -> list[str]:

        with self.filepath.open(
            "r",
            encoding="utf-8",
            errors="ignore",
        ) as f:

            self.lines = [
                line.rstrip("\n")
                for line in f
            ]

        return self.lines
        # ---------------------------------------------------------

    def find_section(
        self,
        keyword: str,
    ) -> int:
        """
        Find a section header.

        Returns
        -------
        line index

        Raises
        ------
        ValueError
        """

        keyword = keyword.upper()

        for i, line in enumerate(self.lines):

            if keyword in line.upper():

                return i

        raise ValueError(
            f"Section '{keyword}' not found."
        )

    # ---------------------------------------------------------

    def section(
        self,
        keyword: str,
    ) -> list[str]:
        """
        Return lines starting from
        the requested section.
        """

        start = self.find_section(keyword)

        return self.lines[start:]
        # ---------------------------------------------------------

    def goto(
        self,
        keyword: str,
    ) -> None:

        self.index = self.find_section(keyword)

    # ---------------------------------------------------------

    def next_line(self) -> str:

        line = self.lines[self.index]

        self.index += 1

        return line

    # ---------------------------------------------------------

    def eof(self) -> bool:

        return self.index >= len(self.lines)