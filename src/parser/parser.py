"""
parser.py

Abstract parser interface for the Context-Adaptive Sustainable
Multi-Mode Resource-Constrained Project Scheduling Problem
(CA-SMRCPSP).

Every parser in this project must inherit from BaseParser.

Examples
--------
PSPLIBParser
JSONParser
XMLParser
CustomIndustryParser

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.models.project import Project


class BaseParser(ABC):
    """
    Abstract parser interface.

    Every project parser must inherit from this class.
    """

    @abstractmethod
    def load(
        self,
        file_path: str | Path,
    ) -> Project:
        """
        Read one scheduling instance and return a Project.

        Parameters
        ----------
        file_path
            Path to benchmark instance.

        Returns
        -------
        Project

        Raises
        ------
        FileNotFoundError
        ValueError
        """

        raise NotImplementedError

    @abstractmethod
    def supports(
        self,
        file_path: str | Path,
    ) -> bool:
        """
        Returns True if parser supports the file.

        Parameters
        ----------
        file_path
            File path.

        Returns
        -------
        bool
        """

        raise NotImplementedError
