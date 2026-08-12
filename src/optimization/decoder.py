"""
decoder.py

Chromosome Decoder

Converts a chromosome into a feasible project schedule
using the Serial Schedule Generation Scheme (SSGS).

Author
------
CA-SMRCPSP Research Project
"""

from __future__ import annotations


from src.optimization.decoded_solution import DecodedSolution
from src.models.project import Project
from src.optimization.chromosome import Chromosome
from src.scheduling.ssgs import SSGS
from src.evaluation.evaluator import Evaluator



class Decoder:
    """
    Chromosome Decoder.
    """

    def __init__(
        self,
        project: Project,
    ) -> None:
        """
        Initialize decoder.
        """

        self.project = project

        self.ssgs = SSGS(project)
        


    # ---------------------------------------------------------
    # Decode
    # ---------------------------------------------------------

    def decode(
        self,
        chromosome: Chromosome,
    ) -> DecodedSolution:
        """
        Decode chromosome into a feasible schedule.
        """

        for activity_id, mode_id in chromosome.mode_assignment.items():

            activity = self.project.get_activity(
                activity_id,
            )

            activity.selected_mode = mode_id

        priority = chromosome.priority_list.copy()

        decoded = self.ssgs.generate(
            priority,
            chromosome.mode_assignment,
        )

        decoded.mode_assignment = chromosome.mode_assignment.copy()

        chromosome.priority_list = priority

        return decoded

    # ---------------------------------------------------------

    def decode_and_evaluate(
        self,
        chromosome,
        evaluator,
        context,
    ):

        decoded = self.decode(chromosome)
        
        chromosome.decoded_schedule = decoded

        result = evaluator.evaluate(
            decoded,
            context,
        )

        return result