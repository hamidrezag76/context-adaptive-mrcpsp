from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.chromosome import Chromosome
from src.optimization.operators.mutation import Mutation


def main():

    parser = MMParser(

        Path("benchmarks/data/j301_1.mm")

    )

    project = parser.parse()

    chromosome = Chromosome(

        priority_list=[

            a.id

            for a in project.activities.values()

        ],

        mode_assignment={

            a.id: a.modes[0].id

            for a in project.activities.values()

        },

    )

    mutation = Mutation(

        project,

        probability=1.0,

        seed=42,

    )

    child = mutation.mutate(chromosome)

    print(child.priority_list[:10])

    print(

        list(child.mode_assignment.items())[:10]

    )


if __name__ == "__main__":

    main()