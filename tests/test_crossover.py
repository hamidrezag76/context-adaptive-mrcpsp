from src.optimization.chromosome import Chromosome
from src.optimization.operators.crossover import Crossover


def main():

    p1 = Chromosome(

        priority_list=[1,2,3,4,5,6,7],

        mode_assignment={
            1:1,
            2:1,
            3:1,
            4:1,
            5:1,
            6:1,
            7:1,
        },
    )

    p2 = Chromosome(

        priority_list=[7,6,5,4,3,2,1],

        mode_assignment={
            1:2,
            2:2,
            3:2,
            4:2,
            5:2,
            6:2,
            7:2,
        },
    )

    op = Crossover(

        probability=1.0,

        seed=42,

    )

    c1, c2 = op.crossover(

        p1,

        p2,

    )

    print(c1.priority_list)

    print(c2.priority_list)

    print(c1.mode_assignment)

    print(c2.mode_assignment)


if __name__ == "__main__":

    main()