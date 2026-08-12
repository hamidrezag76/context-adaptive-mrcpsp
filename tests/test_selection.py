from src.optimization.chromosome import Chromosome
from src.optimization.operators.selection import TournamentSelection


def main():

    p = []

    for i in range(10):

        c = Chromosome(
            priority_list=[1, 2, 3],
            mode_assignment={1: 1},
        )

        c.rank = i % 3
        c.crowding_distance = float(i)

        p.append(c)

    selector = TournamentSelection(seed=42)

    parent = selector.select(p)

    print(parent.rank)
    print(parent.crowding_distance)


if __name__ == "__main__":

    main()