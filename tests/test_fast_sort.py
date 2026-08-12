from src.optimization.chromosome import Chromosome
from src.optimization.fast_non_dominated_sort import FastNonDominatedSort


def main():

    c1 = Chromosome(
        priority_list=[],
        mode_assignment={}
    )
    c1.makespan = 10
    c1.total_cost = 20
    c1.total_carbon = 30
    c1.total_energy = 40

    c2 = Chromosome(
        priority_list=[],
        mode_assignment={}
    )
    c2.makespan = 15
    c2.total_cost = 30
    c2.total_carbon = 35
    c2.total_energy = 45

    c3 = Chromosome(
        priority_list=[],
        mode_assignment={}
    )
    c3.makespan = 9
    c3.total_cost = 50
    c3.total_carbon = 20
    c3.total_energy = 60

    population = [c1, c2, c3]

    sorter = FastNonDominatedSort()

    fronts = sorter.sort(population)

    for i, front in enumerate(fronts):

        print(f"Front {i+1}")

        for c in front:

            print(
                c.rank,
                c.objectives
            )


if __name__ == "__main__":

    main()
