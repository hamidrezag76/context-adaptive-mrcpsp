from src.optimization.population import Population
from src.optimization.chromosome import Chromosome


def main():

    chromosome = Chromosome(

        priority_list=[1, 2, 3],

        mode_assignment={
            1: 1,
            2: 1,
            3: 1,
        },

    )

    population = Population()

    population.add(chromosome)

    print(len(population))

    for individual in population:

        print(individual.priority_list)

        print(individual.mode_assignment)


if __name__ == "__main__":

    main()
