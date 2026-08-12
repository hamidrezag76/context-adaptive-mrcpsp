from src.optimization.chromosome import Chromosome
from src.optimization.nsga2.crowding_distance import CrowdingDistance


def main():

    front = [

        Chromosome([], {}),
        Chromosome([], {}),
        Chromosome([], {}),
        Chromosome([], {}),

    ]

    front[0].makespan = 10
    front[1].makespan = 15
    front[2].makespan = 18
    front[3].makespan = 25

    front[0].total_cost = 40
    front[1].total_cost = 30
    front[2].total_cost = 25
    front[3].total_cost = 20

    front[0].total_carbon = 50
    front[1].total_carbon = 40
    front[2].total_carbon = 30
    front[3].total_carbon = 20

    front[0].total_energy = 60
    front[1].total_energy = 55
    front[2].total_energy = 45
    front[3].total_energy = 30

    CrowdingDistance().assign(front)

    for c in front:

        print(
            c.objectives,
            c.crowding_distance
        )


if __name__ == "__main__":
    main()
