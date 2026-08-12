from src.context.context import Context
from src.context.operator_controller import OperatorController


def main():

    controller = OperatorController()

    context = Context(

        carbon_pressure=0.4,

        energy_pressure=0.6,

        resource_pressure=0.7,

        cost_pressure=0.3,

        schedule_pressure=0.8,

        uncertainty=0.9,

    )

    print(
        controller.crossover_probability(context)
    )

    print(
        controller.mutation_probability(context)
    )


if __name__ == "__main__":

    main()