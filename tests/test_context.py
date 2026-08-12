from src.context.context import Context
from src.context.adaptive_operator import AdaptiveOperator


def main():

    context = Context(

        carbon_pressure=0.90,

        energy_pressure=0.80,

        resource_pressure=0.40,

        cost_pressure=0.30,

        schedule_pressure=0.75,

        uncertainty=0.85,

    )

    operator = AdaptiveOperator()

    result = operator.compute(context)

    print(result)


if __name__ == "__main__":

    main()
