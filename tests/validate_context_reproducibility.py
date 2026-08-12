from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.population_initializer import PopulationInitializer
from src.optimization.decoder import Decoder
from src.evaluation.evaluator import Evaluator
from src.context.context_manager import ContextManager


INSTANCE = Path(
    "benchmarks/data/j3010_1.mm"
)


def run(seed: int):
    project = MMParser(
        INSTANCE
    ).parse()

    population = PopulationInitializer(
        project,
        seed=seed,
    ).initialize(10)

    decoder = Decoder(project)
    evaluator = Evaluator(project)

    context_manager = ContextManager(
        project,
        seed=seed,
    )

    values = []

    for generation in range(10):

        # -------------------------------------------------
        # Evaluate population
        # -------------------------------------------------

        for chromosome in population.individuals:

            decoded = decoder.decode(
                chromosome
            )

            result = evaluator.evaluate(
                decoded
            )

            chromosome.decoded_schedule = decoded

            chromosome.makespan = (
                result.makespan
            )

            chromosome.total_cost = (
                result.total_cost
            )

            chromosome.total_carbon = (
                result.total_carbon
            )

            chromosome.total_energy = (
                result.total_energy
            )

        # -------------------------------------------------
        # Update context from evaluated population
        # -------------------------------------------------

        context = context_manager.update(
            population,
            generation=generation,
            max_generations=10,
        )

        values.append(
            tuple(
                round(x, 10)
                for x in context.as_vector()
            )
        )

    return values


# =========================================================
# Reproducibility
# =========================================================

a = run(42)
b = run(42)
c = run(43)


print(
    "=== CONTEXT REPRODUCIBILITY ==="
)

print(
    "Same seed identical:",
    a == b,
)

print(
    "Different seed different:",
    a != c,
)


print(
    "\nFirst trajectory:"
)

for i, values in enumerate(a):

    print(
        i,
        values,
    )


# =========================================================
# Validation
# =========================================================

assert a == b, (
    "Context trajectory is not reproducible "
    "for identical seeds."
)

assert a != c, (
    "Different seeds produced identical "
    "population-driven context trajectories."
)


print(
    "\nPASS"
)