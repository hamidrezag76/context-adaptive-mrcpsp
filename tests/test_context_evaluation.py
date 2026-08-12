from pathlib import Path

from src.parser.mm_parser import MMParser
from src.optimization.population_initializer import PopulationInitializer
from src.optimization.decoder import Decoder
from src.evaluation.evaluator import Evaluator
from src.context.context import Context


def main():

    project = MMParser(
        Path("benchmarks/data/j301_1.mm")
    ).parse()

    chromosome = PopulationInitializer(project).initialize(1)[0]

    decoder = Decoder(project)

    evaluator = Evaluator(project)

    context = Context.neutral()

    result = decoder.decode_and_evaluate(

        chromosome,

        evaluator,

        context,

    )

    print(result)


if __name__ == "__main__":
    main()