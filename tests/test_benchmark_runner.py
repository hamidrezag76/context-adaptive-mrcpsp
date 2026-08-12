from pathlib import Path

from src.evaluation.benchmark_runner import BenchmarkRunner


def main():

    runner = BenchmarkRunner(

        benchmark_directory=Path("benchmarks/data"),

        output_directory=Path("results/csv"),

    )

    runner.run(
        population_size=20,
        generations=10,
        limit=1,
        repetitions=2,
    )


if __name__ == "__main__":

    main()