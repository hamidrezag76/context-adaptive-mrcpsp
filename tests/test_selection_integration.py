from src.context.context import Context
from src.optimization.ca_nsga2 import NSGA2
from src.parser.mm_parser import MMParser


INSTANCE = "benchmarks/data/j3010_1.mm"


def test_baseline_selection_receives_no_context():

    project = MMParser(INSTANCE).parse()

    algorithm = NSGA2(
        project=project,
        population_size=20,
        generations=1,
        seed=42,
        context_adaptive=False,
        operator_adaptive=False,
    )

    algorithm.prepare()

    captured = {}

    original = algorithm.selection.select_pair

    def spy_select_pair(
        population,
        context=None,
    ):
        captured["context"] = context

        return original(
            population,
            context=context,
        )

    algorithm.selection.select_pair = spy_select_pair

    algorithm.create_offspring()

    assert "context" in captured
    assert captured["context"] is None


def test_context_adaptive_selection_receives_context():

    project = MMParser(INSTANCE).parse()

    algorithm = NSGA2(
        project=project,
        population_size=20,
        generations=1,
        seed=42,
        context_adaptive=True,
        operator_adaptive=False,
    )

    algorithm.prepare()

    captured = {}

    original = algorithm.selection.select_pair

    def spy_select_pair(
        population,
        context=None,
    ):
        captured["context"] = context

        return original(
            population,
            context=context,
        )

    algorithm.selection.select_pair = spy_select_pair

    algorithm.create_offspring()

    assert "context" in captured
    assert captured["context"] is not None
    assert isinstance(
        captured["context"],
        Context,
    )


def test_context_adaptive_selection_is_enabled_only_in_context_mode():

    project = MMParser(INSTANCE).parse()

    baseline = NSGA2(
        project=project,
        population_size=20,
        generations=1,
        seed=42,
        context_adaptive=False,
        operator_adaptive=False,
    )

    context_only = NSGA2(
        project=project,
        population_size=20,
        generations=1,
        seed=42,
        context_adaptive=True,
        operator_adaptive=False,
    )

    assert baseline.selection.context_adaptive is False
    assert context_only.selection.context_adaptive is True