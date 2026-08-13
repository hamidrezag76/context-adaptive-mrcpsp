from __future__ import annotations

from src.models.project import Project

from src.optimization.population import Population
from src.optimization.population_initializer import PopulationInitializer

from src.optimization.decoder import Decoder
from src.evaluation.evaluator import Evaluator

from src.optimization.operators.selection import TournamentSelection
from src.optimization.operators.crossover import Crossover
from src.optimization.operators.mutation import Mutation
from src.optimization.operators.repair import Repair

from src.optimization.nsga2.fast_non_dominated_sort import FastNonDominatedSort
from src.optimization.nsga2.crowding_distance import CrowdingDistance

from src.context.context_manager import ContextManager

from src.context.operator_controller import OperatorController

from src.optimization.archive import EliteArchive

class NSGA2:
    """
    Context-Adaptive NSGA-II
    """

    def __init__(
        self,
        project: Project,
        population_size: int = 50,
        generations: int = 100,
        seed: int | None = None,
        context_adaptive: bool = True,
        operator_adaptive: bool = True,
    ):

        self.project = project

        self.population_size = population_size

        self.generations = generations

        self.seed = seed

        self.context_adaptive = context_adaptive

        self.operator_adaptive = operator_adaptive

        self.population = Population()

        self.initializer = PopulationInitializer(
            project,
            seed=seed,
        )

        self.decoder = Decoder(project)
        
        self.evaluator = Evaluator(project)

        self.selection = TournamentSelection(seed=seed)

        self.crossover = Crossover(
            seed=seed,
        )

        self.mutation = Mutation(
            project,
            seed=seed,
        )

        self.repair = Repair(
            project,
        )

        self.fast_sort = FastNonDominatedSort()

        self.crowding = CrowdingDistance()

        self.context = ContextManager(
            self.project,
        )
        
        self.operator_controller = OperatorController()
        
        self.history = []
        
        self.archive = EliteArchive(
            maximum_size=max(
                100,
                self.population_size * 2,
            )
        )
            # -----------------------------------------------------

    def initialize(self):

        self.population = self.initializer.initialize(
            self.population_size,
        )
            # -----------------------------------------------------

    def evaluate_population(self):

        for chromosome in self.population:

            context = self.context.get()

            result = self.decoder.decode_and_evaluate(

                chromosome,

                self.evaluator,

                context,

            )

            chromosome.makespan = result.makespan

            chromosome.total_cost = result.total_cost

            chromosome.total_carbon = result.total_carbon

            chromosome.total_energy = result.total_energy
            
    def assign_rank_and_crowding(self):

        fronts = self.fast_sort.sort(
            self.population.individuals,
        )

        for rank, front in enumerate(fronts):

            for chromosome in front:

                chromosome.rank = rank

            self.crowding.compute(front)
            
    def prepare(self):

        self.initialize()

        self.evaluate_population()

        self.assign_rank_and_crowding()

        if self.context_adaptive:

            self.context.update(
                self.population,
                generation=0,
                max_generations=self.generations,
            )
        
    def create_offspring(
        self,
    ):

        offspring = Population()

        while len(offspring) < self.population_size:

            # ----------------------------
            # Parent Selection
            # ----------------------------

            parent1, parent2 = self.selection.select_pair(
                self.population.individuals
            )

            # ----------------------------
            # Crossover
            # ----------------------------

            child1, child2 = self.crossover.crossover(
                parent1,
                parent2,
            )

            # ----------------------------
            # Mutation
            # ----------------------------

            child1 = self.mutation.apply(
                child1
            )

            child2 = self.mutation.apply(
                child2
            )

            # ----------------------------
            # Repair
            # ----------------------------

            # child1 = self.repair.apply(child1)
            # child2 = self.repair.apply(child2)

            offspring.add(child1)

            if len(offspring) < self.population_size:

                offspring.add(child2)
            
        return offspring
    
    def survival_selection(
        self,
        offspring,
    ):

        merged = Population()

        merged.extend(
            self.population.individuals
        )

        merged.extend(
            offspring.individuals
        )

        fronts = self.fast_sort.sort(
            merged.individuals
        )
        
        for rank, front in enumerate(fronts):

            for chromosome in front:

                chromosome.rank = rank

        new_population = Population()

        for front in fronts:

            self.crowding.compute(
                front
            )

            front.sort(
                key=lambda c: -c.crowding_distance
            )

            for chromosome in front:

                if len(new_population) >= self.population_size:

                    break

                new_population.add(
                    chromosome
                )

            if len(new_population) >= self.population_size:

                break

        self.population = new_population
        
        self.assign_rank_and_crowding()
        
    def update_archive(self) -> None:
        """
        Update the external Pareto archive using the
        current population.

        Independent chromosome copies are stored so that
        later NSGA-II rank/crowding updates do not mutate
        archived solutions.
        """

        if self.context_adaptive:

            self.archive.set_context(
                self.context.get()
            )

        solutions = [
            chromosome.copy()
            for chromosome in self.population.individuals
        ]

        self.archive.update(
            solutions
        )
        
    def _record_history(self, generation: int) -> None:

        context = self.context.get()

        individuals = self.population.individuals

        if not individuals:
            return

        best_makespan = min(
            c.makespan
            for c in individuals
        )

        best_cost = min(
            c.total_cost
            for c in individuals
        )

        best_carbon = min(
            c.total_carbon
            for c in individuals
        )

        best_energy = min(
            c.total_energy
            for c in individuals
        )

        self.history.append(
            {
                "generation": generation,

                "carbon_pressure":
                    context.carbon_pressure,

                "energy_pressure":
                    context.energy_pressure,

                "resource_pressure":
                    context.resource_pressure,

                "cost_pressure":
                    context.cost_pressure,

                "schedule_pressure":
                    context.schedule_pressure,

                "uncertainty":
                    context.uncertainty,

                "crossover_probability":
                    self.crossover.probability,

                "mutation_probability":
                    self.mutation.probability,

                "best_makespan":
                    best_makespan,

                "best_cost":
                    best_cost,

                "best_carbon":
                    best_carbon,

                "best_energy":
                    best_energy,
            }
        )
        
    def run(
        self,
    ):
        """
        Run the Context-Adaptive NSGA-II algorithm.

        The context and adaptive operator probabilities are
        synchronized at every generation.
        """

        self.prepare()
        
        self.update_archive()

        # ---------------------------------------------------------
        # Generation 0
        #
        # The initial population has already been evaluated by
        # prepare(). Therefore, the initial context is computed
        # before recording the first history entry.
        # ---------------------------------------------------------

        if self.context_adaptive and self.operator_adaptive:

            context = self.context.get()

            self.crossover.probability = (
                self.operator_controller.crossover_probability(
                    context
                )
            )

            self.mutation.probability = (
                self.operator_controller.mutation_probability(
                    context
                )
            )

        else:

            self.crossover.probability = 0.90

            self.mutation.probability = 0.15

        self._record_history(0)

        # ---------------------------------------------------------
        # Evolutionary loop
        # ---------------------------------------------------------

        for generation in range(self.generations):

            # -----------------------------------------------------
            # Create offspring
            # -----------------------------------------------------

            offspring = self.create_offspring()

            # -----------------------------------------------------
            # Evaluate offspring
            # -----------------------------------------------------

            for chromosome in offspring:

                context = self.context.get()

                result = self.decoder.decode_and_evaluate(
                    chromosome,
                    self.evaluator,
                    context,
                )

                chromosome.makespan = result.makespan

                chromosome.total_cost = result.total_cost

                chromosome.total_carbon = result.total_carbon

                chromosome.total_energy = result.total_energy

            # -----------------------------------------------------
            # Survival selection
            # -----------------------------------------------------

            self.survival_selection(
                offspring,
            )
            
            self.update_archive()

            # -----------------------------------------------------
            # Update context AFTER the new population has been
            # established.
            # -----------------------------------------------------

            if self.context_adaptive:

                self.context.update(
                    self.population,
                    generation=generation + 1,
                    max_generations=self.generations,
                )

                context = self.context.get()

                if self.operator_adaptive:

                    self.crossover.probability = (
                        self.operator_controller.crossover_probability(
                            context
                        )
                    )

                    self.mutation.probability = (
                        self.operator_controller.mutation_probability(
                            context
                        )
                    )

                else:

                    self.crossover.probability = 0.90

                    self.mutation.probability = 0.15

            else:

                self.crossover.probability = 0.90

                self.mutation.probability = 0.15

            # -----------------------------------------------------
            # Record synchronized context + operator parameters
            # -----------------------------------------------------

            self._record_history(
                generation + 1
            )

            # -----------------------------------------------------
            # Statistics
            # -----------------------------------------------------

            best = min(
                self.population.individuals,
                key=lambda c: c.makespan,
            )

        return self.population
        
    
