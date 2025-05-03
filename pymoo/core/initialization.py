import numpy as np

from pymoo.core.duplicate import NoDuplicateElimination
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair
from pymoo.util.misc import at_least_2d_array


class Initialization:

    def __init__(self, sampling, repair=None, eliminate_duplicates=None) -> None:

        super().__init__()
        self.sampling = sampling
        self.eliminate_duplicates = (
            eliminate_duplicates if eliminate_duplicates else NoDuplicateElimination()
        )
        self.repair = repair if repair is not None else NoRepair()

    def do(self, problem, n_samples, **kwargs):
        # print(problem, "xxx")

        # provide a whole population object - (individuals might be already evaluated)
        if isinstance(self.sampling, Population):
            # print("a")
            pop = self.sampling

        else:
            if isinstance(self.sampling, np.ndarray):
                # print("b")
                sampling = at_least_2d_array(self.sampling)
                pop = Population.new(X=sampling)
            else:
                # print("c")
                pop = self.sampling(problem, n_samples, **kwargs)
                # print(type(pop))
                # print(len(pop))
                # print(type(pop[0]))
                # print(pop[0]._X)

        # repair all solutions that are not already evaluated
        not_eval_yet = [k for k in range(len(pop)) if len(pop[k].evaluated) == 0]
        if len(not_eval_yet) > 0:
            pop[not_eval_yet] = self.repair(problem, pop[not_eval_yet], **kwargs)

        # filter duplicate in the population
        pop = self.eliminate_duplicates.do(pop)
        # print(type(pop))
        # print(len(pop))
        # print(type(pop[0]))
        # print("2222222222222222")

        return pop
