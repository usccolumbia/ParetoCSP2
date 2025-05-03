import numpy as np

from pymoo.core.operator import Operator
from pymoo.core.population import Population
from pymoo.core.variable import Real, get

from utils.shared_data import SharedData


class Crossover(Operator):

    def __init__(self, n_parents, n_offsprings, prob=0.5, **kwargs):
        super().__init__(**kwargs)
        self.n_parents = n_parents
        self.n_offsprings = n_offsprings
        self.prob = Real(prob, bounds=(0.5, 1.0), strict=(0.0, 1.0))
        # with open('space_group_list.txt', 'r') as file:
        #     lines = file.readlines()
        #     spaceGroupList = [int(j.strip('\n')) for j in lines]
        sd = SharedData()
        spaceGroupList = sd.get_sg()
        self.spaceGroupList = spaceGroupList
        # print(spaceGroupList, 'ccccc')

    def do(self, problem, pop, parents=None, **kwargs):
        # print('Entering the do method of core.Crossover .................')

        # if a parents with array with mating indices is provided -> transform the input first
        if parents is not None:
            pop = [pop[mating] for mating in parents]

        # get the dimensions necessary to create in and output
        n_parents, n_offsprings = self.n_parents, self.n_offsprings
        n_matings, n_var = len(pop), problem.n_var

        # get the actual values from each of the parents
        X = np.swapaxes(
            np.array([[parent.get("X") for parent in mating] for mating in pop]), 0, 1
        )

        # for mating in pop:
        #     print(mating[0].age, '@@@@@@@@@@@@@')
        #     print(mating[1].age, '@@@@@@@@@@@@@')
        #     break

        if self.vtype is not None:
            X = X.astype(self.vtype)

        # the array where the offsprings will be stored to
        Xp = np.empty(shape=(n_offsprings, n_matings, n_var), dtype=X.dtype)

        # the probability of executing the crossover
        # print(self.prob, '#################')
        prob = get(self.prob, size=n_matings)
        # print(prob, '#################')

        # a boolean mask when crossover is actually executed
        cross = np.random.random(n_matings) < prob
        # print(f'cross :{cross} ######################')
        # print(f"Cross count: {sum(cross)}")

        # the design space from the parents used for the crossover
        if np.any(cross):

            # we can not prefilter for cross first, because there might be other variables using the same shape as X
            Q = self._do(problem, X, **kwargs)
            assert Q.shape == (
                n_offsprings,
                n_matings,
                problem.n_var,
            ), "Shape is incorrect of crossover impl."
            Xp[:, cross] = Q[:, cross]

        # now set the parents whenever NO crossover has been applied
        for k in np.flatnonzero(~cross):
            if n_offsprings < n_parents:
                s = np.random.choice(
                    np.arange(self.n_parents), size=n_offsprings, replace=False
                )
            elif n_offsprings == n_parents:
                s = np.arange(n_parents)
            else:
                s = []
                while len(s) < n_offsprings:
                    s.extend(np.random.permutation(n_parents))
                s = s[:n_offsprings]

            Xp[:, k] = np.copy(X[s, k])

        # flatten the array to become a 2d-array
        Xp = Xp.reshape(-1, X.shape[-1])

        # create a population object
        off = Population.new("X", Xp)

        ### sadman

        """allSgCount = {key: 0 for key in range(1, 231)}"""
        for i in range(0, len(off), 2):
            index = int(i / 2)
            if cross[index] == True:
                off[i].age = max(pop[index][0].age, pop[index][1].age) - 1
                off[i + 1].age = max(pop[index][0].age, pop[index][1].age) - 1
            else:
                off[i].age = pop[index][0].age
                off[i + 1].age = pop[index][1].age

            """sg0 = self.spaceGroupList[int(off[i]._X[6])]
            sg1 = self.spaceGroupList[int(off[i + 1]._X[6])]
            allSgCount[sg0] += 1
            allSgCount[sg1] += 1"""

        """for i in range(len(off)):
            sg0 = self.spaceGroupList[int(off[i]._X[6])]
            off[i].sgCount = allSgCount[sg0]"""

        # for i in range(len(off)):
        #     allSgCount[off[i].sgCount] = allSgCount[off[i].sgCount] + 1
        #     off[i].sgCount += 1

        """print('All space group counts after crossover:')
        for key, value in allSgCount.items():
            print(f"{key}: {'%5i' % value}")
        print(f'All space group count sum: {sum(allSgCount.values())}\n\n')"""

        # if int(off[i].age) == -1:
        #     print('\n\n\n')
        #     print('o' * 100)
        #     print('\n\n\n')

        # print('Exiting the do method of core.Crossover .................')
        return off

    def _do(self, problem, X, **kwargs):
        pass
