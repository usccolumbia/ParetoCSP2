import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Structure, PeriodicSite

from pyxtal import pyxtal
from tqdm import tqdm

import gc

from pymoo.core.algorithm import Algorithm
from pymoo.core.duplicate import DefaultDuplicateElimination, NoDuplicateElimination
from pymoo.core.initialization import Initialization
from pymoo.core.mating import Mating
from pymoo.core.population import Population
from pymoo.core.repair import NoRepair

from utils.shared_data import SharedData


class GeneticAlgorithm(Algorithm):

    def __init__(
        self,
        pop_size=None,
        sampling=None,
        selection=None,
        crossover=None,
        mutation=None,
        survival=None,
        n_offsprings=None,
        eliminate_duplicates=DefaultDuplicateElimination(),
        repair=None,
        mating=None,
        advance_after_initial_infill=False,
        **kwargs,
    ):

        super().__init__(**kwargs)

        # the population size used
        self.pop_size = pop_size

        # whether the algorithm should be advanced after initialization of not
        self.advance_after_initial_infill = advance_after_initial_infill

        # the survival for the genetic algorithm
        self.survival = survival

        # number of offsprings to generate through recombination
        self.n_offsprings = n_offsprings

        # if the number of offspring is not set - equal to population size
        if self.n_offsprings is None:
            self.n_offsprings = pop_size

        # set the duplicate detection class - a boolean value chooses the default duplicate detection
        if isinstance(eliminate_duplicates, bool):
            if eliminate_duplicates:
                self.eliminate_duplicates = DefaultDuplicateElimination()
            else:
                self.eliminate_duplicates = NoDuplicateElimination()
        else:
            self.eliminate_duplicates = eliminate_duplicates

        # simply set the no repair object if it is None
        self.repair = repair if repair is not None else NoRepair()

        self.initialization = Initialization(
            sampling, repair=self.repair, eliminate_duplicates=self.eliminate_duplicates
        )

        if mating is None:
            # print(
            #     "Mating not provided, need to do using mutation and crossover function ******************\n"
            # )
            mating = Mating(
                selection,
                crossover,
                mutation,
                repair=self.repair,
                eliminate_duplicates=self.eliminate_duplicates,
                n_max_iterations=100,
            )
        self.mating = mating

        # other run specific data updated whenever solve is called - to share them in all algorithms
        self.n_gen = None
        self.pop = None
        self.off = None

        sd = SharedData()

        crystal_elements = sd.get_ce()
        self.crystal_elements = crystal_elements

        elem_count = sd.get_ec()
        self.elem_count = elem_count

        wyckoffs_dict = sd.get_wd()
        self.wyckoffs_dict = wyckoffs_dict

        max_wyckoffs_count = sd.get_mw()
        self.max_wyckoffs_count = max_wyckoffs_count

        # print(self.crystal_elements, self.elem_count)

    def encode_structure(self, crystal, sg):
        X = []
        s = crystal.to_pymatgen()
        # print(s)
        a, b, c = s.lattice.abc
        alpha, beta, gamma = s.lattice.angles
        # print(a, b, c, alpha, beta, gamma)
        X.extend([a, b, c, alpha, beta, gamma])
        wp_list = self.wyckoffs_dict[sg]

        wck_tuple = ()
        for site in crystal.atom_sites:
            wyck_pos = site.wp
            wyck_pos = str(wyck_pos)
            lines = wyck_pos.split("\n")
            _coords = lines[1:]
            _coords = [[_coord.strip() for _coord in _coords]]
            wck_tuple += (_coords,)
        # print(wck_tuple)

        struc_wp_ind = None
        for wp_ind, _wp in enumerate(wp_list):
            if _wp == wck_tuple:
                struc_wp_ind = wp_ind
                break

        if struc_wp_ind != None:
            struc_wp_val = struc_wp_ind * self.max_wyckoffs_count / len(wp_list)
        else:
            # If no match is found, raise an exception or return a default value
            # print("not found")
            del s
            del crystal
            del wp_list
            del X
            gc.collect()

            raise ValueError(
                "No matching Wyckoff position found for the given atomic positions."
            )
        # print(struc_wp_val)
        X.extend([sg, struc_wp_val])

        for site in s.sites:
            X.extend([site.frac_coords[0], site.frac_coords[1], site.frac_coords[2]])
        # X = np.array(X)

        del s
        del crystal
        del wp_list
        gc.collect()

        return X

    def _initialize_infill(self):
        pop = self.initialization.do(self.problem, self.pop_size, algorithm=self)

        ########### pyxtal
        is_pyxtal = True
        if is_pyxtal == True:
            species = self.crystal_elements
            numIons = self.elem_count

            # Find compatible symmetry groups
            compatible_groups = []
            for group in range(2, 231):
                try:
                    my_crystal = pyxtal()
                    my_crystal.from_random(
                        dim=3, group=group, species=species, numIons=numIons
                    )
                    compatible_groups.append(group)

                    del my_crystal
                    gc.collect()
                except:
                    pass

            print("Compatible symmetry groups from pyxtal:", compatible_groups)
            print(
                f"Total compatible symmetry groups from pyxtal: {len(compatible_groups)}\n"
            )

            stable_crystal_count = 0
            pyxtal_crystal_generation_tries = 5
            for space_group in tqdm(
                compatible_groups,
                desc="Generating inital structures for each space group",
            ):
                # print(f"sg: {space_group}.............")
                attempt = 0
                success = False
                while attempt < pyxtal_crystal_generation_tries and not success:
                    try:
                        # random crystal 1 with specific space group
                        crystal_1 = pyxtal()
                        crystal_1.from_random(
                            dim=3, group=space_group, species=species, numIons=numIons
                        )
                        enoded_vec = self.encode_structure(crystal_1, space_group)
                        # print(enoded_vec)
                        pop[stable_crystal_count]._X = enoded_vec
                        stable_crystal_count += 1
                        success = True

                        del crystal_1
                        del enoded_vec
                        gc.collect()
                    except:
                        attempt += 1

            for space_group in tqdm(
                compatible_groups,
                desc="Generating inital structures for each space group",
            ):
                if stable_crystal_count < self.pop_size:
                    # print(f"sg: {space_group}.............")
                    attempt = 0
                    success = False
                    while attempt < pyxtal_crystal_generation_tries and not success:
                        try:
                            # random crystal 2 with specific space group
                            crystal_2 = pyxtal()
                            crystal_2.from_random(
                                dim=3,
                                group=space_group,
                                species=species,
                                numIons=numIons,
                            )
                            enoded_vec = self.encode_structure(crystal_2, space_group)
                            # print(enoded_vec)
                            pop[stable_crystal_count]._X = enoded_vec
                            stable_crystal_count += 1
                            success = True

                            del crystal_2
                            del enoded_vec
                            gc.collect()
                            # break
                        except:
                            attempt += 1

            print(
                f"Initial total pyxtal genetated structures: {stable_crystal_count}\n"
            )

        # print(
        #     "intializing infil by filling pop using the do method of core.Initialization class *****************"
        # )

        return pop

    def _initialize_advance(self, infills=None, **kwargs):
        # print(len(infills), "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        if self.advance_after_initial_infill:
            self.pop = self.survival.do(
                self.problem, infills, n_survive=len(infills), algorithm=self, **kwargs
            )
        # print(len(self.pop), "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")

    def _infill(self):

        # do the mating using the current population
        # print(len(self.pop), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        off = self.mating.do(self.problem, self.pop, self.n_offsprings, algorithm=self)
        # print(type(off), "666666666666666666666666666666666")
        # print(type(off[0]), "666666666666666666666666666666666")
        # print(off[0])
        # print(f'offspring shape after mating (crossover and mutation): {off.shape}')
        # for offspring in off:
        #     print(offspring.F) ### empty now

        # if the mating could not generate any new offspring (duplicate elimination might make that happen)
        if len(off) == 0:
            self.termination.force_termination = True
            return

        # if not the desired number of offspring could be created
        elif len(off) < self.n_offsprings:
            if self.verbose:
                print(
                    "WARNING: Mating could not produce the required number of (unique) offsprings!"
                )

        # print(len(off), '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
        return off

    def _advance(self, infills=None, **kwargs):

        # the current population
        # print(len(self.pop), "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        pop = self.pop

        # merge the offsprings with the current population
        if infills is not None:
            pop = Population.merge(self.pop, infills)
            # print(len(pop), '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$')

        # execute the survival to find the fittest solutions
        self.pop = self.survival.do(
            self.problem, pop, n_survive=self.pop_size, algorithm=self, **kwargs
        )
        # print(len(self.pop), '@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
