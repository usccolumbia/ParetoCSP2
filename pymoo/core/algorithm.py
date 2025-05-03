import os
import sys
import copy
import time
import warnings
import gc

# gc.set_debug(gc.DEBUG_LEAK)
# from memory_profiler import profile

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from math import gcd
from functools import reduce

from pymatgen.io.cif import CifParser, CifWriter
from tqdm import tqdm

from pymatgen.core.structure import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
from m3gnet.models import Relaxer

# from chgnet.model import StructOptimizer

relaxer = Relaxer()
# relaxer = StructOptimizer()

# logging.getLogger("m3gnet").setLevel(logging.ERROR)
# logging.getLogger("chgnet").setLevel(logging.ERROR)

import contextlib
from pyxtal import pyxtal

from pymoo.core.callback import Callback
from pymoo.core.evaluator import Evaluator
from pymoo.core.population import Population
from pymoo.core.result import Result
from pymoo.termination.default import (
    DefaultMultiObjectiveTermination,
    DefaultSingleObjectiveTermination,
)
from pymoo.util.display.display import Display
from pymoo.util.function_loader import FunctionLoader
from pymoo.util.misc import termination_from_tuple
from pymoo.util.optimum import filter_optimum

from utils.compound_utils import elements_info
from utils.shared_data import SharedData


class Algorithm:

    def __init__(
        self,
        termination=None,
        output=None,
        display=None,
        callback=None,
        archive=None,
        return_least_infeasible=False,
        save_history=False,
        verbose=False,
        seed=None,
        evaluator=None,
        **kwargs,
    ):

        super().__init__()

        # prints the compile warning if enabled
        FunctionLoader.get_instance()

        # the problem to be solved (will be set later on)
        self.problem = None

        # the termination criterion to be used by the algorithm - might be specific for an algorithm
        self.termination = termination

        # the text that should be printed during the algorithm run
        self.output = output

        # an archive kept during algorithm execution (not always the same as optimum)
        self.archive = archive

        # the form of display shown during algorithm execution
        self.display = display

        # callback to be executed each generation
        if callback is None:
            callback = Callback()
        self.callback = callback

        # whether the algorithm should finally return the least infeasible solution if no feasible found
        self.return_least_infeasible = return_least_infeasible

        # whether the history should be saved or not
        self.save_history = save_history

        # whether the algorithm should print output in this run or not
        self.verbose = verbose

        # the random seed that was used
        self.seed = seed

        # an algorithm can defined the default termination which can be overwritten
        self.termination = termination

        # the function evaluator object (can be used to inject code)
        if evaluator is None:
            evaluator = Evaluator()
        self.evaluator = evaluator

        # the history object which contains the list
        self.history = list()

        # the current solutions stored - here considered as population
        self.pop = None

        # a placeholder object for implementation to store solutions in each iteration
        self.off = None

        # the optimum found by the algorithm
        self.opt = None

        # the current number of generation or iteration
        self.n_iter = None

        # can be used to store additional data in submodules
        self.data = {}

        # if the initialized method has been called before or not
        self.is_initialized = False

        # the time when the algorithm has been setup for the first time
        self.start_time = None

        sd = SharedData()
        spaceGroupList = sd.get_sg()
        self.spaceGroupList = spaceGroupList
        # print(spaceGroupList, "eeeee")

        minEnergyAfterEachGen = sd.get_meaeg()
        self.minEnergyAfterEachGen = minEnergyAfterEachGen

        optStrucSpaceGroup = sd.get_ossg()
        self.optStrucSpaceGroup = optStrucSpaceGroup

        crystal_elements = sd.get_ce()
        self.crystal_elements = crystal_elements

        elem_count = sd.get_ec()
        self.elem_count = elem_count

        wyckoffs_dict = sd.get_wd()
        self.wyckoffs_dict = wyckoffs_dict

        max_wyckoffs_count = sd.get_mw()
        self.max_wyckoffs_count = max_wyckoffs_count

        # self.relaxer = Relaxer()

        self.elements_info = elements_info

        self.composition = "".join(
            f"{el}{cnt}" for el, cnt in zip(self.crystal_elements, self.elem_count)
        )
        common_divisor = reduce(gcd, self.elem_count)
        reduced_counts = [cnt // common_divisor for cnt in self.elem_count]
        self.formula = "".join(
            f"{el}{'' if cnt == 1 else cnt}"
            for el, cnt in zip(self.crystal_elements, reduced_counts)
        )

    def write_to_file(self, x, y, z, e, mostSg, optSg, total_time):
        file_name = "info/"
        for _i in range(len(self.crystal_elements)):
            file_name += self.crystal_elements[_i] + str(self.elem_count[_i])
        file_name += ".txt"

        # If this is the first iteration, initialize the file (overwrite)
        if self.n_iter == 1:
            with open(file_name, "w") as file:
                file.write("Space group info after each generation:\n")

        # Append the allSgCount dictionary to the file
        with open(file_name, "a") as file:
            file.write(f"Generation {self.n_iter}: {x}\n")
            file.write(f"Distinct space groups: {y}\n")
            file.write(f"Max space group count: {z}\n")
            file.write(f"Minimum energy after generation {self.n_iter}: {e}\n")
            file.write(
                f"Space group with most structures after generation {self.n_iter}: {', '.join(map(str, mostSg))}\n"
            )
            file.write(
                f"Optimal structure's space group after generation {self.n_iter}: {optSg}\n"
            )
            file.write(
                f"Generation {self.n_iter} - Time Taken: {total_time} seconds\n\n"
            )

    def decode_relax(self, enc_struc):
        # print("inside decode_relax")
        a, b, c, alpha, beta, gamma = (
            enc_struc[0],
            enc_struc[1],
            enc_struc[2],
            enc_struc[3],
            enc_struc[4],
            enc_struc[5],
        )
        sg, wy = int(enc_struc[6]), enc_struc[7]
        all_atoms = [
            element
            for element, count in zip(self.crystal_elements, self.elem_count)
            for _ in range(count)
        ]
        # print(all_atoms)
        wp_list = self.wyckoffs_dict[sg]
        wp = wp_list[int(wy * len(wp_list) / self.max_wyckoffs_count)]

        atom_dict = {}
        for i in range(int((len(enc_struc) - 8) / 3)):
            atom_dict["x" + str(i + 1)] = enc_struc[6 + i * 3 + 0]
            atom_dict["y" + str(i + 1)] = enc_struc[6 + i * 3 + 1]
            atom_dict["z" + str(i + 1)] = enc_struc[6 + i * 3 + 2]

        atoms = []
        atom_positions = []
        count = 0
        for i, wp_i in enumerate(wp):
            for wp_i_j in wp_i:
                atoms += [self.crystal_elements[i]] * len(wp_i_j)

                for wp_i_j_k in wp_i_j:
                    count += 1
                    if "x" in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace(
                            "x", str(atom_dict["x" + str(count)])
                        )
                    if "y" in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace(
                            "y", str(atom_dict["y" + str(count)])
                        )
                    if "z" in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace(
                            "z", str(atom_dict["z" + str(count)])
                        )
                    atom_positions.append(list(eval(wp_i_j_k)))
        # print(atom_positions)

        if sg in [0, 1, 2]:
            lattice = Lattice.from_parameters(
                a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
            )
        elif sg in list(range(3, 15 + 1)):
            lattice = Lattice.from_parameters(
                a=a, b=b, c=c, alpha=90, beta=beta, gamma=90
            )
        elif sg in list(range(16, 74 + 1)):
            lattice = Lattice.from_parameters(
                a=a, b=b, c=c, alpha=90, beta=90, gamma=90
            )
        elif sg in list(range(75, 142 + 1)):
            lattice = Lattice.from_parameters(
                a=a, b=a, c=c, alpha=90, beta=90, gamma=90
            )
        elif sg in list(range(143, 194 + 1)):
            lattice = Lattice.from_parameters(
                a=a, b=a, c=c, alpha=90, beta=90, gamma=120
            )
        elif sg in list(range(195, 230 + 1)):
            lattice = Lattice.from_parameters(
                a=a, b=a, c=a, alpha=90, beta=90, gamma=90
            )
        else:
            lattice = Lattice.from_parameters(
                a=a, b=b, c=c, alpha=alpha, beta=beta, gamma=gamma
            )

        structure = Structure(lattice, all_atoms, atom_positions)

        isValid = True
        isValid = self.atomic_dist_and_volume_limit(structure, all_atoms)
        for site in structure.frac_coords:
            if np.any(site < -1) or np.any(site > 1):
                # print(f"Invalid fractional coordinates detected:")
                isValid = False
                break

        if not isValid:
            del structure
            del atom_positions
            del atoms
            del atom_dict
            del wp_list
            del all_atoms
            del enc_struc
            del lattice
            gc.collect()
            raise Exception("Structure not valid !!!")

        # relaxer = Relaxer()
        relaxed_struc_obj = None
        try:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                    devnull
                ):
                    relaxed_struc_obj = relaxer.relax(structure, steps=10)
        except Exception as e:
            del relaxed_struc_obj
            del structure
            del atom_positions
            del atoms
            del atom_dict
            del wp_list
            del all_atoms
            del enc_struc
            del lattice
            gc.collect()
            raise Exception("error")

        relaxed_struc = relaxed_struc_obj["final_structure"]
        # relaxed_energy = relaxed_struc_obj["trajectory"].energies[-1] / len(
        #     relaxed_struc
        # )
        # print(relaxed_energy)

        symmetry_analyzer = SpacegroupAnalyzer(relaxed_struc, symprec=0.1)
        sg = symmetry_analyzer.get_space_group_number()
        # print("done space group")

        del relaxed_struc_obj
        del structure
        del atom_positions
        del atoms
        del atom_dict
        del wp_list
        del symmetry_analyzer
        del all_atoms
        del enc_struc
        del lattice
        gc.collect()

        return relaxed_struc, sg

    def encode(self, struc, sg):
        # print("inside encode")
        X = []
        a, b, c = struc.lattice.abc
        alpha, beta, gamma = struc.lattice.angles
        # print(a, b, c, alpha, beta, gamma)
        X.extend([a, b, c, alpha, beta, gamma])

        wp_list = self.wyckoffs_dict[sg]

        pyxtal_structure = pyxtal()
        pyxtal_structure._from_pymatgen(struc)

        wck_tuple = ()
        for site in pyxtal_structure.atom_sites:
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
                # print(
                #     "found !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
                # )
                break

        if struc_wp_ind != None:
            struc_wp_val = struc_wp_ind * self.max_wyckoffs_count / len(wp_list)
        else:
            # If no match is found, raise an exception or return a default value
            # print("not found")
            del struc
            del pyxtal_structure
            del wp_list
            del X
            gc.collect()

            raise ValueError(
                "No matching Wyckoff position found for the given atomic positions."
            )
        # print(struc_wp_val)
        X.extend([sg, struc_wp_val])
        # print(type(X))

        for site in struc.sites:
            X.extend([site.frac_coords[0], site.frac_coords[1], site.frac_coords[2]])

        del struc
        del pyxtal_structure
        del wp_list
        gc.collect()

        return X

    def atomic_dist_and_volume_limit(self, struc, all_atoms):
        atom_radii = []
        for i in all_atoms:
            if self.elements_info[i][8] == -1:
                atom_radii.append(100.0 / 100.0)
            else:
                atom_radii.append(float(self.elements_info[i][8]) / 100.0)

        for i in range(sum(self.elem_count) - 1):
            for j in range(i + 1, sum(self.elem_count)):
                if struc.get_distance(i, j) < (atom_radii[i] + atom_radii[j]) * 0.4:
                    # print("invlaid dist!!!")
                    del struc
                    del all_atoms
                    del atom_radii
                    gc.collect()
                    return False

        atom_volume = [4.0 * np.pi * r**3 / 3.0 for r in atom_radii]
        sum_atom_volume = sum(atom_volume) / 0.55
        if not (sum_atom_volume * 0.4 <= struc.volume <= sum_atom_volume * 2.4):
            # print("invlaid volume!!!")
            del struc
            del all_atoms
            del atom_radii
            del atom_volume
            gc.collect()
            return False

        # print("ok 1")
        isVaccumLimitCorrect = self.vacuum_size_limit(struc.copy(), max_size=7.0)

        del struc
        del all_atoms
        del atom_radii
        del atom_volume
        gc.collect()

        if isVaccumLimitCorrect:
            # print("ok 2")
            return True
        else:
            # print("invlaid vaccum limit!!!")
            return False

    def vacuum_size_limit(self, struc, max_size: float = 10.0):
        def get_foot(p, a, b):
            p = np.array(p)
            a = np.array(a)
            b = np.array(b)
            ap = p - a
            ab = b - a
            result = a + np.dot(ap, ab) / np.dot(ab, ab) * ab
            return result

        def get_distance(a, b):
            return np.sqrt(np.sum(np.square(b - a)))

        isValid = True
        struc.make_supercell([2, 2, 2], to_unit_cell=False)
        line_a_points = [
            [0, 0, 0],
        ]
        line_b_points = [
            [0, 0, 1],
            [0, 1, 0],
            [1, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, -1],
            [1, 0, -1],
            [1, -1, 0],
            [1, 1, 1],
            [1, 1, -1],
            [1, -1, 1],
            [-1, 1, 1],
        ]
        for a in line_a_points:
            for b in line_b_points:
                foot_points = []
                for p in struc.frac_coords:
                    f_p = get_foot(p, a, b)
                    foot_points.append(f_p)
                foot_points = sorted(foot_points, key=lambda x: [x[0], x[1], x[2]])

                foot_points = np.asarray(
                    np.mat(foot_points) * np.mat(struc.lattice.matrix)
                )
                for fp_i in range(0, len(foot_points) - 1):
                    fp_distance = get_distance(foot_points[fp_i + 1], foot_points[fp_i])
                    if fp_distance > max_size:
                        # print(fp_distance)
                        isValid = False
                        break
        del struc
        del foot_points
        gc.collect()
        return isValid

    def setup(self, problem, **kwargs):

        # the problem to be solved by the algorithm
        self.problem = problem

        # set all the provided options to this method
        for key, value in kwargs.items():
            self.__dict__[key] = value

        # if seed is a boolean and true, then randomly set a seed (useful to reproduce runs)
        seed = self.seed
        if isinstance(seed, bool) and seed:
            seed = np.random.randint(0, 10000000)
            self.seed = seed

        # if a seed is set, then use it to call the random number generators
        if seed is not None:
            import random

            random.seed(seed)
            np.random.seed(seed)

        # make sure that some type of termination criterion is set
        if self.termination is None:
            self.termination = default_termination(problem)
        else:
            self.termination = termination_from_tuple(self.termination)

        # set up the display during the algorithm execution
        if self.display is None:
            verbose = kwargs.get("verbose", False)
            progress = kwargs.get("progress", False)
            self.display = Display(self.output, verbose=verbose, progress=progress)

        # finally call the function that can be overwritten by the actual algorithm
        self._setup(problem, **kwargs)

        return self

    def run(self):
        while self.has_next():
            self.next()
        return self.result()

    def has_next(self):
        return not self.termination.has_terminated()

    def finalize(self):

        # finalize the display output in the end of the run
        self.display.finalize()

        return self._finalize()

    # @profile
    def next(self):
        # print('entering next ....................')

        # get the infill solutions
        infills = self.infill()
        # print('################# 1')
        # for pop in infills:
        #     print(pop.age)
        # print('################# 2')

        # call the advance with them after evaluation
        allSgCount = {key: 0 for key in range(1, 231)}
        if infills is not None:
            start_time = time.time()

            ### new batching
            batch_size = 32
            num_batches = len(infills) // batch_size + (len(infills) % batch_size > 0)

            for batch_index in range(num_batches):
                start = batch_index * batch_size
                end = start + batch_size
                batch_infills = infills[start:end]

                for i in range(len(batch_infills)):
                    relaxed_struc = None
                    space_group = None

                    try:
                        relaxed_struc, space_group = self.decode_relax(
                            batch_infills[i]._X
                        )
                    except Exception as e:
                        pass  # Handle exceptions as needed

                    # print(space_group, "999999999999")

                    # Update infill with relaxed structure
                    if space_group:
                        try:
                            infills[start + i]._X = self.encode(
                                relaxed_struc, space_group
                            )
                        except:
                            pass

                    del relaxed_struc
                    gc.collect()

                # # After processing each batch, release memory
                del batch_infills
                gc.collect()

            ### new batching

            self.evaluator.eval(self.problem, infills, algorithm=self)

            sd = SharedData()

            for i, p in enumerate(tqdm(infills, desc="Adjusting genotypic ages")):
                # print(int(p._X[6]), "99999999999999999999999999999999")
                # print(p.F.shape,'oooooooooooooooooooooooooooooo')
                # print(infills[i].F, '####################### 3')
                # print(type(p))
                # print(len(p))
                infills[i].F[-2] = p.age
                # print(p.sgCount)
                # infills[i].F[-1] = p.sgCount
                # print(infills[i].F, '####################### 4')
                if infills[i].F[0] < self.minEnergyAfterEachGen:
                    # print("New best structure found !!!")
                    self.minEnergyAfterEachGen = infills[i].F[0]
                    self.optStrucSpaceGroup = int(infills[i]._X[6])

            for i, p in enumerate(tqdm(infills, desc="Counting space groups")):
                sg = int(p._X[6])
                allSgCount[sg] += 1

            for i, p in enumerate(tqdm(infills, desc="Adjusting space group counts")):
                sg = int(p._X[6])
                p.sgCount = allSgCount[sg]
                infills[i].F[-1] = p.sgCount

            print(f"Shared spce group count map: {allSgCount}")

            nonZeroSpaceGroups = len({k for k, v in allSgCount.items() if v != 0})
            print(f"Distinct space groups: {nonZeroSpaceGroups}")
            maxSpaceGroupCount = max(allSgCount.values())
            print(f"Max space group count: {maxSpaceGroupCount}")

            sd.set_meaeg(self.minEnergyAfterEachGen)
            sd.set_ossg(self.optStrucSpaceGroup)
            print(
                f"Minimum energy after generation {self.n_iter}: {self.minEnergyAfterEachGen}"
            )

            # Find the maximum value in the dictionary
            max_value = max(allSgCount.values())

            # Get all keys with that maximum value
            sgWithMostStruc = [
                key for key, value in allSgCount.items() if value == max_value
            ]

            print(
                f"Space group with most structures after generation {self.n_iter}: {sgWithMostStruc}"
            )
            print(
                f"Optimal structure's space group after generation {self.n_iter}: {self.optStrucSpaceGroup}"
            )

            end_time = time.time()
            total_time = end_time - start_time

            print(f"Generation {self.n_iter} - Time Taken: {total_time} seconds\n")

            self.write_to_file(
                allSgCount,
                nonZeroSpaceGroups,
                maxSpaceGroupCount,
                self.minEnergyAfterEachGen,
                sgWithMostStruc,
                self.optStrucSpaceGroup,
                total_time,
            )

            self.advance(infills=infills)

        # if the algorithm does not follow the infill-advance scheme just call advance
        else:
            self.advance()
        # print('exiting next ....................\n')

    def _initialize(self):

        # the time starts whenever this method is called
        self.start_time = time.time()

        # set the attribute for the optimization method to start
        self.n_iter = 1
        self.pop = Population.empty()
        self.opt = None

    def infill(self):
        # print('entering infill ............................')
        if self.problem is None:
            raise Exception("Please call `setup(problem)` before calling next().")

        # the first time next is called simply initial the algorithm - makes the interface cleaner
        if not self.is_initialized:
            # print('infill not initialized, initializing ********************')

            # hook mostly used by the class to happen before even to initialize
            self._initialize()

            # execute the initialization infill of the algorithm
            infills = self._initialize_infill()
            # print(type(infills))
            # print(len(infills))
            # print(type(infills[0]))

        else:
            # request the infill solutions if the algorithm has implemented it
            infills = self._infill()
            # print(infills.shape, 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

        # set the current generation to the offsprings
        if infills is not None:
            infills.set("n_gen", self.n_iter)
            infills.set("n_iter", self.n_iter)

        # print('exiting infill ............................\n')
        return infills

    def advance(self, infills=None, **kwargs):
        # print('enter advance ..........')

        # if infills have been provided set them as offsprings and feed them into advance
        self.off = infills

        # if the algorithm has not been already initialized
        if not self.is_initialized:

            # set the generation counter to 1
            self.n_iter = 1

            # assign the population to the algorithm
            self.pop = infills
            # print(self.pop, '************')
            # print(self.pop[0].F)

            # do what is necessary after the initialization
            self._initialize_advance(infills=infills, **kwargs)

            # set this algorithm to be initialized
            self.is_initialized = True

            # always advance to the next iteration after initialization
            self._post_advance()

        else:

            # call the implementation of the advance method - if the infill is not None
            val = self._advance(infills=infills, **kwargs)

            # always advance to the next iteration - except if the algorithm returns False
            if val is None or val:
                self._post_advance()

        # if the algorithm has terminated, then do the finalization steps and return the result
        if self.termination.has_terminated():
            self.finalize()
            ret = self.result()

        # otherwise just increase the iteration counter for the next step and return the current optimum
        else:
            ret = self.opt

        # add the infill solutions to an archive
        if self.archive is not None and infills is not None:
            self.archive = self.archive.add(infills)

        return ret

    def result(self):
        res = Result()

        # store the time when the algorithm as finished
        res.start_time = self.start_time
        res.end_time = time.time()
        res.exec_time = res.end_time - res.start_time

        res.pop = self.pop
        res.archive = self.archive

        # get the optimal solution found
        opt = self.opt
        if opt is None or len(opt) == 0:
            opt = None

        # if no feasible solution has been found
        elif not np.any(opt.get("feasible")):
            if self.return_least_infeasible:
                opt = filter_optimum(opt, least_infeasible=True)
            else:
                opt = None
        res.opt = opt

        # if optimum is set to none to not report anything
        if res.opt is None:
            X, F, CV, G, H = None, None, None, None, None

        # otherwise get the values from the population
        else:
            X, F, CV, G, H = self.opt.get("X", "F", "CV", "G", "H")

            # if single-objective problem and only one solution was found - create a 1d array
            if self.problem.n_obj == 1 and len(X) == 1:
                X, F, CV, G, H = X[0], F[0], CV[0], G[0], H[0]

        # set all the individual values
        res.X, res.F, res.CV, res.G, res.H = X, F, CV, G, H

        # create the result object
        res.problem = self.problem
        res.history = self.history

        return res

    def ask(self):
        return self.infill()

    def tell(self, *args, **kwargs):
        return self.advance(*args, **kwargs)

    def _set_optimum(self):
        self.opt = filter_optimum(self.pop, least_infeasible=True)

    def _post_advance(self):

        # update the current optimum of the algorithm
        self._set_optimum()

        # update the current termination condition of the algorithm
        self.termination.update(self)

        # display the output if defined by the algorithm
        self.display(self)

        # if a callback function is provided it is called after each iteration
        self.callback(self)

        if self.save_history:
            _hist, _callback, _display = self.history, self.callback, self.display

            self.history, self.callback, self.display = None, None, None
            obj = copy.deepcopy(self)

            self.history, self.callback, self.display = _hist, _callback, _display
            self.history.append(obj)

        self.n_iter += 1

    # =========================================================================================================
    # TO BE OVERWRITTEN
    # =========================================================================================================

    def _setup(self, problem, **kwargs):
        pass

    def _initialize_infill(self):
        pass

    def _initialize_advance(self, infills=None, **kwargs):
        pass

    def _infill(self):
        pass

    def _advance(self, infills=None, **kwargs):
        pass

    def _finalize(self):
        pass

    # =========================================================================================================
    # CONVENIENCE
    # =========================================================================================================

    @property
    def n_gen(self):
        return self.n_iter

    @n_gen.setter
    def n_gen(self, value):
        self.n_iter = value


class LoopwiseAlgorithm(Algorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator = None
        self.state = None

    def _next(self):
        pass

    def _infill(self):
        if self.state is None:
            self._advance()
        return self.state

    def _advance(self, infills=None, **kwargs):
        if self.generator is None:
            self.generator = self._next()
        try:
            self.state = self.generator.send(infills)
        except StopIteration:
            self.generator = None
            self.state = None
            return True

        return False


def default_termination(problem):
    if problem.n_obj > 1:
        termination = DefaultMultiObjectiveTermination()
    else:
        termination = DefaultSingleObjectiveTermination()
    return termination
