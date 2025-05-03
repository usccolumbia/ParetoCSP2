import os
import time
import warnings
import sys
import traceback
import gc

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
from argparse import ArgumentParser

from tqdm import tqdm

from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.core.structure import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import tensorflow as tf
import logging

tf.get_logger().setLevel(logging.ERROR)
from m3gnet.models import M3GNet, Relaxer

# from chgnet.model.model import CHGNet
# from chgnet.model import StructOptimizer

m3gnet_energy = M3GNet.load()
relaxer = Relaxer()

# chgnet_energy = CHGNet.load()
# relaxer = StructOptimizer()

# logging.getLogger("m3gnet").setLevel(logging.ERROR)
# logging.getLogger("chgnet").setLevel(logging.ERROR)

import contextlib

from utils.file_utils import check_and_rename_path
from utils.read_input import ReadInput
from utils.compound_utils import elements_info
from utils.print_utils import print_header, print_run_info
from utils.wyckoff_position.get_wyckoff_position import get_all_wyckoff_combination
from utils.shared_data import SharedData


class PredictStructure:
    @print_header
    def __init__(self, input_file_path="config.in"):

        ### command line arguments
        parser = ArgumentParser()
        parser.add_argument(
            "--path", type=str, default=None, help="path to config (.in) file"
        )
        parser.add_argument(
            "--comp", type=str, default=None, help="composition of the compound"
        )
        parser.add_argument(
            "--energy_model",
            type=str,
            default="M3GNet",
            help="Energy model (M3GNet or CHGNet)",
        )
        parser.add_argument(
            "--alg", type=str, default=None, help="name of the algorithm"
        )
        parser.add_argument("--pop", type=int, default=None, help="population size")
        parser.add_argument(
            "--num_track",
            type=int,
            default=None,
            help="number of distinct space group structures to track",
        )
        parser.add_argument(
            "--num_copy",
            type=int,
            default=None,
            help="number of structures to track from same space group, usually 1 for regular CSP, 2-5 for polymorphism CSP",
        )
        parser.add_argument(
            "--mode",
            type=str,
            default=None,
            help="experiment mode: csp (crystal structure prediction) or poly (polymorphic structure prediction)",
        )
        parser.add_argument(
            "--max_step", type=int, default=None, help="maximum number of steps"
        )
        parser.add_argument("--seed", type=int, default=None, help="seed")
        parser.add_argument("--gpu", action="store_false")

        args = parser.parse_args()
        if args.path != None:
            input_file_path = args.path

        self.input_config = ReadInput(input_file_path=input_file_path, args=args)

        if not self.input_config.is_use_gpu:
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

        print(f"Using {self.input_config.energy_model} as the energy model !!!\n")
        if self.input_config.energy_model == "M3GNet":
            # self.energy_model = M3GNet.load()
            # self.relaxer = Relaxer()
            None
        elif self.input_config.energy_model == "CHGNet":
            # self.energy_model = CHGNet.load()
            # self.relaxer = StructOptimizer()
            None
        else:
            print("Energy model has to be either M3GNet or CHGNet ! Exiting !!!")
            exit()

        self.compound = self.input_config.compound
        self.elements = self.input_config.elements
        self.elements_count = self.input_config.elements_count
        self.space_group = list(
            range(
                self.input_config.space_group[0], self.input_config.space_group[1] + 1
            )
        )

        sd = SharedData()
        sd.set_sg(self.space_group)
        sd.set_ce(self.elements)
        sd.set_ec(self.elements_count)

        self.wyckoffs_dict, self.max_wyckoffs_count = get_all_wyckoff_combination(
            self.space_group, self.elements_count
        )
        sd.set_wd(self.wyckoffs_dict)
        sd.set_mw(self.max_wyckoffs_count)
        self.total_atom_count = self.input_config.total_atom_count
        # self.total_atom_count = sum(self.elements_count)

        if self.input_config.algorithm == "tpe":
            self.output_path = os.path.join(
                self.input_config.output_path,
                self.compound + "_" + self.input_config.algorithm,
            )
        else:
            self.output_path = os.path.join(
                self.input_config.output_path,
                self.compound
                + "_"
                + self.input_config.algorithm
                + "_pop"
                + str(self.input_config.pop),
            )
        check_and_rename_path(self.output_path)
        self.structures_path = os.path.join(self.output_path, "structures")
        check_and_rename_path(self.structures_path)

        self.is_ga = None
        self.elements_info = elements_info

        self.step_number = 0
        self.structure_number = 0
        self.all_atoms = []
        self.minEnergy = 999999999.0
        self.optSpaceGroup = 0
        self.bestCif = ""
        self.mode = self.input_config.mode
        self.k = self.input_config.num_track
        self.j = self.input_config.num_copy
        self.top_k_structures = []
        self.start_time = time.time()

        self.find_stable_structure()

    def predict_structure_energy(self, kwargs):
        self.step_number += 1

        if self.is_ga:
            _dict = {
                "a": kwargs[0],
                "b": kwargs[1],
                "c": kwargs[2],
                "alpha": kwargs[3],
                "beta": kwargs[4],
                "gamma": kwargs[5],
                "sg": int(kwargs[6]),
                "wp": kwargs[7],
            }
            for i in range(int((len(kwargs) - 8) / 3)):
                _dict["x" + str(i + 1)] = kwargs[6 + i * 3 + 0]
                _dict["y" + str(i + 1)] = kwargs[6 + i * 3 + 1]
                _dict["z" + str(i + 1)] = kwargs[6 + i * 3 + 2]
        else:
            _dict = kwargs

        try:
            tmp_structure_file_name = os.path.join(self.structures_path, "temp.cif")
            self.save_structure_file(
                self.all_atoms, _dict, file_name=tmp_structure_file_name
            )
            struc = Structure.from_file(tmp_structure_file_name)
            # print(struc)

            isValid = True
            isValid = self.atomic_dist_and_volume_limit(struc)

            if not isValid:
                raise Exception()

            energy_predict = m3gnet_energy.predict_structure(struc)
            result = energy_predict.numpy()[0][0] / len(struc)

            # energy_predict = chgnet_energy.predict_structure(struc)
            # result = np.float64(energy_predict["e"]) / len(struc)

            self.structure_number += 1
            with open(os.path.join(self.output_path, "energy_data.csv"), "a+") as f:
                f.write(
                    ",".join(
                        [
                            str(self.structure_number),
                            str(self.step_number),
                            str(result),
                            str(_dict["sg"]),
                            str(_dict["wp"]),
                            str(time.time() - self.start_time),
                        ]
                    )
                    + "\n"
                )

            structure_file_name = os.path.join(
                self.structures_path,
                "%s_%d_%f_%d_%d.cif"
                % (
                    self.compound,
                    self.total_atom_count,
                    result,
                    self.structure_number,
                    self.step_number,
                ),
            )

            os.rename(tmp_structure_file_name, structure_file_name)

            if result < self.minEnergy:
                self.minEnergy = result
                self.bestCif = structure_file_name

            if self.mode == "csp":
                self.update_top_k_structures_csp(struc, result, _dict["sg"])
            elif self.mode == "poly":
                self.update_top_k_structures_poly(struc, result, _dict["sg"])

            del struc
            del energy_predict
            gc.collect()

        except Exception as e:
            # print(f"exception: {e}")
            # exc_type, exc_obj, exc_tb = sys.exc_info()
            # print(exc_type, exc_tb.tb_lineno)
            # print(traceback.format_exc())
            result = 999

        if self.is_ga:
            return result
        else:
            None

    @print_run_info("Predict crystal structure")
    def find_stable_structure(self):
        with open(os.path.join(self.output_path, "energy_data.csv"), "w+") as f:
            f.writelines("number,step,energy,sg_number,wp_number,time\n")

        if self.input_config.algorithm == "paretocsp2":
            self.find_stable_structure_by_paretocsp2()
        else:
            print("wrong algorithm name !!!")
            sys.exit(0)

    def find_stable_structure_by_paretocsp2(self):
        from pymoo.config import Config

        Config.warnings["not_compiled"] = False
        from pymoo.algorithms.moo.nsga3 import NSGA3
        from pymoo.optimize import minimize
        from pymoo.util.ref_dirs import get_reference_directions
        from pymoo.core.problem import ElementwiseProblem

        self.is_ga = True

        a = self.input_config.lattice_a
        b = self.input_config.lattice_b
        c = self.input_config.lattice_c
        alpha = self.input_config.lattice_alpha
        beta = self.input_config.lattice_beta
        gamma = self.input_config.lattice_gamma

        lb = [a[0], b[0], c[0], alpha[0], beta[0], gamma[0], self.space_group[0], 0]
        ub = [
            a[1],
            b[1],
            c[1],
            alpha[1],
            beta[1],
            gamma[1],
            len(self.space_group),
            self.max_wyckoffs_count,
        ]

        compound_times = self.total_atom_count / sum(self.elements_count)
        compound_times = int(compound_times)

        for j, a_j in enumerate(self.elements):
            for c_k in range(compound_times * self.elements_count[j]):
                self.all_atoms.append(a_j)
                lb += [0, 0, 0]
                ub += [1, 1, 1]

        max_step = self.input_config.max_step
        pop = self.input_config.pop
        rand_seed = self.input_config.rand_seed
        if rand_seed != -1:
            np.random.seed(rand_seed)

        energy_func = self.predict_structure_energy

        class NSGA3_AFPO(ElementwiseProblem):

            def __init__(self):
                super().__init__(
                    n_var=len(lb),
                    n_obj=3,
                    n_ieq_constr=0,
                    xl=np.array(lb),
                    xu=np.array(ub),
                )

            def _evaluate(self, x, out, *args, **kwargs):
                f1 = energy_func(x)
                f2 = 0.0
                f3 = 0.0

                out["F"] = [f1, f2, f3]

        problem = NSGA3_AFPO()
        ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

        algorithm = NSGA3(pop_size=pop, ref_dirs=ref_dirs)

        res = minimize(
            problem,
            algorithm,
            seed=rand_seed,
            termination=("n_gen", max_step),
            verbose=True,
        )

        print("\nPareto front:\n" + "-" * 13)
        for i in res.F:
            print(i)
        print("\n\n")

        print(
            f"\nOptimal energy structure path: {self.bestCif}, energy: {self.minEnergy}\n"
        )

        # cifParser = CifParser(self.bestCif)
        # struct = cifParser.get_structures()[0]
        struct = Structure.from_file(self.bestCif)
        print(
            f"Relaxing the optimal structure using {self.input_config.energy_model} ..........."
        )
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(
                devnull
            ):
                relaxed_cif = relaxer.relax(struct)
        # print(relaxed_cif['final_structure'])
        print("Relaxing done ...........\n")
        # final_energy = relaxed_cif["trajectory"].energies[-1] / len(relaxed_cif)
        final_energy = relaxed_cif["trajectory"].energies[-1] / len(
            relaxed_cif["final_structure"]
        )

        print("Symmetrizing the relaxed structure ...........")
        spacegroupAnalyzer = SpacegroupAnalyzer(
            relaxed_cif["final_structure"], symprec=0.1
        )
        symmetrized_cif = spacegroupAnalyzer.get_symmetrized_structure()
        print(symmetrized_cif)
        print("Symmetrizing done ...........\n")

        cifWriter = CifWriter(symmetrized_cif, symprec=0.1)
        relaxed_structure_path = os.path.join(
            self.structures_path, self.compound + "_relaxed.cif"
        )
        cifWriter.write_file(relaxed_structure_path)
        print(f"Relaxed structures saved in : {relaxed_structure_path}")
        print(f"Final energy after relaxation: {final_energy}")

        del struct
        del relaxed_cif
        del spacegroupAnalyzer
        del symmetrized_cif
        del cifWriter
        gc.collect()

        if self.mode == "csp":
            self.save_top_k_structures_csp()
        elif self.mode == "poly":
            self.save_top_k_structures_poly()

    def save_structure_file(self, all_atoms, struc_parameters, file_name):
        # keysList = list(struc_parameters.keys())
        # print(keysList)
        sg = struc_parameters["sg"]
        wp_list = self.wyckoffs_dict[sg]
        # print(wp_list)
        wp = wp_list[
            int(struc_parameters["wp"] * len(wp_list) / self.max_wyckoffs_count)
        ]

        atoms = []
        atom_positions = []
        count = 0
        for i, wp_i in enumerate(wp):
            for wp_i_j in wp_i:
                atoms += [self.elements[i]] * len(wp_i_j)

                for wp_i_j_k in wp_i_j:
                    count += 1
                    if "x" in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace(
                            "x", str(struc_parameters["x" + str(count)])
                        )
                    if "y" in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace(
                            "y", str(struc_parameters["y" + str(count)])
                        )
                    if "z" in wp_i_j_k:
                        wp_i_j_k = wp_i_j_k.replace(
                            "z", str(struc_parameters["z" + str(count)])
                        )
                    atom_positions.append(list(eval(wp_i_j_k)))
        # print(atom_positions)

        if sg in [0, 1, 2]:
            lattice = Lattice.from_parameters(
                a=struc_parameters["a"],
                b=struc_parameters["b"],
                c=struc_parameters["c"],
                alpha=struc_parameters["alpha"],
                beta=struc_parameters["beta"],
                gamma=struc_parameters["gamma"],
            )
        elif sg in list(range(3, 15 + 1)):
            lattice = Lattice.from_parameters(
                a=struc_parameters["a"],
                b=struc_parameters["b"],
                c=struc_parameters["c"],
                alpha=90,
                beta=struc_parameters["beta"],
                gamma=90,
            )
        elif sg in list(range(16, 74 + 1)):
            lattice = Lattice.from_parameters(
                a=struc_parameters["a"],
                b=struc_parameters["b"],
                c=struc_parameters["c"],
                alpha=90,
                beta=90,
                gamma=90,
            )
        elif sg in list(range(75, 142 + 1)):
            lattice = Lattice.from_parameters(
                a=struc_parameters["a"],
                b=struc_parameters["a"],
                c=struc_parameters["c"],
                alpha=90,
                beta=90,
                gamma=90,
            )
        elif sg in list(range(143, 194 + 1)):
            lattice = Lattice.from_parameters(
                a=struc_parameters["a"],
                b=struc_parameters["a"],
                c=struc_parameters["c"],
                alpha=90,
                beta=90,
                gamma=120,
            )
        elif sg in list(range(195, 230 + 1)):
            lattice = Lattice.from_parameters(
                a=struc_parameters["a"],
                b=struc_parameters["a"],
                c=struc_parameters["a"],
                alpha=90,
                beta=90,
                gamma=90,
            )
        else:
            lattice = Lattice.from_parameters(
                a=struc_parameters["a"],
                b=struc_parameters["b"],
                c=struc_parameters["c"],
                alpha=struc_parameters["alpha"],
                beta=struc_parameters["beta"],
                gamma=struc_parameters["gamma"],
            )

        structure = Structure(lattice, all_atoms, atom_positions)
        structure.to(fmt="cif", filename=file_name)

        del structure
        del wp_list
        del all_atoms
        del struc_parameters
        del lattice
        del atoms
        del atom_positions
        gc.collect()

    ### csp
    def update_top_k_structures_csp(self, struc, struc_energy, space_group):
        # print("inside update_top_k_structures")
        dir_path = f"results/{self.compound}_paretocsp2_pop{self.input_config.pop}/top_structures"
        # First, check if the space group is already in the top structures
        # Create a directory to store the top structures if it doesn't exist
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        found = False
        for i, (
            existing_cif_path,
            existing_energy_val,
            existing_space_group,
        ) in enumerate(self.top_k_structures):
            if existing_space_group == space_group:
                found = True
                if struc_energy < existing_energy_val:
                    # Generate a unique filename for the structure
                    file_name = f"{self.compound}_structure_sg_{space_group}_energy_{struc_energy:.8f}.cif"
                    file_path = os.path.join(dir_path, file_name)
                    # Save the file
                    cifWriter = CifWriter(struc, symprec=0.1)
                    cifWriter.write_file(file_path)

                    # Replace with the new structure if it has a lower energy
                    self.top_k_structures[i] = (file_path, struc_energy, space_group)
                break  # Space group found, no need to continue looping

        if not found:
            # Generate a unique filename for the structure
            file_name = f"structure_sg_{space_group}_energy_{struc_energy:.8f}.cif"
            file_path = os.path.join(dir_path, file_name)
            # Save the file
            cifWriter = CifWriter(struc, symprec=0.1)
            cifWriter.write_file(file_path)

            # Add the new structure since its space group is not in the list
            self.top_k_structures.append((file_path, struc_energy, space_group))

        # Sort the structures by their energy value in ascending order
        self.top_k_structures.sort(key=lambda x: x[1])

        # Ensure that only up to self.k structures are stored
        if len(self.top_k_structures) > self.k:
            # Identify structures beyond the top k
            excess_structures = self.top_k_structures[self.k :]
            # Delete files associated with excess structures
            for excess_structure in excess_structures:
                file_path = excess_structure[0]
                if os.path.exists(file_path):
                    os.remove(file_path)  # Delete the file from disk
            # Keep only the top k structures in the list
            self.top_k_structures = self.top_k_structures[: self.k]

        del struc
        del cifWriter
        gc.collect()

    def save_top_k_structures_csp(self):
        print("\n\n")
        print(f"Top {self.k} Structures before deep relaxation:")
        for idx, (cif_path, energy_val, space_group) in enumerate(
            self.top_k_structures, start=1
        ):
            print(
                f"{idx}. CIF Path: {cif_path}, Energy: {energy_val}, Space Group: {space_group}"
            )
        print("\n")

        print(f"Top {self.k} Structures after deep relaxation:")
        top_k_structures_list = []
        isDeepRelax = True

        for idx, (file_path, energy_val, space_group) in tqdm(
            enumerate(self.top_k_structures, start=1),
            total=len(self.top_k_structures),
            desc=f"Processing and saving top {self.k} structures",
        ):
            struc = Structure.from_file(file_path)
            if isDeepRelax:
                try:
                    # relaxer = Relaxer()
                    with open(os.devnull, "w") as devnull:
                        with contextlib.redirect_stdout(
                            devnull
                        ), contextlib.redirect_stderr(devnull):
                            relaxed_struc = relaxer.relax(struc)
                    # relaxed_energy = relaxed_struc["trajectory"].energies[-1] / len(
                    #     relaxed_struc
                    # )
                    struc = relaxed_struc["final_structure"]
                except:
                    pass

                energy_predict = m3gnet_energy.predict_structure(struc)
                relaxed_energy = energy_predict.numpy()[0][0] / len(struc)

                # energy_predict = chgnet_energy.predict_structure(struc)
                # relaxed_energy = np.float64(energy_predict["e"]) / len(struc)
                # print(relaxed_energy)

            if isDeepRelax:
                try:
                    spacegroupAnalyzer = SpacegroupAnalyzer(struc, symprec=0.1)
                    struc = spacegroupAnalyzer.get_symmetrized_structure()
                    relaxed_sg = spacegroupAnalyzer.get_space_group_number()
                except:
                    pass

            cifWriter = CifWriter(struc, symprec=0.1)
            relaxed_structure_path = f"results/{self.compound}_paretocsp2_pop{self.input_config.pop}/top_structures/{self.compound}_top_{idx}.cif"
            cifWriter.write_file(relaxed_structure_path)

            if isDeepRelax:
                top_k_structures_list.append(
                    (relaxed_structure_path, relaxed_energy, relaxed_sg)
                )
            else:
                top_k_structures_list.append(
                    (relaxed_structure_path, energy_val, space_group)
                )

            del struc
            del cifWriter
            if isDeepRelax:
                del relaxed_struc
                del spacegroupAnalyzer
            gc.collect()

        for idx, (cif_path, energy_val, space_group) in enumerate(
            top_k_structures_list, start=1
        ):
            print(
                f"{idx}. CIF Path: {cif_path}, Energy: {energy_val}, Space Group: {space_group}"
            )

    ### poly
    def update_top_k_structures_poly(self, struc, struc_energy, space_group):
        dir_path = f"results/{self.compound}_paretocsp2_pop{self.input_config.pop}/top_structures"
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        # Initialize top_k_structures as a dictionary if not already done
        if not hasattr(self, "top_k_structures_dict"):
            self.top_k_structures_dict = {}

        # Define tolerance for energy comparison
        ENERGY_TOLERANCE = (
            1e-6  # Adjust this value as needed (e.g., 1e-6 for 6 decimal places)
        )

        # Generate a unique filename for the structure
        file_name = f"{self.compound}_structure_sg_{space_group}_energy_{struc_energy:.8f}_{self.structure_number}.cif"
        file_path = os.path.join(dir_path, file_name)

        # Save the structure temporarily
        cifWriter = CifWriter(struc, symprec=0.1)
        cifWriter.write_file(file_path)

        # Add or update the space group entry
        if space_group not in self.top_k_structures_dict:
            if len(self.top_k_structures_dict) < self.k:
                # Add new space group with this structure
                self.top_k_structures_dict[space_group] = [
                    (file_path, struc_energy, space_group)
                ]
        else:
            # Check if the energy is within tolerance of any existing energy in this space group
            current_structures = self.top_k_structures_dict[space_group]
            existing_energies = [s[1] for s in current_structures]

            # Skip if the energy is within tolerance of an existing one
            is_duplicate = any(
                abs(struc_energy - existing_energy) < ENERGY_TOLERANCE
                for existing_energy in existing_energies
            )
            if is_duplicate:
                # Remove the temporary file since we won't use it
                if os.path.exists(file_path):
                    os.remove(file_path)
            else:
                # Add structure to existing space group
                current_structures.append((file_path, struc_energy, space_group))
                # Sort by energy and keep only the top j structures
                current_structures.sort(key=lambda x: x[1])
                if len(current_structures) > self.j:
                    # Remove the highest-energy structure and delete its file
                    excess_structure = current_structures.pop()
                    if os.path.exists(excess_structure[0]):
                        os.remove(excess_structure[0])
                self.top_k_structures_dict[space_group] = current_structures

        # If we exceed k space groups, remove the one with the highest minimum energy
        if len(self.top_k_structures_dict) > self.k:
            # Find the space group with the highest minimum energy
            sg_to_remove = max(
                self.top_k_structures_dict,
                key=lambda sg: min([s[1] for s in self.top_k_structures_dict[sg]]),
            )
            # Delete all files associated with this space group
            for file_path, _, _ in self.top_k_structures_dict[sg_to_remove]:
                if os.path.exists(file_path):
                    os.remove(file_path)
            del self.top_k_structures_dict[sg_to_remove]

        del struc
        del cifWriter
        gc.collect()

    def save_top_k_structures_poly(self):
        print("\n\n")
        print(
            f"Top {self.k} Space Groups with up to {self.j} Structures each before deep relaxation:"
        )
        for sg, structures in self.top_k_structures_dict.items():
            print(f"Space Group {sg}:")
            for idx, (cif_path, energy_val, _) in enumerate(structures, start=1):
                print(f"  {idx}. CIF Path: {cif_path}, Energy: {energy_val}")

        print(
            f"\nTop {self.k} Space Groups with up to {self.j} Structures each after deep relaxation:"
        )
        top_k_structures_list = []
        isDeepRelax = True

        # Flatten the dictionary into a list for processing
        all_structures = []
        for sg, structures in self.top_k_structures_dict.items():
            all_structures.extend(structures)

        for idx, (file_path, energy_val, space_group) in tqdm(
            enumerate(all_structures, start=1),
            total=len(all_structures),
            desc=f"Processing and saving top {self.k * self.j} structures",
        ):
            struc = Structure.from_file(file_path)
            relaxed_energy = energy_val
            relaxed_sg = space_group

            if isDeepRelax:
                try:
                    with open(os.devnull, "w") as devnull:
                        with contextlib.redirect_stdout(
                            devnull
                        ), contextlib.redirect_stderr(devnull):
                            relaxed_struc = relaxer.relax(struc)
                    struc = relaxed_struc["final_structure"]
                    energy_predict = m3gnet_energy.predict_structure(struc)
                    relaxed_energy = energy_predict.numpy()[0][0] / len(struc)
                except:
                    pass

                try:
                    spacegroupAnalyzer = SpacegroupAnalyzer(struc, symprec=0.1)
                    struc = spacegroupAnalyzer.get_symmetrized_structure()
                    relaxed_sg = spacegroupAnalyzer.get_space_group_number()
                except:
                    pass

            # Save the relaxed structure
            cifWriter = CifWriter(struc, symprec=0.1)
            relaxed_structure_path = f"results/{self.compound}_paretocsp2_pop{self.input_config.pop}/top_structures/{self.compound}_top_sg{relaxed_sg}_{idx}.cif"
            cifWriter.write_file(relaxed_structure_path)

            top_k_structures_list.append(
                (relaxed_structure_path, relaxed_energy, relaxed_sg)
            )

            del struc
            del cifWriter
            if isDeepRelax:
                del relaxed_struc
                del spacegroupAnalyzer
            gc.collect()

        # Group by space group for display
        grouped_structures = {}
        for cif_path, energy_val, space_group in top_k_structures_list:
            if space_group not in grouped_structures:
                grouped_structures[space_group] = []
            grouped_structures[space_group].append((cif_path, energy_val))

        # Display the results
        for sg, structures in grouped_structures.items():
            print(f"Space Group {sg}:")
            for idx, (cif_path, energy_val) in enumerate(structures, start=1):
                print(f"  {idx}. CIF Path: {cif_path}, Energy: {energy_val}")

    def atomic_dist_and_volume_limit(self, struc: Structure):
        atom_radii = []
        for i in self.all_atoms:
            if self.elements_info[i][8] == -1:
                atom_radii.append(100.0 / 100.0)
            else:
                atom_radii.append(float(self.elements_info[i][8]) / 100.0)

        for i in range(self.total_atom_count - 1):
            for j in range(i + 1, self.total_atom_count):
                if struc.get_distance(i, j) < (atom_radii[i] + atom_radii[j]) * 0.4:
                    # raise Exception()
                    # print("invlaid dist!!!")
                    del struc
                    del atom_radii
                    gc.collect()
                    return False

        atom_volume = [4.0 * np.pi * r**3 / 3.0 for r in atom_radii]
        sum_atom_volume = sum(atom_volume) / 0.55
        if not (sum_atom_volume * 0.4 <= struc.volume <= sum_atom_volume * 2.4):
            # print(sum_atom_volume)
            # raise Exception()
            # print("invlaid volume!!!")
            del struc
            del atom_radii
            del atom_volume
            gc.collect()
            return False

        # print("ok 1")
        isVaccumLimitCorrect = self.vacuum_size_limit(struc.copy(), max_size=7.0)

        del struc
        del atom_radii
        del atom_volume
        gc.collect()

        if isVaccumLimitCorrect:
            # print("ok 2")
            return True
        else:
            # print("invlaid vaccum limit!!!")
            return False

    def vacuum_size_limit(self, struc: Structure, max_size: float = 10.0):
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
                        # raise Exception()
                        isValid = False
                        break
        del struc
        del foot_points
        gc.collect()
        return isValid


if __name__ == "__main__":
    csp = PredictStructure(input_file_path="config.in")
