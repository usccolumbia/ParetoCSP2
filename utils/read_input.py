import os
import ast
import configparser

from utils.print_utils import print_run_info
from utils.compound_utils import compound_split


class ReadInput:
    @print_run_info("Read input file")
    def __init__(self, input_file_path="config.in", input_config=None, args=None):
        if input_file_path:
            if not os.path.isfile(input_file_path):
                raise IOError(
                    "Could not find the input configuration file (.in file) in the specified path !"
                )

            config = configparser.RawConfigParser()
            config.read(input_file_path, encoding="utf-8")
        elif input_file_path is None and input_config:
            config = input_config
        else:
            raise RuntimeError("Please input some thing!")

        if args.comp != None:
            self._compound = args.comp
        else:
            self._compound = config.get("BASE", "compound").replace(" ", "")
        self.elements, self.elements_count = compound_split(self._compound)
        # self._total_atom_count = config.getint('BASE', 'atom_count')
        self._total_atom_count = sum(self.elements_count)

        if args.energy_model != None:
            self._energy_model = args.energy_model
        else:
            self._energy_model = config.get("BASE", "energy_model")

        self._output_path = config.get("BASE", "output_path")

        if args.mode != None:
            self._mode = args.mode
        else:
            self._mode = config.get("BASE", "mode")

        if args.gpu != None:
            self._is_use_gpu = args.gpu
        else:
            self._is_use_gpu = config.getboolean("BASE", "use_gpu")

        self._space_group = ast.literal_eval(config.get("LATTICE", "space_group"))
        self._lattice_a = ast.literal_eval(config.get("LATTICE", "lattice_a"))
        self._lattice_b = ast.literal_eval(config.get("LATTICE", "lattice_b"))
        self._lattice_c = ast.literal_eval(config.get("LATTICE", "lattice_c"))
        self._lattice_alpha = ast.literal_eval(config.get("LATTICE", "lattice_alpha"))
        self._lattice_beta = ast.literal_eval(config.get("LATTICE", "lattice_beta"))
        self._lattice_gamma = ast.literal_eval(config.get("LATTICE", "lattice_gamma"))

        if args.alg != None:
            self._algorithm = args.alg
        else:
            self._algorithm = config.get("PROGRAM", "algorithm")

        if args.pop != None:
            self._pop = args.pop
        else:
            self._pop = config.getint("PROGRAM", "pop")

        if args.num_track != None:
            self._num_track = args.num_track
        else:
            self._num_track = config.getint("PROGRAM", "num_track")

        if args.num_copy != None:
            self._num_copy = args.num_copy
        else:
            self._num_copy = config.getint("PROGRAM", "num_copy")

        if args.max_step != None:
            self._max_step = args.max_step
        else:
            self._max_step = config.getint("PROGRAM", "max_step")

        if args.seed != None:
            self._rand_seed = args.seed
        else:
            self._rand_seed = config.getint("PROGRAM", "rand_seed")

        print(
            "  Compound: %s    Total atoms count: %d"
            % (self._compound, self._total_atom_count)
        )
        print("  a:", self._lattice_a, "  b:", self._lattice_b, "  c:", self._lattice_c)
        print(
            "  alpha:",
            self._lattice_alpha,
            "  beta:",
            self._lattice_beta,
            "  gamma:",
            self._lattice_gamma,
        )
        print("  algorithm: %s    Max step: %d\n" % (self._algorithm, self._max_step))

    @property
    def energy_model(self):
        return self._energy_model

    @energy_model.setter
    def energy_model(self, energy_model):
        self._energy_model = energy_model

    @property
    def is_use_gpu(self):
        return self._is_use_gpu

    @is_use_gpu.setter
    def is_use_gpu(self, is_use_gpu):
        self._is_use_gpu = is_use_gpu

    @property
    def output_path(self):
        return self._output_path

    @output_path.setter
    def output_path(self, output_path):
        self._output_path = output_path

    @property
    def compound(self):
        return self._compound

    @compound.setter
    def compound(self, compound):
        self._compound = compound

    @property
    def total_atom_count(self):
        return self._total_atom_count

    @total_atom_count.setter
    def total_atom_count(self, total_atom_count):
        self._total_atom_count = total_atom_count

    @property
    def space_group(self):
        return self._space_group

    @space_group.setter
    def space_group(self, space_group):
        self._space_group = space_group

    @property
    def lattice_a(self):
        return self._lattice_a

    @lattice_a.setter
    def lattice_a(self, lattice_a):
        self._lattice_a = lattice_a

    @property
    def lattice_b(self):
        return self._lattice_b

    @lattice_b.setter
    def lattice_b(self, lattice_b):
        self._lattice_b = lattice_b

    @property
    def lattice_c(self):
        return self._lattice_c

    @lattice_c.setter
    def lattice_c(self, lattice_c):
        self._lattice_c = lattice_c

    @property
    def lattice_alpha(self):
        return self._lattice_alpha

    @lattice_alpha.setter
    def lattice_alpha(self, lattice_alpha):
        self._lattice_alpha = lattice_alpha

    @property
    def lattice_beta(self):
        return self._lattice_beta

    @lattice_beta.setter
    def lattice_beta(self, lattice_beta):
        self._lattice_beta = lattice_beta

    @property
    def lattice_gamma(self):
        return self._lattice_gamma

    @lattice_gamma.setter
    def lattice_gamma(self, lattice_gamma):
        self._lattice_gamma = lattice_gamma

    @property
    def algorithm(self):
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm):
        self._algorithm = algorithm

    @property
    def pop(self):
        return self._pop

    @pop.setter
    def pop(self, pop):
        self._pop = pop

    @property
    def num_track(self):
        return self._num_track

    @num_track.setter
    def num_track(self, num_track):
        self._num_track = num_track

    @property
    def num_copy(self):
        return self._num_copy

    @num_copy.setter
    def num_copy(self, num_copy):
        self._num_copy = num_copy

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, mode):
        self._mode = mode

    @property
    def max_step(self):
        return self._max_step

    @max_step.setter
    def max_step(self, max_step):
        self._max_step = max_step

    @property
    def rand_seed(self):
        return self._rand_seed

    @rand_seed.setter
    def rand_seed(self, rand_seed):
        self._rand_seed = rand_seed
