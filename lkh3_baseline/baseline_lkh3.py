import os
from subprocess import check_call
import time
import sys
import pickle
import numpy as np
from tqdm import tqdm

from utils import read_from_pickle

class BaselineLKH3():
    def __init__(self, cur_dir, dataset_path, executable_path):
        """
        Args:
            cur_dir: Working directory (for lkh3 files)
            dataset_path: Path to graph data
            executable_path: Path to LKH-3 executable (LKH file)
        """

        print('This class was written and tested for Unix systems only')

        self.platform = sys.platform

        self.dir = cur_dir

        print('Creating directory ', self.dir)
        os.makedirs(self.dir, exist_ok=True)

        print('Loading validation dataset ', dataset_path)
        self.val_data = read_from_pickle(dataset_path, return_tf_data_set=False)

        self.problem_files = []

        self.tour_files= []

        self.params_files = []

        self.executable = executable_path

        self.depot_list = []
        self.loc_list = []
        self.demands_list = []
        self.tour_list = []

        # Params for LKH-3
        self.runs = 1
        self.seed = 1234

    def get_file_dir(self, filename):
        return os.path.join(self.dir, filename)

    def create_lkh_data(self):
        """Transforms data to LKH-3 format and saves it
        """

        len_of_val_dataset = self.val_data[0].shape[0]  # num of graphs
        self.n_size = self.val_data[2].shape[1]  # num of nodes in graph

        for i in range(len_of_val_dataset):

            depot, loc, demands = self.val_data[0][i].numpy().tolist(), \
                                  self.val_data[1][i].numpy().tolist(), \
                                  self.val_data[2][i].numpy().tolist()

            # LKH-3 requires int as demands
            demands = [int(30.0 * x) for x in demands]

            self.depot_list.append(depot)
            self.loc_list.append(loc)
            self.demands_list.append(demands)

            problem_file = self.get_file_dir('graph_{}.vrp'.format(i))

            self.problem_files.append(problem_file)

            BaselineLKH3.make_vrp_file(problem_file, depot, loc, demands, 30, 1)

            tour_file = self.get_file_dir('graph_{}.tour'.format(i))

            self.tour_files.append(tour_file)

            params = {"PROBLEM_FILE": problem_file,
                      "OUTPUT_TOUR_FILE": tour_file,
                      "RUNS": self.runs,
                      "SEED": self.seed}

            params_file = self.get_file_dir('graph_{}.par'.format(i))

            self.params_files.append(params_file)

            BaselineLKH3.write_lkh_par(params_file, params)

            if i%1000 == 0:
                print('Number of processed graphs: {}'.format(i))

        file_paths = list(zip(self.params_files,
                              self.tour_files))

        self.save_to_pickle(file_paths, self.get_file_dir('all_files_paths.pkl'))

        print('LKH-3 data has been successfully created and saved into ', self.dir)

    def save_to_pickle(self, object, path):
        with open(path, 'wb') as f:
            pickle.dump(object, f)

    def run_lkh3(self, path_tuples_for_lkh3=None):

        assert self.platform == 'linux', 'LKH3 can be run only on Unix systems'

        duration_list = []

        if path_tuples_for_lkh3 is None:
            path_tuples_for_lkh3 = zip(self.params_files, self.tour_files)

        for cur_params_file, cur_tour_file in tqdm(path_tuples_for_lkh3):

            base=os.path.basename(cur_params_file)
            log_filename = os.path.join(self.dir, "{}.lkh{}.log".format(os.path.splitext(base)[0], self.runs))

            # Run LKH-3 and write log into log_filename
            with open(log_filename, 'w') as f:
                start = time.time()
                check_call([self.executable, cur_params_file], stdout=f, stderr=f)
                duration = time.time() - start

            duration_list.append(duration)

            # Read tour file (output from LKH-3)
            tour = BaselineLKH3.read_tour_file(cur_tour_file, n=self.n_size)
            self.tour_list.append(tour)

        # Calculate costs for each tour
        vrp_costs_list = []
        for depot, loc, tour in zip(self.depot_list, self.loc_list, self.tour_list):
            vrp_costs_list.append(BaselineLKH3.calc_vrp_cost(depot, loc, tour))

        self.save_to_pickle(vrp_costs_list, self.get_file_dir('full_vrp_costs_list.pkl'))
        self.save_to_pickle(self.tour_list, self.get_file_dir('full_vrp_tour_list.pkl'))
        self.save_to_pickle(duration_list, self.get_file_dir('full_vrp_duration_list.pkl'))

        return vrp_costs_list, self.tour_list, duration_list

    @staticmethod
    def read_tour_file(filename, n):
        """Parse output file from LKH-3
        """
        with open(filename, 'r') as f:
            tour = []
            dimension = 0
            started = False
            for line in f:
                if started:
                    loc = int(line)
                    if loc == -1:
                        break
                    tour.append(loc)
                if line.startswith("DIMENSION"):
                    dimension = int(line.split(" ")[-1])

                if line.startswith("TOUR_SECTION"):
                    started = True

        # We subtract 1 from all node indices since LKH-3 indices start from 1
        tour = np.array(tour).astype(int) - 1
        # We replace all indices larger then total number of nodes in graph by 0 (since they mark return to the depot)
        tour[tour > n] = 0
        return tour[1:].tolist()

    @staticmethod
    def calc_vrp_cost(depot, loc, tour):
        """Calculate total length of tour
        """
        loc_with_depot = np.vstack((np.array(depot)[None, :], np.array(loc)))
        sorted_locs = loc_with_depot[np.concatenate(([0], tour, [0]))]
        return np.linalg.norm(sorted_locs[1:] - sorted_locs[:-1], axis=-1).sum()

    @staticmethod
    def make_vrp_file(filename, depot, loc, demand, capacity, grid_size, name="problem"):
        """Convert graph data to files in LKH-3 format
        """
        with open(filename, 'w') as f:
            f.write("\n".join([
                "{} : {}".format(k, v)
                for k, v in (
                    ("NAME", name),
                    ("TYPE", "CVRP"),
                    ("DIMENSION", len(loc) + 1),
                    ("EDGE_WEIGHT_TYPE", "EUC_2D"),
                    ("CAPACITY", capacity)
                )
            ]))
            f.write("\n")
            f.write("NODE_COORD_SECTION\n")
            f.write("\n".join([
                "{}\t{}\t{}".format(i + 1, int(x / grid_size * 100000 + 0.5), int(y / grid_size * 100000 + 0.5))  # VRPlib does not take floats
                #"{}\t{}\t{}".format(i + 1, x, y)
                for i, (x, y) in enumerate([depot] + loc)
            ]))
            f.write("\n")
            f.write("DEMAND_SECTION\n")
            f.write("\n".join([
                "{}\t{}".format(i + 1, d)
                for i, d in enumerate([0] + demand)
            ]))
            f.write("\n")
            f.write("DEPOT_SECTION\n")
            f.write("1\n")
            f.write("-1\n")
            f.write("EOF\n")

    @staticmethod
    def write_lkh_par(filename, parameters):
        """Prepare LKH-3 parameter files
        """

        default_parameters = {"SPECIAL": None,
                              "MAX_TRIALS": 10000,
                              "RUNS": 10,
                              "TRACE_LEVEL": 1,
                              "SEED": 0
                             }

        with open(filename, 'w') as f:
            for k, v in {**default_parameters, **parameters}.items():
                if v is None:
                    f.write("{}\n".format(k))
                else:
                    f.write("{} = {}\n".format(k, v))