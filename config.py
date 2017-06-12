from argparse import ArgumentParser
__author__ = 'Jayeol Chun'

parser = ArgumentParser()

population = parser.add_argument_group('Population Variables')
population.add_argument("-n", "--sample_size", nargs='?', default=15, type=int,
                        help="init population sample size, subject to change in experiment. "
                         "The value is final for testing")
population.add_argument("-e", "--sample_size_end", nargs='?', default=25, type=int,
                        help="end of sample size range for experiment")
population.add_argument("-s", "--sample_size_step", nargs='?', default=3, type=int,
                        help="sample size range step unit for experiment")
population.add_argument("-m", "--mu", nargs='?', default=0.5, type=float,
                        help="init mutation rate value, subject to change in experiment. "
                         "The value is final for testing")
population.add_argument("-o", "--mu_th", nargs='?', default=5.0, type=float,
                        help="mu range step unit for experiment") # arbitrary..

sims = parser.add_argument_group('Simulation & Experiment Variables')
sims.add_argument("-i", "--num_iter", nargs='?', default=3, type=int,
                  help="number of iterations for one experiment or test")
sims.add_argument("-p", "--num_proc", nargs='?', default=1, type=int,
                  help="number of processes for experiment (only for experiment mode)")
sims.add_argument("-t", "--num_tests", nargs='?', default=1, type=int,
                  help="number of processes for experiment (only for experiment mode)")

dirs = parser.add_argument_group('Directories')
dirs.add_argument("--input_dir", type=str, default=None,
                  help="path to the saved trees")
dirs.add_argument("--output_dir", type=str,default=None,
                  help="path to the trees to be saved")

flags = parser.add_argument_group('Miscellaneous Flags for Experiment Control')
flags.add_argument("--test", action="store_true",
                   help="test by creating and plotting trees for each model")
flags.add_argument("--graphics", action="store_true",
                   help="produce plots or graphics. Default is no graphics.")
flags.add_argument("--verbose", action="store_true",
                   help="increase output verbosity")

def get_config():
    config, _ = parser.parse_known_args()
    return config
