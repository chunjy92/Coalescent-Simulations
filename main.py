# -*- coding: utf-8 -*-
import time

from config import get_config
from simulation import Simulation

__author__ = 'Jayeol Chun'


def main(config):
  tic = time.time()
  print("******* Coalescent Simulations Main *******")

  coalSim = Simulation(config)

  # initialize models
  coalSim.setup()

  # run experiments
  coalSim.run()

  # save model
  coalSim.save()

  print("\n******* Program Execution Time: {:.2f} s *******".format(time.time()-tic))

  # if not config.no_exp:
  #   log_data = experiment(models, config)
  #   print("\n******* Program Execution Time: {:.2f} s *******".format(time.time()-tic))
  #
  #   print("Log DATA:")
  #   print(log_data)
  #   plot_result(log_data)

if __name__ == '__main__':
  main(get_config())
