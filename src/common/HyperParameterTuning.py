from ModelTrainer import ModelTrainer

import keras
import kerastuner
import numpy as np

import pandas as pd
from params import QUIPU_LEN_CUT, QUIPU_N_LABELS

class CVTuner(kerastuner.engine.tuner.Tuner):
  def run_trial(self, trial, batch_size=128, n_epochs_max=1, n_splits = 5):
    mt = ModelTrainer(n_epochs_max = n_epochs_max, use_weights=True, use_brow_aug = True, batch_size=batch_size, model_name=self.hypermodel.__class__.__name__)
    train_acc, valid_acc, test_acc, n_epoch = mt.hpo_crossval(trial, self.hypermodel, n_splits=n_splits, data_folder=f'../../results/QuipuTrainedWithES_trial_{trial.trial_id}.csv')

    self.oracle.update_trial(trial.trial_id, {'val_accuracy': test_acc})

class Tuner():

  def __init__(self, model, search_function = "random", n_epochs_max = 10, max_trials = 10, batch_size = 128):
    self.model = model
    self.search_function = search_function
    self.n_epochs_max = n_epochs_max
    self.max_trials = max_trials
    self.batch_size = batch_size
   
  def get_tuner(self):
    oracle = None
    if self.search_function == "random":
        oracle = kerastuner.oracles.RandomSearchOracle(objective='val_accuracy',max_trials=self.max_trials)
    elif self.search_function == "bayesian":
      oracle = kerastuner.oracles.BayesianOptimizationOracle(objective='val_accuracy',max_trials=self.max_trials)
    else:
      oracle = kerastuner.oracles.RandomSearchOracle(objective='val_accuracy',max_trials=self.max_trials)
    return CVTuner(
            hypermodel=self.model,
            overwrite=False,
            directory=f"../../results/{self.search_function}",
            project_name=self.model.__class__.__name__,
            oracle=oracle)

  def tune(self):
    tuner = self.get_tuner()
    tuner.search(batch_size=self.batch_size,
                n_epochs_max=self.n_epochs_max)    
    tuner.results_summary()

  def summary(self):
    self.get_tuner().results_summary()
