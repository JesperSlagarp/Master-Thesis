from ModelTrainer import ModelTrainer

import kerastuner
import numpy as np

import pandas as pd
from params import QUIPU_LEN_CUT, QUIPU_N_LABELS

class CVTuner(kerastuner.engine.tuner.Tuner):
  def run_trial(self, trial, batch_size=32, n_epochs_max=1, n_splits = 5):

    print(f"Running trial {trial.trial_id}")

    mt = ModelTrainer(n_epochs_max = n_epochs_max, use_weights=True, use_brow_aug = True)
    train_acc, valid_acc, test_acc, n_epoch = mt.hpo_crossval_es(trial, self.hypermodel, n_splits=n_splits, data_folder=f'../../results/QuipuTrainedWithES_trial_{trial.trial_id}.csv')

    self.oracle.update_trial(trial.trial_id, {'val_accuracy': test_acc})

def tune(model, search_function = "random", n_epochs_max = 10, max_trials = 10):
    mt = ModelTrainer(n_epochs_max=n_epochs_max)

    oracle = None

    if search_function == "random":
      oracle = kerastuner.oracles.RandomSearchOracle(objective='val_accuracy',max_trials=max_trials)
    elif search_function == "bayesian":
      oracle = kerastuner.oracles.BayesianOptimizationOracle(objective='val_accuracy',max_trials=max_trials)
    else:
      oracle = kerastuner.oracles.RandomSearchOracle(objective='val_accuracy',max_trials=max_trials)

    tuner = CVTuner(
        hypermodel=model,
        overwrite=True,
        directory="../../results",
        project_name="something something",
        oracle=oracle)

    tuner.search(batch_size=64, n_epochs_max=n_epochs_max)    

    tuner.results_summary()
