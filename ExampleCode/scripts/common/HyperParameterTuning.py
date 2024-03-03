from ModelTrainer import ModelTrainer
import keras_tuner

def tune(model, search_function = "random_eval", validation = "quick", n_epochs_max = 10, max_trials = 10):
    mt = ModelTrainer(n_epochs_max=n_epochs_max)

    tuner = None

    if(search_function == "random_eval"):
        tuner = keras_tuner.RandomSearch(
            hypermodel=model,
            max_trials=max_trials,
            overwrite=True,
            directory="../../results",
            project_name="custom_eval",
        )
    elif(search_function == "something"): # Place holder for something else
        tuner = keras_tuner.RandomSearch(
            hypermodel=model,
            max_trials=max_trials,
            overwrite=True,
            directory="../../results",
            project_name="custom_eval",
        )
    else:
        tuner = keras_tuner.RandomSearch(
            hypermodel=model,
            max_trials=max_trials,
            overwrite=True,
            directory="../../results",
            project_name="custom_eval",
        )

    if(validation == "cross"):
        tuner.search(mt.hpo_crossval_es)
    elif(validation == "quick"):
        tuner.search(mt.hpo_train_es)
    else:
        tuner.search(mt.hpo_train_es)

    tuner.results_summary()
