import os
import yaml
import argparse
import numpy as np
import torch

from utils.helpers import create_logger, save_experiment
from utils.data_loading import import_data
from utils.model_training import train_update


if __name__ == "__main__":
    # Load configuration
    parser = argparse.ArgumentParser(description="LEXNet")
    parser.add_argument(
        "-c",
        "--config",
        default="configuration/config_1.yml",
        help="Configuration File",
    )
    args = parser.parse_args()
    with open(args.config, "r") as config_file:
        configuration = yaml.safe_load(config_file)

    os.environ["CUDA_VISIBLE_DEVICES"] = configuration["gpu"]

    # Create experiment folder
    xp_dir = (
        "./results/"
        + str(configuration["dataset"])
        + "/"
        + str(configuration["model_name"])
        + "/"
        + str(configuration["experiment_run"])
        + "/"
    )
    save_experiment(xp_dir, args.config)
    log, logclose = create_logger(log_filename=os.path.join(xp_dir, "experiment.log"))

    # Load dataset
    X_train, y_train, X_validation, y_validation, X_test, y_test = import_data(
        configuration["dataset"], xp_dir, val_split=configuration["validation_split"]
    )

    # Initial values
    change = True
    epoch_update = 0
    nbclass = len(np.unique(y_train))
    num_prototypes = nbclass * configuration["prototypes"]
    prototype_shape = (
        num_prototypes,
        configuration["base_architecture_last_filters"],
        configuration["prototype_size"][0],
        configuration["prototype_size"][1],
    )
    prototype_class_identity = torch.zeros(num_prototypes, nbclass)
    for j in range(num_prototypes):
        prototype_class_identity[j, j // configuration["prototypes"]] = 1
    num_prototypes_per_class = {}
    for i in range(nbclass):
        num_prototypes_per_class[i] = [i]

    # Train the model
    while change:
        (
            change,
            prototype_shape,
            prototype_class_identity,
            num_prototypes_per_class,
            epoch_update,
            results,
        ) = train_update(
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            configuration,
            prototype_shape,
            prototype_class_identity,
            num_prototypes_per_class,
            epoch_update,
            nbclass,
            xp_dir,
            log,
        )
    # Print results
    print(results)
    logclose()
