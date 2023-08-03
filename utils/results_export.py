import torch
import pandas as pd

from utils.data_loading import get_loaders


def get_accuracy_loader(model, dataloader):
    """
    Get accuracy from a model and dataloader

    Parameters
    ----------
    model: model
        Trained model

    dataloader: tensor
        Data loader

    Returns
    -------
    accuracy: float
        Accuracy from the model on the dataloader
    """
    n_examples = 0
    n_correct = 0

    model.eval()
    for i, (sample, label) in enumerate(dataloader):
        sample_input = sample.cuda()
        target = label.cuda()
        target = target.to(dtype=torch.long)

        grad_req = torch.no_grad()
        with grad_req:
            output, _ = model(sample_input)
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

    return n_correct / n_examples


def export_results(
    model,
    configuration,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    xp_dir,
):
    """
    Export the results of the experiment

    Parameters
    ----------
    model: model
        Trained model

    configuration: array
        Elements of the configuration file

    X_train: array
        Train set

    y_train: array
        Labels of the train set

    X_validation: array
        Validation set

    y_validation: array
        Labels of the validation set

    X_test: array
        Test set

    y_test: array
        Labels of the test set

    xp_dir: string
        Folder of the experiment

    Returns
    -------
    results: array
        Results of the experiment
    """
    # Compute accuracies
    loader_train, loader_validation, loader_test = get_loaders(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        batch_size=configuration["batch_size"],
    )
    acc_train = get_accuracy_loader(model, loader_train)
    acc_validation = get_accuracy_loader(model, loader_validation)
    acc_test = get_accuracy_loader(model, loader_test)

    # Export results
    results_export = pd.DataFrame()
    results_export = pd.concat(
        [
            results_export,
            pd.DataFrame(
                [
                    [
                        configuration["dataset"],
                        configuration["model_name"],
                        X_train.shape,
                        X_validation.shape,
                        X_test.shape,
                        acc_train,
                        acc_validation,
                        acc_test,
                    ]
                ]
            ),
        ],
        axis=0,
    )
    results_export.columns = [
        "Dataset",
        "Architecture",
        "Train_Size",
        "Validation_Size",
        "Test_Size",
        "Accuracy_Train",
        "Accuracy_Validation",
        "Accuracy_Test",
    ]
    results_export.to_csv(
        xp_dir
        + "/results_"
        + str(configuration["dataset"])
        + "_"
        + str(configuration["model_name"])
        + "_"
        + str(configuration["experiment_run"])
        + ".csv",
        index=False,
    )
    return results_export
