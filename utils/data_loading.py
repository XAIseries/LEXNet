import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold


def load_data(dataset, preprocess=True):
    """
    Import train and test sets

    Parameters
    ----------
    dataset: string
        Name of the dataset

    preprocess: boolean
        Perform preprocessing if True

    Returns
    -------
    sets: array
        Train and test sets
    """
    ### Train set ###
    # UTS to MTS (format of x: tabular [n_samples, n_packets], features: +/- packet size)
    X_train = np.load("./data/" + dataset + "/train_x.npy")
    X1 = abs(X_train)
    X2 = np.sign(X_train)

    if preprocess:
        scaler1 = preprocessing.MinMaxScaler()
        X1 = scaler1.fit_transform(X1)

        scaler2 = preprocessing.MinMaxScaler()
        X2 = scaler2.fit_transform(X2)

    X3 = np.concatenate(np.array([X1, X2]), axis=1)
    X_train = np.reshape(X3, (X_train.shape[0], 1, X_train.shape[1], 2), order="F")

    # Labels encoding
    y_train = np.load("./data/" + dataset + "/train_y.npy")
    le = preprocessing.LabelEncoder()
    y_train = le.fit_transform(y_train)

    ### Test set ###
    # UTS to MTS (format of x: tabular [n_samples, n_packets], features: +/- packet size)
    X_test = np.load("./data/" + dataset + "/test_x.npy")
    X1 = abs(X_test)
    X2 = np.sign(X_test)

    if preprocess:
        X1 = scaler1.transform(X1)
        X2 = scaler2.transform(X2)

    X3 = np.concatenate(np.array([X1, X2]), axis=1)
    X_test = np.reshape(X3, (X_test.shape[0], 1, X_test.shape[1], 2), order="F")

    # Labels encoding
    y_test = np.load("./data/" + dataset + "/test_y.npy")
    y_test = le.transform(y_test)

    return X_train, y_train, X_test, y_test


def import_data(dataset, xp_dir, preprocess=True, val_split=[5, 1]):
    """
    Generate train, validation and test sets

    Parameters
    ----------
    dataset: string
        Name of the dataset

    xp_dir: string
        Folder of the experiment

    preprocess: boolean
        Perform preprocessing if True

    val_split: array
        Number of folds and the selected one

    Returns
    -------
    sets: array
        Train, validation and test sets
    """
    # Load data
    X_train, y_train, X_test, y_test = load_data(dataset, preprocess=preprocess)

    # Train/validation split
    skf = StratifiedKFold(n_splits=5, shuffle=False)
    train_index, val_index = list(skf.split(X_train, y_train))[val_split[1] - 1]
    X_train, X_val = X_train[train_index], X_train[val_index]
    y_train, y_val = y_train[train_index], y_train[val_index]
    np.save(xp_dir + "train_index.npy", train_index)
    np.save(xp_dir + "validation_index.npy", val_index)

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_loaders(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    batch_size,
    shuffle=True,
    log=print,
):
    """
    Convert arrays to data loaders

    Parameters
    ----------
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

    batch_size: integer
        Batch size

    shuffle: boolean
        Shuffle the data if True

    log: string
        Processing of the outputs

    Returns
    -------
    loaders: tensor
        Train, validation and test loaders
    """
    # Parameters
    params = {"batch_size": batch_size, "shuffle": shuffle, "num_workers": 2}

    # Loaders
    train_set = TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    train_loader = DataLoader(train_set, **params)

    validation_set = TensorDataset(
        torch.Tensor(X_validation), torch.Tensor(y_validation)
    )
    validation_loader = DataLoader(validation_set, **params)

    test_set = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
    test_loader = torch.utils.data.DataLoader(test_set, **params)

    log("training set size: {0}".format(len(train_loader.dataset)))
    log("validation set size: {0}".format(len(validation_loader.dataset)))
    log("test set size: {0}".format(len(test_loader.dataset)))
    log("number of classes: {0}".format(len(np.unique(y_train))))
    log("batch size: {0}".format(batch_size))

    return train_loader, validation_loader, test_loader
