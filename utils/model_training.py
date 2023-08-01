import os
import torch

from utils.data_loading import get_loaders
from utils.model_loading import load_model
import utils.push as push
from utils.results_export import export_results
import utils.train_and_test as tnt


def train_model_with_prototypes(
    configuration,
    model,
    epoch_update,
    loader_train,
    loader_validation,
    loader_test,
    xp_dir,
    log=print,
):
    """
    Train a model with prototypes

    Parameters
    ----------
    configuration: array
        Elements from the configuration file

    model: model
        Model to train

    epoch_update: integer
        Number of epochs already computed

    loader_train: tensor
        Loader of the train set

    loader_validation: tensor
        Loader of the validation set

    loader_test: tensor
        Loader of the test set

    xp_dir: string
        Folder of the experiment

    log: string
        Processing of the output
    """
    # Optimizers
    joint_optimizer_specs = [
        {
            "params": model.features.parameters(),
            "lr": float(configuration["lrs"]["features"]),
            "weight_decay": 1e-3,
        },
        {
            "params": model.prototype_vectors,
            "lr": float(configuration["lrs"]["prototype_vectors"]),
        },
    ]
    warm_optimizer_specs = [
        {
            "params": model.prototype_vectors,
            "lr": float(configuration["lrs"]["prototype_vectors"]),
        }
    ]
    last_layer_optimizer_specs = [
        {
            "params": model.last_layer.parameters(),
            "lr": float(configuration["lrs"]["last_layer_optimizer_lr"]),
        }
    ]

    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        joint_optimizer,
        step_size=configuration["joint_lr_step_size"],
        gamma=configuration["gamma_lr"],
    )
    model_multi = torch.nn.DataParallel(model)
    push_epochs = [
        i for i in range(configuration["epochs"]) if i % configuration["push_freq"] == 0
    ]

    # Training
    for epoch in range(1, configuration["epochs"] + 1 - epoch_update):
        log("epoch: \t{0}".format(epoch + epoch_update))

        if epoch < configuration["warm_epochs"]:
            tnt.warm_only(model=model_multi, log=log)
            accu_train, accu_validation = tnt.train(
                model=model_multi,
                loader_train=loader_train,
                loader_validation=loader_validation,
                optimizer=warm_optimizer,
                coefs=configuration["coefs_list"],
                epoch=epoch,
                epoch_update=epoch_update,
                xp_dir=xp_dir,
                phase="Training",
                log=log,
            )
        else:
            tnt.joint(model=model_multi, log=log)
            accu_train, accu_validation = tnt.train(
                model=model_multi,
                loader_train=loader_train,
                loader_validation=loader_validation,
                optimizer=joint_optimizer,
                coefs=configuration["coefs_list"],
                epoch=epoch,
                epoch_update=epoch_update,
                xp_dir=xp_dir,
                phase="Training",
                log=log,
            )
            joint_lr_scheduler.step()

        accu_test, _ = tnt.test(
            model=model_multi,
            loader_test=loader_test,
            coefs=configuration["coefs_list"],
            epoch=epoch,
            epoch_update=epoch_update,
            xp_dir=xp_dir,
            phase="Training",
            log=log,
        )
        torch.save(
            obj=model,
            f=os.path.join(
                xp_dir,
                (configuration["model_name"] + "_{0:.2f}.pth").format(accu_test * 100),
            ),
        )

        if (epoch + epoch_update >= configuration["push_start"]) and (
            epoch + epoch_update in push_epochs
        ):
            # Push
            (
                change,
                prototype_shape,
                prototype_class_identity,
                num_prototypes_per_class,
            ) = push.push_prototypes(
                loader_train,
                prototype_network_parallel=model_multi,
                prototype_layer_stride=1,
                root_dir_for_saving_prototypes=os.path.join(xp_dir, "prototypes"),
                epoch_number=epoch + epoch_update,
                prototype_img_filename_prefix="prototype-img",
                prototype_self_act_filename_prefix="prototype-self-act",
                proto_bound_boxes_filename_prefix="bb",
                save_prototype_class_identity=True,
                log=log,
                percentile=99,
            )

            if change:
                epoch_update += epoch
                break

            # Fully connected training
            if configuration["prototype_activation_function"] != "linear":
                tnt.last_only(model=model_multi, log=log)
                for i in range(1, configuration["push_last_epochs"] + 1):
                    log("iteration: \t{0}".format(i))
                    accu_train, accu_validation = tnt.train(
                        model=model_multi,
                        loader_train=loader_train,
                        loader_validation=loader_validation,
                        optimizer=last_layer_optimizer,
                        coefs=configuration["coefs_list"],
                        epoch=i,
                        epoch_update=epoch_update,
                        xp_dir=xp_dir,
                        phase="Push",
                        log=log,
                    )
                    accu_test, _ = tnt.test(
                        model=model_multi,
                        loader_test=loader_test,
                        coefs=configuration["coefs_list"],
                        epoch=i,
                        epoch_update=epoch_update,
                        xp_dir=xp_dir,
                        phase="Push",
                        log=log,
                    )
                    torch.save(
                        obj=model,
                        f=os.path.join(
                            xp_dir,
                            (configuration["model_name"] + "_{0:.2f}.pth").format(
                                accu_test * 100
                            ),
                        ),
                    )

    return (
        change,
        prototype_shape,
        prototype_class_identity,
        num_prototypes_per_class,
        epoch_update,
    )


def train_model(
    configuration,
    model,
    epoch_update,
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    xp_dir,
    log=print,
):
    """
    Train specified model

    Parameters
    ----------
    configuration: array
        Elements from the configuration file

    model: model
        Model to train

    epoch_update: integer
        Number of epochs already computed

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

    log: string
        Processing of the output
    """
    log("Start Training")
    loader_train, loader_validation, loader_test = get_loaders(
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        batch_size=configuration["batch_size"],
        shuffle=True,
        log=log,
    )
    (
        change,
        prototype_shape,
        prototype_class_identity,
        num_prototypes_per_class,
        epoch_update,
    ) = train_model_with_prototypes(
        configuration,
        model,
        epoch_update,
        loader_train,
        loader_validation,
        loader_test,
        xp_dir,
        log=log,
    )
    return (
        change,
        prototype_shape,
        prototype_class_identity,
        num_prototypes_per_class,
        epoch_update,
    )


def train_update(
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
):
    """
    Build and train the model

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

    configuration: array
        Elements from the configuration file

    prototype_shape: array
        Size of the prototype layer

    prototype_class_identity: array
        Mapping prototypes to their class

    num_prototypes_per_class: array
        Number of prototypes for each class

    epoch_update: integer
        Number of epochs already computed

    nbclass: integer
        Number of classes

    xp_dir: string
        Folder of the experiment

    log: string
        Processing of the output
    """
    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        configuration,
        prototype_shape,
        prototype_class_identity,
        num_prototypes_per_class,
        size=[X_train.shape[2], X_train.shape[3]],
        nbclass=nbclass,
    )
    model = model.to(device)

    # Train model
    (
        change,
        prototype_shape,
        prototype_class_identity,
        num_prototypes_per_class,
        epoch_update,
    ) = train_model(
        configuration,
        model,
        epoch_update,
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        xp_dir=xp_dir,
        log=log,
    )

    # Export results
    if change:
        results = []
    else:
        results = export_results(
            model,
            configuration,
            X_train,
            y_train,
            X_validation,
            y_validation,
            X_test,
            y_test,
            xp_dir,
        )
    del model
    return (
        change,
        prototype_shape,
        prototype_class_identity,
        num_prototypes_per_class,
        epoch_update,
        results,
    )
