import time
import torch

from utils.helpers import list_of_distances


def train_or_test(
    model,
    dataloader,
    loader_validation=None,
    optimizer=None,
    epoch=None,
    epoch_update=None,
    coefs=None,
    tot_set=None,
    phase=None,
    class_specific=True,
    use_l_mask=True,
    xp_dir=None,
    log=print,
):
    """
    Train or test the model

    Parameters
    ----------
    model: model
        Model to process

    dataloader: tensor
        Data loader

    loader_validation: tensor
        Loader of the validation set

    optimizer: array
        Optimizer setting

    epoch: integer
        Number of epochs

    epoch_update: integer
        Number of epochs already computed

    coefs: array
        Loss function coefficients

    tot_set: string
        Indicate train or test

    phase: string
        Training or push phase

    class_specific: boolean
        Class-specific processing if True

    use_l_mask: boolean
        Use mask on regularization if True

    xp_dir: string
        Folder of the experiment

    log: string
        Processing of the output

    Returns
    -------
    accuracy: float
        Calculated accuracy
    """
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    total_separation_cost = 0
    total_avg_separation_cost = 0

    n_examples_validation = 0
    n_correct_validation = 0
    accu_validation = None

    for i, (sample, label) in enumerate(dataloader):
        input = sample.cuda()
        target = label.cuda()
        target = target.to(dtype=torch.long)

        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            output, min_distances = model(input)
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                max_dist = (
                    model.module.prototype_shape[1]
                    * model.module.prototype_shape[2]
                    * model.module.prototype_shape[3]
                )

                # Calculate cluster cost
                prototypes_of_correct_class = torch.t(
                    model.module.prototype_class_identity[:, target.cpu()]
                ).cuda()
                inverted_distances, _ = torch.max(
                    (max_dist - min_distances) * prototypes_of_correct_class, dim=1
                )
                cluster_cost = torch.mean(max_dist - inverted_distances)

                # Calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                inverted_distances_to_nontarget_prototypes, _ = torch.max(
                    (max_dist - min_distances) * prototypes_of_wrong_class, dim=1
                )
                separation_cost = torch.mean(
                    max_dist - inverted_distances_to_nontarget_prototypes
                )

                # Calculate avg cluster cost
                avg_separation_cost = torch.sum(
                    min_distances * prototypes_of_wrong_class, dim=1
                ) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)

                # Prototypes L2 regularization
                l21 = model.module.prototype_vectors.norm(p=2)

                # Regularization
                if use_l_mask:
                    l2_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
                    l22 = (model.module.last_layer.weight * l2_mask).norm(p=2)
                else:
                    l22 = model.module.last_layer.weight.norm(p=2)

            else:
                min_distance, _ = torch.min(min_distances, dim=2)
                cluster_cost = torch.mean(min_distance)
                l21 = model.module.prototype_vectors.norm(p=2)
                l22 = model.module.last_layer.weight.norm(p=2)

            # Evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (
                        float(coefs["crs_ent"]) * cross_entropy
                        + float(coefs["clst"]) * cluster_cost
                        + float(coefs["sep"]) * separation_cost
                        + float(coefs["reg_protos"]) * l21
                        + float(coefs["reg_last"]) * l22
                    )
                else:
                    loss = (
                        cross_entropy
                        + 0.8 * cluster_cost
                        - 0.08 * separation_cost
                        + 1e-3 * l21
                        + 1e-4 * l22
                    )
            else:
                if coefs is not None:
                    loss = (
                        float(coefs["crs_ent"]) * cross_entropy
                        + float(coefs["clst"]) * cluster_cost
                        + float(coefs["reg_protos"]) * l21
                        + float(coefs["reg_last"]) * l22
                    )
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-3 * l21 + 1e-4 * l22
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        del input
        del target
        del output
        del predicted
        del min_distances

    # Validation
    if is_train:
        model.eval()

        for i, (sample, label) in enumerate(loader_validation):
            sample_input = sample.cuda()
            target = label.cuda()
            target = target.to(dtype=torch.long)

            grad_req = torch.enable_grad() if is_train else torch.no_grad()
            with grad_req:
                output, _ = model(sample_input)
                _, predicted = torch.max(output.data, 1)
                n_examples_validation += target.size(0)
                n_correct_validation += (predicted == target).sum().item()

    end = time.time()

    log("\tTime: \t{0}".format(end - start))
    log("\tCross entropy: \t{0}".format(total_cross_entropy / n_batches))
    log("\tCluster: \t{0}".format(total_cluster_cost / n_batches))
    if class_specific:
        log("\tSeparation:\t{0}".format(total_separation_cost / n_batches))
        log("\tAvg separation:\t{0}".format(total_avg_separation_cost / n_batches))
    log("\tAccuracy: \t\t{0}%".format(n_correct / n_examples * 100))
    if is_train:
        log(
            "\tAccuracy validation: \t\t{0}%".format(
                n_correct_validation / n_examples_validation * 100
            )
        )
        accu_validation = n_correct_validation / n_examples_validation
    p = model.module.prototype_vectors.view(model.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log("\tp dist pair: \t{0}".format(p_avg_pair_dist.item()))

    f = open(xp_dir + "accuracy.log", "a")
    text = (
        phase
        + " "
        + tot_set
        + " "
        + str(epoch + epoch_update)
        + " "
        + str(n_correct / n_examples)
        + " "
        + str(accu_validation)
    )
    f.write(text + "\n")
    f.close

    return n_correct / n_examples, accu_validation


def train(
    model,
    loader_train,
    loader_validation,
    optimizer,
    epoch,
    epoch_update,
    coefs,
    xp_dir,
    phase,
    class_specific=True,
    log=print,
):
    """
    Launch train

    Parameters
    ----------
    model: model
        Model to train

    loader_train: tensor
        Loader of the train set

    loader_validation: tensor
        Loader of the validation set

    optimizer: array
        Optimizer setting

    epoch: integer
        Number of epochs

    epoch_update: integer
        Number of epochs already computed

    coefs: array
        Loss function coefficients

    xp_dir: string
        Folder of the experiment

    phase: string
        Training or push phase

    class_specific: boolean
        Class-specific processing if True

    log: string
        Processing of the output

    Returns
    -------
    accuracy: float
        Train accuracy
    """
    log("\ttrain")
    model.train()

    return train_or_test(
        model=model,
        dataloader=loader_train,
        loader_validation=loader_validation,
        optimizer=optimizer,
        epoch=epoch,
        epoch_update=epoch_update,
        coefs=coefs,
        tot_set="Train",
        phase=phase,
        class_specific=class_specific,
        xp_dir=xp_dir,
        log=log,
    )


def test(
    model,
    loader_test,
    epoch,
    epoch_update,
    coefs,
    xp_dir,
    phase,
    class_specific=True,
    log=print,
):
    """
    Launch test

    Parameters
    ----------
    model: model
        Model to test

    loader_test: tensor
        Loader of the test set

    epoch: integer
        Number of epochs

    epoch_update: integer
        Number of epochs already computed

    coefs: array
        Loss function coefficients

    xp_dir: string
        Folder of the experiment

    phase: string
        Training or push phase

    class_specific: boolean
        Class-specific processing if True

    log: string
        Processing of the output

    Returns
    -------
    accuracy: float
        Test accuracy
    """
    log("\ttest")
    model.eval()

    return train_or_test(
        model=model,
        dataloader=loader_test,
        optimizer=None,
        epoch=epoch,
        epoch_update=epoch_update,
        tot_set="Test",
        phase=phase,
        class_specific=class_specific,
        xp_dir=xp_dir,
        log=log,
    )


def warm_only(model, log=print):
    """
    Configure model for warm stage

    Parameters
    ----------
    model: model
        Model to train

    log: string
        Processing of the output
    """
    for p in model.module.features.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\twarm")


def joint(model, log=print):
    """
    Configure model for joint training

    Parameters
    ----------
    model: model
        Model to train

    log: string
        Processing of the output
    """
    for p in model.module.features.parameters():
        p.requires_grad = True
    model.module.prototype_vectors.requires_grad = True
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\tjoint")


def last_only(model, log=print):
    """
    Configure model for training last layer only

    Parameters
    ----------
    model: model
        Model to train

    log: string
        Processing of the output
    """
    for p in model.module.features.parameters():
        p.requires_grad = False
    model.module.prototype_vectors.requires_grad = False
    for p in model.module.last_layer.parameters():
        p.requires_grad = True

    log("\tlast layer")
