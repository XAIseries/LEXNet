from models.lexnet import model_lexnet


def load_model(
    configuration,
    prototype_shape,
    prototype_class_identity,
    num_prototypes_per_class,
    size,
    nbclass,
):
    """
    Load specified model

    Parameters
    ----------
    configuration: array
        Elements from the configuration file

    prototype_shape: array
        Size of the prototype layer

    prototype_class_identity: array
        Mapping prototypes to their class

    num_prototypes_per_class: array
        Number of prototypes for each class

    size: array
        Dimensions of the input data

    nbclass: integer
        Number of classes

    Returns
    -------
    model: model
        Specified model
    """
    models_dict = {
        "lexnet": model_lexnet,
    }
    model = models_dict[configuration["model_name"]](
        configuration,
        prototype_shape,
        prototype_class_identity,
        num_prototypes_per_class,
        size,
        nbclass,
    )
    return model
