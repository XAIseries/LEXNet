import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lexnet_backbone import lexnet_backbone
from utils.receptive_field import compute_proto_layer_rf_info


base_architecture_to_features = {
    "lexnet_backbone": lexnet_backbone,
}


class LEXNet(nn.Module):
    """Class used to create a LEXNet model"""

    def __init__(
        self,
        features,
        size,
        prototype_shape,
        prototype_class_identity,
        num_prototypes_per_class,
        proto_layer_rf_info,
        nbclass,
        init_weights=True,
        prototype_activation_function="log",
        log=print,
    ):
        """
        Parameters
        ----------
        features: string
            Name of the CNN backbone

        size: array
            Dimensions of the input data

        prototype_shape: array
            Size of the prototype layer

        prototype_class_identity: array
            Mapping prototypes to their class

        num_prototypes_per_class: array
            Number of prototypes for each class

        proto_layer_rf_info: array
            Prototype receptive fields information

        nbclass: integer
            Number of classes

        init_weights: boolean
            Initialize the weights of the last layer if True

        prototype_activation_function: string
            Activation function of the prototype layer

        log: string
            Processing of the outputs
        """
        super(LEXNet, self).__init__()
        self.size = size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.nbclass = nbclass
        self.epsilon = 1e-4
        self.prototype_activation_function = prototype_activation_function
        self.prototype_class_identity = prototype_class_identity
        self.num_prototypes_per_class = num_prototypes_per_class
        self.proto_layer_rf_info = proto_layer_rf_info
        self.features = features
        self.prototype_vectors = nn.Parameter(
            torch.rand(self.prototype_shape), requires_grad=True
        )
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)
        self.last_layer = nn.Linear(self.num_prototypes, self.nbclass, bias=False)
        if init_weights:
            self._initialize_weights()

    def conv_features(self, x):
        x = self.features(x)
        return x

    def _l2_convolution(self, x):
        xp = F.conv2d(input=x, weight=self.prototype_vectors)
        x = F.conv2d(input=x**2, weight=self.ones)
        p = torch.sum(self.prototype_vectors**2, dim=(1, 2, 3))
        p = p.view(-1, 1, 1)
        return F.relu(x - 2 * xp + p)

    def prototype_distances(self, x):
        conv_features = self.conv_features(x)
        distances = self._l2_convolution(conv_features)
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == "log":
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == "linear":
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def forward(self, x):
        distances = self.prototype_distances(x)
        min_distances = -F.max_pool2d(
            -distances, kernel_size=(distances.size()[2], distances.size()[3])
        )
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        return logits, min_distances

    def push_forward(self, x):
        conv_output = self.conv_features(x)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def __repr__(self):
        rep = (
            "LEXNet(\n"
            "\tfeatures: {},\n"
            "\tsize: {},\n"
            "\tprototype_shape: {},\n"
            "\tproto_layer_rf_info: {},\n"
            "\tnum_classes: {},\n"
            "\tepsilon: {}\n"
            ")"
        )
        return rep.format(
            self.features,
            self.size,
            self.prototype_shape,
            self.proto_layer_rf_info,
            self.nbclass,
            self.epsilon,
        )

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations
        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations
        )

    def _initialize_weights(self):
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)


def construct_LEXNet(
    base_architecture,
    size,
    prototype_shape,
    prototype_class_identity,
    num_prototypes_per_class,
    nbclass,
    baseline_activation_function="relu",
    prototype_activation_function="log",
    log=print,
):
    """
    Collect the inputs and build the model

    Parameters
    ----------
    base_architecture: string
        Name of the CNN backbone

    size: array
        Dimensions of the input data

    prototype_shape: array
        Size of the prototype layer

    prototype_class_identity: array
        Mapping prototypes to their class

    num_prototypes_per_class: array
        Number of prototypes for each class

    nbclass: integer
        Number of classes

    baseline_activation_function: string
        Activation function of the last layer of the CNN backbone

    prototype_activation_function: string
        Activation function of the prototype layer

    log: string
        Processing of the outputs

    Returns
    -------
    model: model
        LEXNet model
    """
    features = base_architecture_to_features[base_architecture](
        baseline_activation_function=baseline_activation_function
    )
    layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    proto_layer_rf_info = compute_proto_layer_rf_info(
        size=size,
        layer_filter_sizes=layer_filter_sizes,
        layer_strides=layer_strides,
        layer_paddings=layer_paddings,
        prototype_kernel_size=prototype_shape[2],
    )
    return LEXNet(
        features=features,
        size=size,
        prototype_shape=prototype_shape,
        prototype_class_identity=prototype_class_identity,
        num_prototypes_per_class=num_prototypes_per_class,
        proto_layer_rf_info=proto_layer_rf_info,
        nbclass=nbclass,
        init_weights=True,
        prototype_activation_function=prototype_activation_function,
        log=log,
    )


def model_lexnet(
    configuration,
    prototype_shape,
    prototype_class_identity,
    num_prototypes_per_class,
    size,
    nbclass,
):
    """
    Generate the model

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
        LEXNet model
    """
    model = construct_LEXNet(
        base_architecture=configuration["base_architecture"],
        size=size,
        prototype_shape=prototype_shape,
        prototype_class_identity=prototype_class_identity,
        num_prototypes_per_class=num_prototypes_per_class,
        nbclass=nbclass,
        baseline_activation_function=configuration["baseline_activation_function"],
        prototype_activation_function=configuration["prototype_activation_function"],
    )
    return model
