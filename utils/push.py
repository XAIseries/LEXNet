import torch
import numpy as np
import cv2
import os
import time
from scipy.stats import kurtosis

from utils.receptive_field import compute_rf_prototype
from utils.helpers import makedir, find_high_activation_crop


def push_prototypes(
    dataloader,
    prototype_network_parallel,
    class_specific=True,
    preprocess_input_function=None,
    prototype_layer_stride=1,
    root_dir_for_saving_prototypes=None,
    epoch_number=None,
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    proto_bound_boxes_filename_prefix=None,
    save_prototype_class_identity=True,
    prototype_activation_function_in_numpy=None,
    percentile=95,
    log=print,
):
    """
    Push prototypes to latent patch from the training set

    Parameters
    ----------
    dataloader: tensor
        Data loader

    prototype_network_parallel: model
        Model to process

    class_specific: boolean
        Class-specific processing if True

    preprocess_input_function: string
        Name of preprocessing function

    prototype_layer_stride: integer
        Stride used by the prototype layer on the feature maps

    root_dir_for_saving_prototypes: string
        Folder to save prototypes

    epoch_number: integer
        Number of epochs

    prototype_img_filename_prefix: string
        Prefix of the file to save prototype

    prototype_self_act_filename_prefix: string
        Prefix of the file to save prototype activation values

    proto_bound_boxes_filename_prefix: string
        Prefix of the file to save the prototype boundaries of highly activated region

    save_prototype_class_identity: boolean
        Save prototypes along with the class number if True

    prototype_activation_function_in_numpy: string
        Name of the activation function if not 'log' or 'linear'

    percentile: integer
        Percentile of activation values

    log: string
        Processing of the outputs
    """
    prototype_network_parallel.eval()
    log("\tpush")

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    global_min_fmap_patches = np.zeros(
        [n_prototypes, prototype_shape[1], prototype_shape[2], prototype_shape[3]]
    )

    global_samples_dist = np.full((len(dataloader.dataset), n_prototypes), np.inf)

    """
    proto_rf_boxes and proto_bound_boxes column:
    0: index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    """
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6], fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5], fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(
                root_dir_for_saving_prototypes, "epoch-" + str(epoch_number)
            )
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size
    nbclass = prototype_network_parallel.module.nbclass

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        """
        start_index_of_search keeps track of the index of the sample
        assigned to serve as prototype
        """
        start_index_of_search_batch = push_iter * search_batch_size
        update_prototypes_on_batch(
            search_batch_input,
            start_index_of_search_batch,
            prototype_network_parallel,
            epoch_number,
            global_min_proto_dist,
            global_min_fmap_patches,
            global_samples_dist,
            proto_rf_boxes,
            proto_bound_boxes,
            class_specific=class_specific,
            search_y=search_y,
            nbclass=nbclass,
            preprocess_input_function=preprocess_input_function,
            prototype_layer_stride=prototype_layer_stride,
            dir_for_saving_prototypes=proto_epoch_dir,
            prototype_img_filename_prefix=prototype_img_filename_prefix,
            prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
            prototype_activation_function_in_numpy=prototype_activation_function_in_numpy,
            log=log,
            percentile=percentile,
        )

    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(
            os.path.join(
                proto_epoch_dir,
                proto_bound_boxes_filename_prefix
                + "-receptive_field"
                + str(epoch_number)
                + ".npy",
            ),
            proto_rf_boxes,
        )
        np.save(
            os.path.join(
                proto_epoch_dir,
                proto_bound_boxes_filename_prefix + str(epoch_number) + ".npy",
            ),
            proto_bound_boxes,
        )

    log("\tExecuting push ...")
    prototype_update = np.reshape(global_min_fmap_patches, tuple(prototype_shape))
    prototype_network_parallel.module.prototype_vectors.data.copy_(
        torch.tensor(prototype_update, dtype=torch.float32).cuda()
    )

    update_kurt = False
    update_prototype_shape = None
    update_prototype_class_identity = None
    update_num_prototypes_per_class = None

    if epoch_number < 100:
        log("\tEpoch: \t{0}".format(epoch_number))

        num_protos_class = prototype_network_parallel.module.num_prototypes_per_class
        output = np.full((global_samples_dist.shape[0], nbclass), np.inf)
        for i in num_protos_class:
            protos = num_protos_class[i]
            temp = global_samples_dist[:, protos]
            for j in range(0, global_samples_dist.shape[0]):
                val = np.amin(temp[j, :])
                output[j, i] = val

        # Avg across all samples [n_class]
        output_class = np.mean(output, axis=0)

        # Compute Kurtosis
        kurt = kurtosis(output_class)
        log("\tKurtosis: \t{0}".format(kurt))
        if kurt > 0:
            if epoch_number < 60:
                p_upper = np.percentile(output_class, 75)
            else:
                p_upper = np.percentile(output_class, 90)

            indices_classes = [
                ind for ind in range(len(output_class)) if output_class[ind] > p_upper
            ]
            log("\tIndices_classes: {},".format(indices_classes))
            for k in indices_classes:
                prototype_network_parallel.module.num_prototypes += 1

                num_prototypes_per_class = list(
                    prototype_network_parallel.module.num_prototypes_per_class[k]
                )
                num_prototypes_per_class.append(
                    int(prototype_network_parallel.module.num_prototypes) - 1
                )
                prototype_network_parallel.module.num_prototypes_per_class[k] = tuple(
                    num_prototypes_per_class
                )

                prototype_shape = list(
                    prototype_network_parallel.module.prototype_shape
                )
                prototype_shape[0] += 1
                prototype_network_parallel.module.prototype_shape = tuple(
                    prototype_shape
                )

                second_tensor = torch.zeros(1, nbclass)
                second_tensor[0, k] = 1
                prototype_network_parallel.module.prototype_class_identity = torch.cat(
                    (
                        prototype_network_parallel.module.prototype_class_identity,
                        second_tensor,
                    ),
                    0,
                )

            update_kurt = True
            (
                update_prototype_shape,
                update_prototype_class_identity,
                update_num_prototypes_per_class,
            ) = (
                prototype_network_parallel.module.prototype_shape,
                prototype_network_parallel.module.prototype_class_identity,
                prototype_network_parallel.module.num_prototypes_per_class,
            )
            log(
                "\tnumber of prototypes: {},".format(
                    prototype_network_parallel.module.num_prototypes
                )
            )

            # Export
            np.save(
                os.path.join(proto_epoch_dir, "prototype_class_identity.npy"),
                update_prototype_class_identity,
            )

    end = time.time()
    log("\tpush time: \t{0}".format(end - start))
    return (
        update_kurt,
        update_prototype_shape,
        update_prototype_class_identity,
        update_num_prototypes_per_class,
    )


def update_prototypes_on_batch(
    search_batch_input,
    start_index_of_search_batch,
    prototype_network_parallel,
    epoch_number,
    global_min_proto_dist,
    global_min_fmap_patches,
    global_samples_dist,
    proto_rf_boxes,
    proto_bound_boxes,
    class_specific=True,
    search_y=None,
    nbclass=None,
    preprocess_input_function=None,
    prototype_layer_stride=1,
    dir_for_saving_prototypes=None,
    prototype_img_filename_prefix=None,
    prototype_self_act_filename_prefix=None,
    prototype_activation_function_in_numpy=None,
    percentile=95,
    log=print,
):
    """
    Update prototypes with current batch

    Parameters
    ----------
    search_batch_input: tensor
        Batch from the data loader

    start_index_of_search_batch: integer
        Start index of the batch

    prototype_network_parallel: model
        Model to process

    epoch_number: integer
        Number of epochs

    global_min_proto_dist: array
        Minimum distance to latent patch for each protototype

    global_min_fmap_patches: array
        Closest latent patch for each prototype

    global_samples_dist: array
        Distances samples to prototypes

    proto_rf_boxes: array
        Prototype receptive fields information

    proto_bound_boxes: array
        Prototype boundaries of highly activated region

    class_specific: boolean
        Class-specific processing if True

    search_y: tensor
        Labels of the batch

    nbclass: integer
        Number of classes

    preprocess_input_function: string
        Name of preprocessing function

    prototype_layer_stride: integer
        Stride used by the prototype layer on the feature maps

    dir_for_saving_prototypes: string
        Folder to save prototypes

    prototype_img_filename_prefix: string
        Prefix of the file to save prototypes

    prototype_self_act_filename_prefix: string
        Prefix of the file to save prototype activation values

    prototype_activation_function_in_numpy: string
        Name of the activation function if not 'log' or 'linear'

    percentile: integer
        Percentile of activation values

    log: string
        Processing of the outputs
    """
    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        (
            protoL_input_torch,
            proto_dist_torch,
        ) = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(nbclass)}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    if epoch_number < 100:
        for spl in range(0, proto_dist_.shape[0]):
            for spl_proto in range(n_prototypes):
                dist_sample = np.amin(proto_dist_[spl, spl_proto, :, :])
                global_samples_dist[
                    start_index_of_search_batch + spl, spl_proto
                ] = dist_sample

    for j in range(n_prototypes):
        if class_specific:
            target_class = torch.argmax(
                prototype_network_parallel.module.prototype_class_identity[j]
            ).item()
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][
                :, j, :, :
            ]
        else:
            proto_dist_j = proto_dist_[:, j, :, :]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = list(
                np.unravel_index(np.argmin(proto_dist_j, axis=None), proto_dist_j.shape)
            )
            if class_specific:
                # Change the argmin index from the index among samples of the target class to the index in the entire search batch
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][
                    batch_argmin_proto_dist_j[0]
                ]

            # Retrieve the corresponding feature map patch
            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = (
                batch_argmin_proto_dist_j[1] * prototype_layer_stride
            )
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = (
                batch_argmin_proto_dist_j[2] * prototype_layer_stride
            )
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[
                img_index_in_batch,
                :,
                fmap_height_start_index:fmap_height_end_index,
                fmap_width_start_index:fmap_width_end_index,
            ]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j

            # Get the receptive field boundary of the patch that generates the representation
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(
                search_batch.size(2), batch_argmin_proto_dist_j, protoL_rf_info
            )

            # Get the whole sample
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size0 = original_img_j.shape[1]
            original_img_size1 = original_img_j.shape[0]

            # Save the prototype receptive field information
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            # Find the highly activated region of the original sample
            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if prototype_network_parallel.module.prototype_activation_function == "log":
                proto_act_img_j = np.log(
                    (proto_dist_img_j + 1)
                    / (proto_dist_img_j + prototype_network_parallel.module.epsilon)
                )
            elif (
                prototype_network_parallel.module.prototype_activation_function
                == "linear"
            ):
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(
                    proto_dist_img_j
                )
            upsampled_act_img_j = cv2.resize(
                proto_act_img_j,
                dsize=(original_img_size0, original_img_size1),
                interpolation=cv2.INTER_CUBIC,
            )
            proto_bound_j = find_high_activation_crop(
                upsampled_act_img_j, percentile=percentile
            )

            # Save the prototype boundary (rectangular boundary of highly activated region)
            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_img_filename_prefix is not None:
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(
                        upsampled_act_img_j
                    )
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(
                        rescaled_act_img_j
                    )
                    heatmap = cv2.applyColorMap(
                        np.uint8(255 * rescaled_act_img_j), cv2.COLORMAP_JET
                    )
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[..., ::-1]

                    np.save(
                        os.path.join(
                            dir_for_saving_prototypes,
                            "original_img_j" + str(j) + ".npy",
                        ),
                        original_img_j,
                    )
                    np.save(
                        os.path.join(
                            dir_for_saving_prototypes, "heatmap" + str(j) + ".npy"
                        ),
                        heatmap,
                    )

    if class_specific:
        del class_to_img_index_dict
