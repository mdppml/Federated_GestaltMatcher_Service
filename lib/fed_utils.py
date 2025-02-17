import copy

import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Unified Argument Parser for GestaltMatcher')

    # Run parameters
    parser.add_argument('--session', type=int, dest='session',
                        help='session used to distinguish model tests.')
    parser.add_argument('--batch_size', type=int, default=128, metavar='BN',
                        help='input batch size for training (default: 280)')
    parser.add_argument('--epochs', type=int, default=50, metavar='E',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',
                        help='learning rate (default: 0.005)')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='how many images (not batches) to wait before logging training status')
    parser.add_argument('--val_interval', type=int, default=10000,
                        help='how many images (not batches) to wait before validation is evaluated (and optimizer is '
                             'stepped).')
    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help='Use tensorboard for logging')

    # Model parameters
    parser.add_argument('--model_type', default='glint360k_r50', dest='model_type',
                        help='model backend to use')
    parser.add_argument('--in_channels', default=3, dest='in_channels', type=int,
                        help='number of color channels of the images used as input (default: 1)')
    parser.add_argument('--img_size', default=112, dest='img_size', type=int,
                        help='input image size of the model (default: 100)')
    parser.add_argument('--unfreeze', action='store_false', default=True, dest='freeze',
                        help='flag to set if you want to unfreeze the base model weights.')
    parser.add_argument('--paper_model', default='None', dest='paper_model', type=str,
                        help='Use when reproducing paper models a) r50-mix, or b) r100')

    # Dataset parameters
    parser.add_argument('--dataset', default='gmdb', dest='dataset',
                        help='which dataset to use. (Options: "casia", "gmdb")')
    parser.add_argument('--dataset_type', default='', dest='dataset_type',
                        help='type of the dataset to use, e.g. normal (="") or augmented(="aug") (default="")')
    parser.add_argument('--dataset_version', default='v1.0.3', dest='dataset_version', type=str,
                        help='version of the dataset to use (default="v1.0.3")')
    parser.add_argument('--lookup_table', default='', dest='lookup_table_path',
                        help='lookup table path, use if you want to load path instead of generating a lookup table ('
                             'default = "")')
    parser.add_argument('--lookup_table_save_path', default='', dest='lookup_table_save_path',
                        help='lookup table save path (default = "")')
    parser.add_argument('--lookup_table_dir', default='', dest='lookup_table_dir',
                        help='the path of the lookup table files')

    # File locations
    parser.add_argument('--data_dir', default='/home/uenal/GestaltMatcher/data', dest='data_dir',
                        help='Location of the data directory (not dataset). (default = server)')
    parser.add_argument('--weight_dir', default='saved_models', dest='weight_dir',
                        help='Location of the model weights directory. (default = "saved_models")')
    parser.add_argument('--model_dir', default='', dest='model_dir',
                        help='Path to the data directory containing the trained model weights.')
    parser.add_argument('--encoding_dir', default='', dest='encoding_dir',
                        help='Path to the data directory where the encodings will be saved or the path of the '
                             'directory for encodings.')

    # Miscellaneous
    parser.add_argument('--local', action='store_true', default=False,
                        help='Running on local machine, fewer num_workers')
    parser.add_argument('--num_workers', default=4, dest='num_workers', type=int,
                        help='Number of workers for dataloaders (default = 4)')

    # Federated learning parameters
    parser.add_argument('--num_parties', default=4, dest='num_parties', type=int,
                        help='Number of parties in the federated setting if it is federated (default = 4)')
    parser.add_argument('--party_id', default=None, dest='party_id', type=int,
                        help='The id of the party in the federated setting if it is federated (default = "")')
    parser.add_argument('--num_classes', default=204, dest='num_classes', type=int,
                        help='The number of classes - it needs to be shared and the same among parties (default = 204)')
    parser.add_argument('--aggregation_period', default=1, dest='aggregation_period', type=int,
                        help='Number of epochs after which the models are aggregated (default = 1)')
    parser.add_argument('--federated_metadata_path', default='', dest='federated_metadata_path',
                        help='Location of the federated learning related metadata files (default = "")')
    parser.add_argument('--aggregation_method', default='mean', dest='aggregation_method', type=str,
                        help='Aggregation method of local models - options: mean, weighted_average, only_encoder ('
                             'default = mean)')
    parser.add_argument('--base_path', default='', dest='base_path', help='the base path of the files')
    parser.add_argument('--metadata_file_prefix', default='', dest='metadata_file_prefix',
                        help='Metadata file prefix determining the experiment files (default = ""')

    return parser.parse_args()


def initialize_zero_model(model, ignore_keys=None):
    """
    Initialize a model with all parameters set to zero based on the given model structure.

    :param model: A dictionary representing the model with its parameters as tensors
    :param ignore_keys: (Optional) List of keys to ignore while creating a new parameter dictionary of the given model
    :return: A new model with the same structure but all parameters set to zero
    """
    if ignore_keys is None:
        ignore_keys = []
    zero_model = {}
    for key, value in model.items():
        if key not in ignore_keys:
            zero_model[key] = torch.zeros_like(value)
    return zero_model


def mean_aggregation(models):
    '''
    Aggregate given models using mean aggregation. This is similar to weighted averaging where all weights are equal.
    :param args: Parsed arguments
    :param models: Local models that will be aggregated
    :return: Aggregated model
    '''
    print("Mean aggregation is called!")
    agg_model = copy.deepcopy(models[0])
    for p in range(1, len(models)):
        for (agg_name, p_agg), (sing_name, p_sing) in zip(agg_model.items(), models[p].items()):
            agg_model[agg_name] = p_agg + p_sing
        # print(f"{tmp_path} is aggregated!")

    # average aggregation over the parameters of the individual models
    for agg_name, p_agg in agg_model.items():
        agg_model[agg_name] = p_agg / len(models)
    return agg_model


def weighted_average_aggregation(args, models, weights):
    '''
    Aggregate given models using weighted average aggregation.
    :param args: Parsed arguments
    :param models: Local models that will be aggregated - collections.OrderedDict
    :param weights: Weights of local models - note that the weights are per class - not per local model!
    :return: Aggregated model
    '''
    print("Weighted average aggregation is called!")
    agg_model = initialize_zero_model(models[0])
    for p in range(0, args.num_parties):
        for (agg_name, p_agg), (sing_name, p_sing) in zip(agg_model.items(), models[p].items()):
            if agg_name == 'classifier.1.weight':
                agg_model[agg_name] = p_agg + (weights[:, p].reshape(-1, 1) * p_sing)
            else:
                agg_model[agg_name] = p_agg + p_sing
        # print(f"{tmp_path} is aggregated!")

    # weighted average aggregation for the classifier layer and mean average over the parameters of the individual models
    for agg_name, p_agg in agg_model.items():
        if agg_name == 'classifier.1.weight':
            agg_model[agg_name] = p_agg / torch.sum(weights, axis=1).reshape(-1, 1)
        else:
            agg_model[agg_name] = p_agg / args.num_parties
    return agg_model


def encoder_aggregation(models, weights=None):
    '''
    Aggregate the given models' encoder parts only - omit the classifier layer-related components
    :param models: List of local models that will be aggregated - List[collections.OrderedDict]
    :param weights: (Optional) 1D Tensor indicating the weight of each model in the aggregation - if None, equal weights
    are used for all models
    :return: Aggregated model
    '''
    if weights is None:
        weights = torch.tensor([1 / len(models)] * len(models))
    agg_model = initialize_zero_model(models[0])
    ignore_classifier_layer = ['classifier.1.weight', 'classifier.1.bias']  # set of layers to ignore in the aggregation
    for p in range(len(models)):
        for (agg_name, p_agg), (sing_name, p_sing) in zip(agg_model.items(), models[p].items()):
            if agg_name not in ignore_classifier_layer:
                agg_model[agg_name] = p_agg + (p_sing * weights[p])

    return agg_model
