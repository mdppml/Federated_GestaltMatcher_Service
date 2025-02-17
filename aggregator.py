# Created by Ali Burak on 22/04/2024
import os
import numpy as np
import torch

from lib.utils import wait_until_file_ready
from lib.fed_utils import mean_aggregation, weighted_average_aggregation, encoder_aggregation, parse_args
from lib.models.my_arcface import MyArcFace


def main():
    print("Parameters are being parsed...")
    args = parse_args()
    print("Parameter parsing is done!")

    #########################################################################################################
    #########################################################################################################
    ##################################### TRAINING AGGREGATED MODEL #########################################
    #########################################################################################################
    #########################################################################################################
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    ignore_layers = None

    # read the syndrome info of clients if the aggregation method is weighted averaging
    if args.aggregation_method == 'weighted_average':
        client_unique_labels = []
        for p in range(1, args.num_parties + 1):
            tmp_fname = os.path.join(args.data_dir, "GestaltMatcherDB", args.dataset_version, "gmdb_metadata",
                                     args.federated_metadata_path, f"s{args.session}_syndrome_ids_party_{p}.csv")
            wait_until_file_ready(tmp_fname)
            client_unique_labels.append(np.loadtxt(tmp_fname, delimiter=',', dtype=int))

        syndrome_info = torch.zeros((args.num_classes, args.num_parties), dtype=torch.float)
        for p in range(args.num_parties):
            syndrome_info[client_unique_labels[p], p] = 1
        syndrome_info = syndrome_info.to(device)
        print(f"Syndrome info: {syndrome_info.device}")
    elif args.aggregation_method == 'only_encoder':
        ignore_layers = ['classifier.1.layer', 'classifier.1.bias']

    for m in ['a', 'b']:

        if m == 'a':  # r50 mix
            args.model_type = 'glint360k_r50'
            args.batch_size = 64
        elif m == 'b':  # r100
            args.model_type = 'glint360k_r100'
            args.batch_size = 128
        else:
            print(f"Undefined model type: Model type '{m}' is not known!")
            exit(-1)
        # set the base path for the individual models
        base_model_name = f"s{args.session}_{args.model_type}_512d_gmdb__v1.0.3_bs{args.batch_size}_size112_channels3_" \
                          f"last_model"

        print("Epoch loop will start...")

        # aggregate the individual encoders for the given number of epochs
        for e in range(1, args.epochs + 1):
            print(f"\n=================== Epoch {e} ============================")
            local_models = []
            if e % args.aggregation_period == 0:
                model_path = os.path.join(args.weight_dir, f"s{args.session}_{args.model_type}_512d_{args.dataset}"
                                                           f"_{args.dataset_type}_{args.dataset_version}"
                                                           f"_bs{args.batch_size}_size{args.img_size}"
                                                           f"_channels{args.in_channels}"
                                                           f"_{args.num_parties}_parties_party")

                for p in range(1, args.num_parties + 1):
                    tmp_path = f"{model_path}_{p}_e{e}.pt"
                    wait_until_file_ready(tmp_path)
                    local_models.append(torch.load(tmp_path))

                agg_model = None
                match args.aggregation_method:
                    case 'mean':
                        agg_model = mean_aggregation(local_models)
                    case 'weighted_average':
                        agg_model = weighted_average_aggregation(args, local_models, syndrome_info)
                    case 'only_encoder':
                        agg_model = encoder_aggregation(local_models, ignore_layers)
                    case _:
                        raise ValueError(f"The given aggregation method, {args.aggregation_method}, is not correct.")

                epoch_agg_save_path = os.path.join(args.weight_dir, f"s{args.session}_{args.model_type}_512d_"
                                                                    f"{args.dataset}_{args.dataset_type}"
                                                                    f"_{args.dataset_version}_bs{args.batch_size}_"
                                                                    f"size{args.img_size}_channels{args.in_channels}_"
                                                                    f"{args.num_parties}_parties_aggregated_model_e{e}.pt")
                print(f"Saving epoch-wise aggregated model in {epoch_agg_save_path}...")
                torch.save(agg_model, epoch_agg_save_path)
            print(f"=================== Epoch {e} is over ============================\n")

        # aggregate the final local models in case clients further trained their models after the last aggregation
        if args.epochs % args.aggregation_period != 0:
            print("Additional aggregation is being performed...")
            local_final_models = []
            local_final_models_state_dicts = []
            for p in range(1, args.num_parties + 1):
                tmp_path = os.path.join(args.model_dir, f"{base_model_name}_{args.num_parties}_parties_party_{p}.pth")
                wait_until_file_ready(tmp_path)
                local_final_models.append(torch.load(tmp_path, map_location=device))
                local_final_models_state_dicts.append(local_final_models[-1].state_dict())

            aggregated_model_state_dict = None
            match args.aggregation_method:
                case 'mean':
                    aggregated_model_state_dict = mean_aggregation(local_final_models_state_dicts)
                case 'weighted_average':
                    aggregated_model_state_dict = weighted_average_aggregation(args, local_final_models_state_dicts, syndrome_info)
                case 'only_encoder':
                    aggregated_model_state_dict = encoder_aggregation(local_final_models_state_dicts, ignore_layers)
                case _:
                    raise ValueError(f"The given aggregation method, {args.aggregation_method}, is not correct.")
            aggregated_model = MyArcFace(args.num_classes, dataset_base=os.path.join(args.weight_dir,
                                                                                     f'{args.model_type}.onnx'),
                               device=device, freeze=True).to(device)
            aggregated_model.load_state_dict(aggregated_model_state_dict)
            print("Additional aggregation is done!")

            # # read the first model and set it as base for the aggregated model
            # tmp_path = os.path.join(args.model_dir, f"{base_model_name}_{args.num_parties}_parties_party_{1}.pth")
            # # print(f"(3) Waiting for {tmp_path}...")
            #
            # wait_until_file_ready(tmp_path)
            #
            # # aggregated_model = torch.load(tmp_path, map_location=device).to(device)
            # aggregated_model = torch.load(tmp_path, map_location=device)
            #
            # # read the rest of the individual models and add their parameters' values to the aggregated model
            # for i in range(2, args.num_parties + 1):
            #     tmp_path = os.path.join(args.model_dir, f"{base_model_name}_{args.num_parties}_parties_party_{i}.pth")
            #
            #     wait_until_file_ready(tmp_path)
            #
            #     # single_model = torch.load(tmp_path, map_location=device).to(device)
            #     single_model = torch.load(tmp_path, map_location=device)
            #     for (agg_name, p_agg), (sing_name, p_sing) in zip(aggregated_model.named_parameters(),
            #                                                       single_model.named_parameters()):
            #         p_agg.data += p_sing
            #
            # # divide the parameters' values of the aggregated model by the number of parties to perform average aggregation
            # for agg_name, p_agg in aggregated_model.named_parameters():
            #     p_agg.data /= args.num_parties

            # save the aggregated model
            torch.save(aggregated_model, os.path.join(args.model_dir, f"{base_model_name}_{args.num_parties}_parties_"
                                                                      f"aggregated.pth"))
        # else:
        #     torch.save(agg_model, os.path.join(args.model_dir, f"{base_model_name}_{args.num_parties}_parties_"
        #                                                               f"aggregated.pth"))

    #########################################################################################################
    #########################################################################################################
    ################################# EVALUATE THE ENSEMBLE MODEL #######################################
    #########################################################################################################
    #########################################################################################################


if __name__ == '__main__':
    main()
