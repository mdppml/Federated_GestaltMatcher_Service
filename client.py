import argparse
import copy
import datetime
import os
import random
import time
import cv2

import numpy as np
import pandas as pd
from collections import defaultdict
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from onnx2torch import convert
import albumentations as A

from lib.datasets.utils import get_train_and_val_datasets
from lib.models.my_arcface import MyArcFace
from lib.utils import seed_worker
from lib.utils import wait_until_file_ready
from lib.fed_utils import parse_args


def update_encoder(base_model, new_model, ignore_layers=None):
    if ignore_layers is None:
        ignore_layers = []
    for layer, value in new_model.items():
        if layer not in ignore_layers:
            try:
                getattr(base_model, layer).weight.data = value
            except AttributeError:
                print(f"{layer} does not exist in the base model!")
    return base_model


def save_model(model, path, party_id=-1, num_parties=-1, is_noisy=False, noise_path=None):
    '''
    Save the given model's state_dict on the specified path. If is_noisy, adds the corresponding noise to the model
    parameters before saving.
    :param model: model to be saved
    :param path: path indicating where the model state_dict will be saved
    :param party_id: party id
    :param num_parties: number of parties
    :param is_noisy: true if noise needs to be added before saving the model
    :return: None if not is_noisy, else denoiser
    '''
    # print(f"Parameters:"
    #       f"Path: {path}"
    #       f"Party id: {party_id}"
    #       f"Number of parties: {num_parties}"
    #       f"is_noisy: {is_noisy}"
    #       f"noise_path: {noise_path}")
    if is_noisy:
        # Validation checks
        assert party_id > 0, "Party id is not given or wrong!"
        assert num_parties > 0, "Number of parties is not given or wrong!"
        assert party_id <= num_parties, "Party id cannot be larger than the number of parties!"

        noisy_model = copy.deepcopy(model.state_dict())
        denoiser = {}

        # Precompute all noise values to ensure consistent RNG calls across all parties
        all_noise = {}

        for i in range(1, num_parties + 1):  # Including last party
            tmp_noise = {}
            for key, item in model.state_dict().items():
                # print(f"Key: {key} - Item shape: {item.shape} - Item style: {item.dtype}")
                if torch.is_floating_point(item):
                    # tmp_noise[key] = torch.randn_like(item)  # Always generate noise

                    np_noise = np.random.randn(*item.shape).astype(np.float32)
                    tmp_noise[key] = torch.tensor(np_noise, dtype=item.dtype, device=item.device)
                else:
                    tmp_noise[key] = torch.zeros_like(item)
            all_noise[i] = tmp_noise
            # torch.save(tmp_noise, f"{noise_path}_noise_party_{i}.pt")  # Save each party’s noise

        # Apply noise based on party_id
        if party_id == num_parties:  # Last party subtracts all previous noise
            # print(f"Party id {party_id} equals to the number of party {num_parties}")
            for i in range(1, num_parties):
                for key in model.state_dict().keys():
                    noisy_model[key] -= all_noise[i][key]  # Subtract previous noise

            # Add this party's own noise
            for key in model.state_dict().keys():
                noisy_model[key] += all_noise[num_parties][key]
                denoiser[key] = all_noise[num_parties][key] / num_parties

        else:  # Other parties only add their own noise
            # print(f"Party id: {party_id} - else statement")
            for key in model.state_dict().keys():
                noisy_model[key] += all_noise[party_id][key]

            # Last party's noise for denoising later
            for key in model.state_dict().keys():
                denoiser[key] = all_noise[num_parties][key] / num_parties

        # Save the noisy model
        torch.save(noisy_model, path)

        return denoiser
    else:
        torch.save(model.state_dict(), path)
        return None


def load_model(path, does_denoise=False, denoiser=None):
    deserialized_dict = torch.load(path)
    if does_denoise:
        assert denoiser is not None, "Denoiser is not given!"
        for key, item in deserialized_dict.items():
            if torch.is_floating_point(item):
                deserialized_dict[key] -= denoiser[key]
    return deserialized_dict


# Training loop
def train(args, model, device, train_loader, optimizer, epochs=-1, val_loader=None, scheduler=None):
    model.train()

    # Time measurements
    tick = datetime.datetime.now()

    # Tensorboard Writer
    if args.use_tensorboard:
        writer = SummaryWriter(comment=f"-s{args.session}_{args.num_parties}_parties_party_{args.party_id}_"
                                       f"{args.model_type}")
    global_step = 0

    if epochs == -1:
        epochs = args.epochs

    print(f"Number of total epochs: {epochs}")

    for epoch in range(1, epochs + 1):
        print(f"\n=================== Epoch {epoch} ============================")
        epoch_loss = 0.
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.int64).unsqueeze(1)

            pred, pred_rep = model(data)
            loss = F.cross_entropy(pred, target.view(-1), weight=args.ce_weights)
            loss.backward()

            ## Clipping gradients here, if we get exploding gradients we should revise...
            # nn.utils.clip_grad_value_(model.parameters(), 0.1)

            optimizer.step()
            optimizer.zero_grad()

            del pred, pred_rep, data, target

            epoch_loss += loss.item()
            if (batch_idx + 1) % args.log_interval == 0:
                tock = datetime.datetime.now()
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t(Elapsed time {:.1f}s)'.format(
                    tock.strftime("%H:%M:%S"), epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                                                      100. * batch_idx / len(train_loader), loss.item(),
                    (tock - tick).total_seconds()))
                tick = tock

                if args.use_tensorboard:
                    writer.add_scalar('Train/ce_loss', loss.item(), global_step)

            del loss
            if val_loader:
                if (batch_idx + 1) % args.val_interval == 0:
                    # avg_val_loss, t_acc, t5_acc, ma_t1_acc, ma_t5_acc = validate(model, device, val_loader, args)
                    avg_val_loss, t_acc, t5_acc = validate(model, device, val_loader, args)

                    tick = datetime.datetime.now()

                    if args.use_tensorboard:
                        writer.add_scalar('Val/ce_loss', avg_val_loss, global_step)
                        writer.add_scalar('Val/top_acc', t_acc, global_step)
                        writer.add_scalar('Val/top_5_acc', t5_acc, global_step)
                        # writer.add_scalar('Val/top_1_mean_acc', ma_t1_acc, global_step)
                        # writer.add_scalar('Val/top_5_mean_acc', ma_t5_acc, global_step)

                    # if scheduler:
                    #     scheduler.step(ma_t5_acc)
                    #     # scheduler.step(avg_val_loss)

            global_step += args.batch_size

        # Epoch is completed
        print(f"Overall average training loss: {epoch_loss / len(train_loader):.6f}")
        if args.use_tensorboard:
            writer.add_scalar('Train/ce_loss', epoch_loss / len(train_loader), global_step)

        # Plot the performance on the validation set
        # print("Validating...")
        # avg_val_loss, t_acc, t5_acc, ma_t1_acc, ma_t5_acc = validate(model, device, val_loader, args)
        avg_val_loss, t_acc, t5_acc = validate(model, device, val_loader, args)
        # print("Validation is over!")
        if args.use_tensorboard:
            writer.add_scalar('Val/ce_loss', avg_val_loss, global_step)
            writer.add_scalar('Val/top_acc', t_acc, global_step)
            writer.add_scalar('Val/top_5_acc', t5_acc, global_step)
            # writer.add_scalar('Val/top_1_mean_acc', ma_t1_acc, global_step)
            # writer.add_scalar('Val/top_5_mean_acc', ma_t5_acc, global_step)

        # print("Before scheduler...")

        # if scheduler:
        #   scheduler.step(ma_t5_acc)
        #    #scheduler.step(avg_val_loss)

        # print("Before saving model...")

        if epoch % args.aggregation_period == 0:
            # Save model
            save_path = os.path.join(args.weight_dir, f"s{args.session}_{args.model_type}_512d_{args.dataset}_"
                                                      f"{args.dataset_type}"f"_{args.dataset_version}_"
                                                      f"bs{args.batch_size}_size{args.img_size}_"
                                                      f"channels{args.in_channels}_{args.num_parties}_parties"
                                                      f"_party_{args.party_id}_e{epoch}.pt")
            # print(f"Size of the state dict: {len(model.state_dict())}")
            # print(f"==============\n{model.state_dict()}\n================")
            print(f"Saving model in: {save_path}")
            torch.save(model.state_dict(),
                       os.path.join(args.weight_dir, f"s{args.session}_{args.model_type}_512d_{args.dataset}_"
                                                     f"{args.dataset_type}"f"_{args.dataset_version}_"
                                                     f"bs{args.batch_size}_size{args.img_size}_"
                                                     f"channels{args.in_channels}_{args.num_parties}_parties"
                                                     f"_party_{args.party_id}_e{epoch}_no_noise.pt"))
            base_noise_path = os.path.join(args.weight_dir, f"s{args.session}_{args.model_type}_512d_{args.dataset}_"
                                                            f"{args.dataset_type}"f"_{args.dataset_version}_"
                                                            f"bs{args.batch_size}_size{args.img_size}_"
                                                            f"channels{args.in_channels}_{args.num_parties}_parties"
                                                            f"_party_{args.party_id}_e{epoch}_")

            denoiser = save_model(model, save_path, args.party_id, args.num_parties, True, base_noise_path)

            # # update the model with the aggregated one
            aggregated_model_path = os.path.join(args.weight_dir, f"s{args.session}_{args.model_type}_512d_"
                                                                  f"{args.dataset}_{args.dataset_type}_"
                                                                  f"{args.dataset_version}_bs{args.batch_size}_"
                                                                  f"size{args.img_size}_channels{args.in_channels}_"
                                                                  f"{args.num_parties}_parties_aggregated_model_"
                                                                  f"e{epoch}.pt")
            print(f"Reading the aggregated model from {aggregated_model_path}...")
            # while not os.path.exists(aggregated_model_path):
            #     # print(f"Waiting for {aggregated_model_path}")
            #     time.sleep(1)
            # print(f"{aggregated_model_path} is found!")
            # time.sleep(30)

            wait_until_file_ready(aggregated_model_path)

            # model = torch.load(aggregated_model_path, map_location=device).to(device)
            if args.aggregation_method == 'only_encoder':
                # model = update_encoder(model, torch.load(aggregated_model_path))  # TODO: unmask the model weights
                model = update_encoder(model, load_model(aggregated_model_path, does_denoise=True, denoiser=denoiser))
            else:
                # model.load_state_dict(torch.load(aggregated_model_path))  # TODO: unmask the model weights
                model.load_state_dict(load_model(aggregated_model_path, does_denoise=True, denoiser=denoiser))
            model.train()

        # print("Aggregated model is read!")
        print(f"=================== Epoch {epoch} is over ============================\n")

    if args.use_tensorboard:
        writer.flush()
        writer.close()


# Validation loop
def validate(model, device, val_loader, args, out=False):
    model.eval()
    val_ce_loss = 0.
    top_acc = 0.
    top_5_acc = 0.

    # print("Creating pred_per_class...")
    pred_per_class = [[] for _ in range(args.num_classes)]
    # print("pred_per_class is created!")

    tick = datetime.datetime.now()
    val_size = 0
    with torch.no_grad():
        # print("Creating diag...")
        diag = torch.eye(args.val_bs, device=device)
        # print("diag is created!")
        # print(f"Validation data loader: {len(val_loader)}")
        for idx, (data, target) in enumerate(val_loader):
            # print("Data is loading...")
            data = data.to(device, dtype=torch.float32)
            # print("Target is loading...")
            target = target.to(device, dtype=torch.int64).unsqueeze(1)

            # print(f"Cross-entropy calculation for {idx}...")

            pred, pred_rep = model(data)
            pred, pred_rep = pred.detach(), pred_rep.detach()
            val_ce_loss += F.cross_entropy(pred, target.view(-1), weight=args.ce_weights, reduction='sum').item()

            if out:
                for i in range(args.val_bs):
                    print(f"{target[i].item()},{pred[i].tolist()}")

            # some times the last batch might not be the same size
            bs = len(data)
            if bs != args.val_bs:
                diag = torch.eye(len(data), device=device)

            # print("Extra stats are being computed...")

            # extra stats
            max_pred, max_idx = torch.max(pred, dim=-1)
            top_pred, top_idx = torch.topk(pred, k=5, dim=-1)
            top_acc += torch.sum((target == max_idx) * diag).item()
            top_5_acc += np.sum([target[i] in top_idx[i] for i in range(bs)]).item()  # ... yep, quite ugly

            # TODO: support bs > 1
            if bs == 1:
                # append a ranked list of predictions to the correct class
                pred_per_class[target[0]].append(top_idx[0].cpu().numpy())

            val_size += bs

    # print("Loop over the validation samples is over!")

    top_acc = torch.true_divide(top_acc, val_size).item()
    top_5_acc = torch.true_divide(top_5_acc, val_size).item()

    # print("Predictions: \n", pred_per_class)

    # calculate the mean of the average performance per class
    # mean_average_top_1 = np.mean([np.mean([(class_idx == prediction[0]) for prediction in class_pred_list])
    #                              for class_idx, class_pred_list in enumerate(pred_per_class)])
    # mean_average_top_5 = np.mean([np.mean([(class_idx in prediction) for prediction in class_pred_list])
    #                              for class_idx, class_pred_list in enumerate(pred_per_class)])

    model.train()

    print(f"Average BCE Loss ({val_ce_loss / val_size}) during validation")
    print(f"\tTop-1 accuracy: {top_acc}, Top-5 accuracy: {top_5_acc}")
    # print(f"\tMean Top-1 accuracy: {mean_average_top_1}, Mean top-5 accuracy: {mean_average_top_5}")
    print(f"Elapsed time during validation: {(datetime.datetime.now() - tick).total_seconds():.1f}s")

    # return val_ce_loss / val_size, top_acc, top_5_acc, mean_average_top_1, mean_average_top_5
    return val_ce_loss / val_size, top_acc, top_5_acc


def preprocess(img, img_size=112, gray=False, flip=False):
    img = cv2.resize(img, (img_size, img_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if gray:
        # desired number of channels is 1, so we convert to gray
        img = A.to_gray(img)
    # else: color

    if flip:
        img = A.hflip(img)
    # else: normal

    # normalize pixel values in range [-1,1]
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    return img


def predict(models, device, image_ids, f_name, data_path):
    for model in models:
        model.eval()

    f = open(f_name, "w+")
    f.write(f"img_name;model;flip;gray;class_conf;representations\n")

    # data_path = os.path.join(args.data_dir, "GestaltMatcherDB", args.dataset_version, "gmdb_align")
    # data = os.listdir(data_path)

    tick = datetime.datetime.now()
    with torch.no_grad():
        for idx_img, img_id in enumerate(image_ids):
            print(f"Image path: {data_path}/{img_id}_rot_aligned.jpg")
            img = cv2.imread(f"{data_path}/{img_id}_rot_aligned.jpg")

            for idx, model in enumerate(models):
                for flip in [False, True]:
                    for gray in [False, True]:
                        img_p = preprocess(img,
                                           gray=gray,
                                           flip=flip
                                           ).to(device, dtype=torch.float32)

                        pred_rep = model(img_p)
                        if len(pred_rep) == 1:  # type == onnx --> 1 output: pred_rep
                            pred = [0]
                        else:
                            pred, pred_rep = pred_rep
                            pred = pred.squeeze().tolist()

                        pred_rep = F.normalize(pred_rep)
                        f.write(f"{img_id};m{idx};{int(flip)};{int(gray)};"
                                f"{pred};{pred_rep.squeeze().tolist()}\n")

    f.flush()
    f.close()

    print(f"Predictions took {datetime.datetime.now() - tick}s")
    return


def main():
    # Training settings
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {'GPU.' if use_cuda else 'CPU, as was explicitly requested, or as GPU is not available.'}")

    # Seed everything
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # When using GPU we need to set extra seeds
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    # torch.set_deterministic(True)

    print(f"Seed: {args.seed}")
    print(f"Sample random value: {torch.randn(1)}")

    # Dataset and dataloaders
    kwargs = {}
    if use_cuda:
        kwargs.update({'num_workers': (0 if args.local else args.num_workers), 'pin_memory': True})

    #########################################################################################################
    #########################################################################################################
    ##################################### TRAINING AGGREGATED MODEL #########################################
    #########################################################################################################
    #########################################################################################################
    models = []  # to hold fine-tuned Resnet50 and Resnet100 models

    dataset_train = dataset_val = None
    lookup_table = [i for i in range(args.num_classes)]
    distr = [1] * args.num_classes  # TO-DO: Finding the distribution of the classes among clients to set the weights

    # Create and get the training and validation datasets
    dataset_train, dataset_val = get_train_and_val_datasets(args.dataset, args.dataset_type, args.dataset_version,
                                                            args.img_size, args.in_channels, args.data_dir,
                                                            img_postfix='_rot_aligned',
                                                            lookup_table=lookup_table,
                                                            federated_folder=args.federated_metadata_path,
                                                            federated_suffix=f"_{args.num_parties}_parties_"
                                                                             f"party_{args.party_id}",
                                                            metadata_file_suffix=args.metadata_file_prefix)

    assert len(set(dataset_val.get_unique_labels()).difference(set(dataset_train.get_unique_labels()))) == 0, \
        f"There are syndromes which appear only in validation set for party {args.party_id}!"

    print("Saving the syndrome information...")
    np.savetxt(os.path.join(args.data_dir, "GestaltMatcherDB", args.dataset_version, "gmdb_metadata",
                            args.federated_metadata_path, f"s{args.session}_syndrome_ids_party_{args.party_id}.csv"),
               dataset_train.get_unique_labels(), delimiter=',', fmt='%i')

    print(f"Validation dataset: {dataset_val}")
    print(f"Validation dataset number of classes: {dataset_val.get_num_classes()}")
    print(f"Validation dataset size: {dataset_val.get_distribution()}")

    # Get the number of classes from the dataset
    # args.num_classes = dataset_train.get_num_classes() # commented out to have the same num of neurons in output layer

    # Try and get the dataset distribution and lookup table
    '''
    try:
        distr = dataset_train.get_distribution()
        print(f"Training dataset size: {sum(distr)}, with {len(distr)} classes and distribution: {distr}")
        print(f"Validation dataset size: {sum(dataset_val.get_distribution())}, "
              f"with distribution: {dataset_val.get_distribution()}")

        lookup_table = dataset_train.get_lookup_table()
    except:
        print("An error occurred while getting the dataset distribution and lookup table. Likely the dataset does not "
              "have an implementation for these functions.")
    '''

    # Write lookup table to file if we generated a lookup table for GMDB (i.e. when we don't supply one)
    if (lookup_table is not None) and (args.dataset != 'casia') and (args.lookup_table_path == ''):
        f = open(os.path.join(args.lookup_table_save_path,
                              f"lookup_table_{args.dataset}_{args.dataset_version}_{args.num_parties}_parties_"
                              f"party_{args.party_id}.txt"), "w+")
        f.write("index_id to disorder_id\n")
        f.write(f"{lookup_table}")
        f.flush()
        f.close()

    for paper_model in ['a', 'b']:
        # Set validation batch size to 1
        args.val_bs = 1

        # Create dataloaders
        train_loader = torch.utils.data.DataLoader(dataset_train, **kwargs, shuffle=True, batch_size=args.batch_size,
                                                   worker_init_fn=seed_worker, drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset_val, pin_memory=True, num_workers=args.num_workers,
                                                 shuffle=False, drop_last=False, worker_init_fn=seed_worker,
                                                 batch_size=args.val_bs)

        # Attempt to deal with data imbalance: inverse frequency divided by lowest frequency class (0.5 < class_weight <= 1)
        if distr is not None:
            # args.ce_weights = (torch.tensor([(sum(distr) / freq) / (sum(distr) / min(distr)) for freq in distr]).float()
            #                    .to(device)) * 0.5 + 0.5
            tmp = np.loadtxt(os.path.join(args.data_dir, "GestaltMatcherDB", args.dataset_version, "gmdb_metadata",
                                          "ce_weights_central.csv"), delimiter=',')
            args.ce_weights = torch.tensor(tmp).float().to(device)
        else:
            args.ce_weights = None
        # print(f"Weighted cross entropy weights: {args.ce_weights}")

        # If we want to reproduce paper results, by giving command-line argument `args.paper_model` == 'a' or 'b',
        # we replace some other parameters
        f_wd = c_wd = lr = c_lr = -1
        if paper_model == 'a':  # r50 mix
            args.model_type = 'glint360k_r50'
            args.batch_size = 64
            f_wd = 5e-5
            c_wd = 5e-4
            lr = 5e-4
            c_lr = 1e-3
        elif paper_model == 'b':  # r100
            args.model_type = 'glint360k_r100'
            args.batch_size = 128
            f_wd = 0.
            c_wd = 5e-4
            lr = 1e-3
            c_lr = 1e-3
        else:
            print(f"Undefined model type: Model type '{paper_model}' is not known!")
            exit(-1)

        # Create model
        model = MyArcFace(args.num_classes, dataset_base=os.path.join(args.weight_dir, f'{args.model_type}.onnx'),
                          device=device, freeze=True).to(device)
        print(f"Created {'frozen ' if args.freeze else ''}{args.model_type} model with {args.in_channels} in channel"
              f"{'s' if args.in_channels > 1 else ''}, 512d feature dimensionality and {args.num_classes} classes")

        # Set log intervals
        args.log_interval = 1000  # set the log interval to the default value
        args.val_interval = 10000  # set the validation interval to the default value
        args.log_interval = args.log_interval // args.batch_size
        args.val_interval = args.val_interval // args.batch_size

        ## Continue training/testing:
        # model.load_state_dict(torch.load(f"saved_models/<saved weights>.pt", map_location=device))

        ## Init optimizer
        # We seperate the optimizer for cnn-base and classifier
        optimizer = optim.Adam([
            {'params': model.base.parameters()},
            {'params': model.features.parameters(), 'weight_decay': f_wd if f_wd != -1 else 5e-5},
            {'params': model.classifier.parameters(),
             'weight_decay': c_wd if c_wd != -1 else 5e-4, 'lr': c_lr if c_lr != -1 else 1e-3
             }
        ], lr=lr if lr != -1 else args.lr, weight_decay=0.)

        # Init scheduler
        scheduler = lr_sched.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True, min_lr=1e-5, mode="max", patience=5,
                                               threshold=5e-4)

        ## Call explicit model weight initialization (only do this for the base task, if at all)
        # model.init_layer_weights()

        # Run training loop
        train(args, model, device, train_loader, optimizer, val_loader=val_loader, scheduler=scheduler)

        ## Run final validation step with extra output
        # validate(model, device, val_loader, args, out=True)

        # Save entire model
        model_path = os.path.join(args.model_dir, f"s{args.session}_{args.model_type}_512d_{args.dataset}"
                                                  f"_{args.dataset_type}_{args.dataset_version}_bs{args.batch_size}"
                                                  f"_size{args.img_size}_channels{args.in_channels}_last_model"
                                                  f"_{args.num_parties}_parties_party_{args.party_id}.pth")
        if args.epochs % args.aggregation_period != 0:
            denoiser = save_model(model, model_path, args.party_id, args.num_parties, is_noisy=True)

            tmp_path = os.path.join(args.model_dir, f"s{args.session}_{args.model_type}_512d_{args.dataset}"
                                                    f"_{args.dataset_type}_{args.dataset_version}_bs{args.batch_size}"
                                                    f"_size{args.img_size}_channels{args.in_channels}_last_model"
                                                    f"_{args.num_parties}_parties_aggregated.pth")

            wait_until_file_ready(tmp_path)
            model.load_state_dict(load_model(tmp_path, does_denoise=True, denoiser=denoiser))

        save_model(model, model_path)

        models.append(model)

    #########################################################################################################
    #########################################################################################################
    ################################# PREDICTION VIA AGGREGATED MODEL #######################################
    #########################################################################################################
    #########################################################################################################
    print("data variable will be created...")
    # data = os.listdir(os.path.join(args.data_dir, "GestaltMatcherDB", args.dataset_version, "gmdb_align"))

    # original r100
    models.append(convert(os.path.join(args.weight_dir, "glint360k_r100.onnx")).to(device))
    print("Model 3 is created.")

    # get the ids of test images
    test_images = pd.read_csv(os.path.join(args.data_dir, "GestaltMatcherDB", args.dataset_version, "gmdb_metadata",
                                           args.federated_metadata_path, f"{args.metadata_file_prefix}gmdb_test_images_"
                                                                         f"{args.dataset_version}_{args.num_parties}_"
                                                                         f"parties_party_{args.party_id}.csv"))

    print("Prediction time of frequent samples...")
    predict(models, device, pd.concat((pd.concat((dataset_train.get_image_ids(), dataset_val.get_image_ids())),
                                       test_images.image_id)),
            os.path.join(args.encoding_dir, f"s{args.session}_federated_encodings_{args.num_parties}_parties_party_"
                                            f"{args.party_id}.csv"),
            os.path.join(args.data_dir, "GestaltMatcherDB", args.dataset_version, "gmdb_align"))

    print("Prediction time of rare samples...")
    rare_cases = pd.read_csv(os.path.join(args.data_dir, "GestaltMatcherDB", args.dataset_version, "gmdb_metadata",
                                          args.federated_metadata_path, f"gmdb_rare_images_{args.dataset_version}_"
                                                                        f"{args.num_parties}_parties_party_"
                                                                        f"{args.party_id}.csv"))
    predict(models, device, rare_cases.image_id,
            os.path.join(args.encoding_dir,
                         f"s{args.session}_federated_rare_encodings_{args.num_parties}_parties_party_"
                         f"{args.party_id}.csv"),
            os.path.join(args.data_dir, "GestaltMatcherDB", args.dataset_version, "gmdb_align"))

    #########################################################################################################
    #########################################################################################################
    ##################################### EVALUATION OF THE MODELS ##########################################
    #########################################################################################################
    #########################################################################################################


if __name__ == '__main__':
    main()
