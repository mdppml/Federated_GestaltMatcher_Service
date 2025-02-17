#!/usr/bin/env python
# coding: utf-8

# # FLAKE integration and patient inference tests

# In[1]:


import os
import random

import numpy as np
import pandas as pd
import json
from scipy.linalg import sqrtm

from lib.fed_utils import parse_args

args = parse_args()

seed = 36
k_higher_dimension = 600
dataset_version = 'v1.0.3'

encoding_dir = '/home/uenal/GestaltMatcher-Arc/encodings/'
metadata_path = '/home/uenal/GestaltMatcher-Arc/data/GestaltMatcherDB/v1.0.3/gmdb_metadata/'
federated_metadata_path = '/home/uenal/GestaltMatcher-Arc/data/GestaltMatcherDB/v1.0.3/gmdb_metadata/federated_metadata/'
result_path = '/home/uenal/GestaltMatcher-Arc/results/'

if args.local:
    encoding_dir = '/Users/aliburak/Projects/Federated_GestaltMatcher/encodings/'
    metadata_path = '/Users/aliburak/Projects/Federated_GestaltMatcher/data/gmdb_metadata/'
    federated_metadata_path = '/Users/aliburak/Projects/Federated_GestaltMatcher/data/gmdb_metadata/federated_metadata/'
    result_path = '/Users/aliburak/Federated_GestaltMatcher/results/'

masked_data = {}

model_info = ['m0', 'm0', 'm0', 'm0', 'm1', 'm1', 'm1', 'm1', 'm2', 'm2', 'm2', 'm2']
flip_info = [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
gray_info = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

num_of_models = len(model_info)

debugging = False

# for session_suffix in ['s17_', 's18_', 's19_', 's20_', 's21_', 's22_', 's25_', 's24_', 's25_']:
for session_suffix in [f"s{args.session}_"]:
    # ## Reading encodings and labels
    test_all_fed_encodings = []
    fed_representations = []
    fed_rare_representations = []
    n_samples = 0

    for party_id in range(1, args.num_parties + 1):
        print(f"Party {party_id}: ", end='')

        # read the encodings of the samples
        representation_df = pd.read_csv(
            os.path.join(encoding_dir, f"{session_suffix}federated_encodings_{args.num_parties}_parties_"
                                       f"party_{party_id}.csv"), delimiter=";")
        print(f"Rep: {len(representation_df) / num_of_models}")
        rare_rep_df = pd.read_csv(
            os.path.join(encoding_dir, f"{session_suffix}federated_rare_encodings_{args.num_parties}_parties_"
                                       f"party_{party_id}.csv"), delimiter=";")

        representation_df = pd.concat([representation_df, rare_rep_df], axis=0, ignore_index=True)

        print(len(representation_df) / num_of_models)
        assert (
                       len(representation_df) % num_of_models) == 0, f"The number of representations is not divisible by the number of models without remainder. # representations: {len(representation_df)} - # models: {num_of_models}"
        n_samples += len(representation_df) // num_of_models

        representation_df = representation_df.groupby("img_name").agg(lambda x: list(x)).reset_index()

        representation_df.representations = representation_df.representations.apply(
            lambda x: np.array([json.loads(i) for i in x]))
        representation_df.class_conf = representation_df.class_conf.apply(lambda x: [json.loads(i) for i in x])

        # read the metadata
        metadata_df = pd.concat(
            [pd.read_csv(
                os.path.join(federated_metadata_path, f"{args.metadata_file_prefix}gmdb_{ds}_images_{dataset_version}_"
                                                      f"{args.num_parties}_parties_party_{party_id}.csv"))
                for ds in ["train", "val", "test"]], axis=0, ignore_index=True)
        print(len(metadata_df))

        rare_metadata = pd.read_csv(os.path.join(federated_metadata_path, f"gmdb_rare_images_{dataset_version}_"
                                                                          f"{args.num_parties}_parties_party_{party_id}.csv"))

        metadata_df = pd.concat([metadata_df, rare_metadata], axis=0, ignore_index=True)

        print(metadata_df.shape)
        print(representation_df.shape)
        assert metadata_df.shape[0] == representation_df.shape[
            0], f"Mismatch between the metadata and the encodings for party {party_id}! {len(metadata_df)} vs {len(representation_df)}"

        representation_df['rep2meta'] = representation_df.img_name.map(
            pd.Series(metadata_df.index, index=metadata_df.image_id))
        representation_df['label'] = metadata_df.label.values[representation_df.rep2meta.values]
        representation_df['subject'] = metadata_df.subject.values[representation_df.rep2meta.values]
        fed_representations.append(representation_df)

    print(f"Total number of samples: {n_samples}")

    # In[9]:

    # concatenate individual federated representations
    all_representations = pd.concat(fed_representations, ignore_index=True)

    # initialize a common seed among clients
    # Seed everything
    np.random.seed(seed)
    random.seed(seed)

    # retrieve encodings as a list
    encoding_list = [np.stack(fed_representations[p].representations.values, axis=1) for p in
                     range(0, args.num_parties)]
    print(f"encoding_list[0]: {encoding_list[0].shape}")

    #  generate common mask
    N = np.random.rand(k_higher_dimension, encoding_list[0].shape[2])
    N_sqrt = np.real(sqrtm(N @ N.T))
    print(f"N: {N.shape}")

    # masking the encodings
    masked_encoding_list = []
    for p in range(args.num_parties):
        # left inverse
        L = np.linalg.pinv(N)

        tmp = []
        # mask each model's encodings
        for m in range(encoding_list[0].shape[0]):
            tmp.append(encoding_list[p][m] @ L @ N_sqrt)
        masked_encoding_list.append(np.stack(tmp, axis=0))

    # compute parts of Gram matrices
    gram_matrix_parts = []
    for m in range(masked_encoding_list[0].shape[0]):
        tmp = []
        for p1 in range(args.num_parties):
            for p2 in range(args.num_parties):
                tmp.append(masked_encoding_list[p1][m] @ masked_encoding_list[p2][m].T)
        gram_matrix_parts.append(tmp)


    def form_gram_matrix(gms, num_parties):
        return np.concatenate(
            [np.concatenate(gms[p * num_parties:(p + 1) * num_parties], axis=1) for p in range(num_parties)], axis=0)


    # In[16]:

    # form Gram matrices
    print(len(gram_matrix_parts))
    print(len(gram_matrix_parts[0]))

    gram_matrices = [form_gram_matrix(gram_matrix_parts[m], args.num_parties) for m in range(num_of_models)]

    # In[19]:

    #  TEST: correctness of Gram matrices
    if debugging:
        concat_encodings = np.concatenate([encoding_list[m] for m in range(args.num_parties)], axis=1)
        for m in range(num_of_models):
            assert np.allclose((concat_encodings[m] @ concat_encodings[m].T),
                               gram_matrices[m]), f"Not close for Gram matrix {m}!"


    def compute_distance_matrix(gm):
        assert gm.shape[0] == gm.shape[1], f"Gram matrix's dimensions do not match! The shape is {gm.shape}"
        norms = np.sqrt(np.diag(gm))
        norm_matrix = np.outer(norms, norms)
        cosine_similarity_matrix = np.divide(gm, norm_matrix, where=norm_matrix != 0)
        cosine_similarity_matrix[norm_matrix == 0] = 0
        cosine_distances = 1 - cosine_similarity_matrix

        return cosine_distances


    distance_matrices = np.stack([compute_distance_matrix(gram_matrices[m]) for m in range(len(gram_matrices))], axis=0)
    distance_matrix = np.mean(distance_matrices, axis=0)


    def evaluate(gallery_df, test_df, distance_matrix, num_classes: int = args.num_classes,
                 warning_messages: bool = False):
        '''
        Evaluate the performance of the model using the provided test and gallery image dataframes, and the distance matrix
        '''
        assert test_df.label.value_counts().size <= gallery_df.label.value_counts().size, f"There are more labels in test_df than gallery_df! {test_df.label.value_counts().size} vs {gallery_df.label.value_counts().size}"
        if test_df.label.value_counts().size < gallery_df.label.value_counts().size and warning_messages:
            print("WARNING: There are less labels in test_df than gallery_df in evaluate!")

        # extract test2gallery part from the distance matrix
        dist_test2gallery = distance_matrix[test_df.test2fed.values][:, gallery_df.gal2fed.values]
        #  print(dist_test2gallery.shape)

        # get the sorting index vector for each test sample
        sorting_ind = np.argsort(dist_test2gallery, axis=1)
        #  print(sorting_ind.shape)

        # sort the labels of the gallery samples for each test sample
        sorted_synd = gallery_df.label.values[sorting_ind]
        #  print(sorted_synd.shape)

        # get the sorted unique occurrences of labels
        guessed_all = np.empty((len(test_df), gallery_df.label.nunique()))
        #  print(guessed_all.shape)
        for s in range(len(test_df)):
            guessed_all[s] = sorted_synd[s][np.sort(np.unique(sorted_synd[s], return_index=True)[1])]

        # Top-n performance
        test_synd_ids = test_df.label.values
        corr = np.zeros(4)  # 4 because [1,5,10,30]
        results = np.zeros((4, num_classes, 2), dtype=int)  # to check the distribution of correctly predicted labels
        # result_index_offset = min(test_df.label)
        acc_per = []
        for i, n in enumerate([1, 5, 10, 30]):
            for idx in range(len(test_synd_ids)):
                # guessed_all[np.sort(np.unique(guessed_all, return_index=True)[1])]
                top_n_guessed = guessed_all[idx, 0:n]
                if test_synd_ids[idx] in top_n_guessed:
                    corr[i] += 1
                    # results[i][test_synd_ids[idx] - result_index_offset][0] += 1
                    results[i][test_synd_ids[idx]][0] += 1
                else:
                    # results[i][test_synd_ids[idx] - result_index_offset][1] += 1
                    results[i][test_synd_ids[idx]][1] += 1

            # Bit cluttered.., but this calculates the top-n per syndrome accuracy
            acc_per.append(sum([sum(tl in g[0:n] for g in guessed_all[np.where(test_synd_ids == tl)[0]]) / len(
                np.where(test_synd_ids == tl)[0]) for tl in list(set(test_synd_ids))]) / len(
                list(set(test_synd_ids))))

        return acc_per, results


    def save_confusion_matrix(arr, base_fn, k_list=None):
        if k_list is None:
            k_list = [1, 5, 10, 30]
        for i, k in enumerate(k_list):
            np.savetxt(f"{base_fn}_top_{k}.csv", arr[i], delimiter=',', fmt='%d')


    # ## Test: Frequent - Gallery: Frequent
    # read frequent gallery and test ids
    gallery_df = pd.read_csv(os.path.join(metadata_path, f"gmdb_frequent_gallery_images_{dataset_version}.csv"))
    test_df = pd.read_csv(os.path.join(metadata_path, f"gmdb_frequent_test_images_{dataset_version}.csv"))

    # mapping of gallery and test images to the order of images in distance matrix
    # patient_metadata_index_mapping = pd.Series(patient_metadata.index, index=patient_metadata.image_id)
    # gallery_df['index_mapping'] = gallery_df.image_id.map(patient_metadata_index_mapping)
    # test_df['index_mapping'] = test_df.image_id.map(patient_metadata_index_mapping)

    # mapping of gallery and test images to the order of images in distance matrix
    patient_metadata_index_mapping = pd.Series(all_representations.index, index=all_representations.img_name)
    gallery_df['gal2fed'] = gallery_df.image_id.map(patient_metadata_index_mapping)
    test_df['test2fed'] = test_df.image_id.map(patient_metadata_index_mapping)

    ff_acc_per, ff_results = evaluate(gallery_df, test_df, distance_matrix)
    save_confusion_matrix(ff_results, f"{session_suffix}_ff_confusion_matrix")

    # ## Test: Rare - Gallery: Rare
    rare_gallery = pd.read_csv(os.path.join(metadata_path, f"gmdb_rare_gallery_images_{dataset_version}.csv"))
    rare_test = pd.read_csv(os.path.join(metadata_path, f"gmdb_rare_test_images_{dataset_version}.csv"))

    rare2rep_mapping = pd.Series(all_representations.index, index=all_representations.img_name)
    rare_gallery['gal2fed'] = rare_gallery.image_id.map(rare2rep_mapping)
    rare_test['test2fed'] = rare_test.image_id.map(rare2rep_mapping)

    num_splits = max(rare_gallery.split) + 1
    rr_acc_per_list = []
    for f in range(num_splits):
        # extract the gallery and test dfs for this fold
        tmp_gallery_df = rare_gallery[rare_gallery.split == f]
        tmp_test_df = rare_test[rare_test.split == f]
        eval_result, rr_results = evaluate(tmp_gallery_df, tmp_test_df, distance_matrix,
                                           num_classes=max(max(rare_gallery.label), max(rare_test.label)) + 1)
        rr_acc_per_list.append(eval_result)
        save_confusion_matrix(rr_results, f"{session_suffix}_rr_split_{f}_confusion_matrix")

    rr_acc_per_list = np.array(rr_acc_per_list)

    # ## Test : Frequent - Gallery: Frequent + Rare
    num_splits = max(rare_gallery.split) + 1
    ffr_acc_per_list = []
    for f in range(num_splits):
        # extract the gallery and test dfs for this fold
        tmp_gallery_df = pd.concat([gallery_df, rare_gallery[rare_gallery.split == f]], axis=0, ignore_index=True)
        tmp_gallery_df.split = tmp_gallery_df.split.fillna(int(f))
        eval_result, ffr_results = evaluate(tmp_gallery_df, test_df, distance_matrix)
        ffr_acc_per_list.append(eval_result)
        save_confusion_matrix(ffr_results, f"{session_suffix}_ffr_split_{f}_confusion_matrix")

    ffr_acc_per_list = np.array(ffr_acc_per_list)

    # ## Test: Rare - Gallery: Frequent + Rare
    num_splits = max(rare_gallery.split) + 1
    rfr_acc_per_list = []
    for f in range(num_splits):
        # extract the gallery and test dfs for this fold
        tmp_gallery_df = pd.concat([gallery_df, rare_gallery[rare_gallery.split == f]], axis=0, ignore_index=True)
        tmp_test_df = rare_test[rare_test.split == f]
        eval_result, rfr_results = evaluate(tmp_gallery_df, tmp_test_df, distance_matrix,
                                            num_classes=max(max(rare_gallery.label), max(rare_test.label)) + 1)
        rfr_acc_per_list.append(eval_result)
        save_confusion_matrix(rfr_results, f"{session_suffix}_rfr_split_{f}_confusion_matrix")

    rfr_acc_per_list = np.array(rfr_acc_per_list)

    # ## Overall results as a single table
    print('===========================================================')
    print('---------   test: Frequent, gallery: Frequent    ----------')
    print('|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|')
    print('|{}|{}    |{}   |{:.2f} |{:.2f} |{:.2f} |{:.2f} |'.format("GMDB-frequent",
                                                                     len(gallery_df),
                                                                     len(test_df),
                                                                     (ff_acc_per[0]) * 100,
                                                                     (ff_acc_per[1]) * 100,
                                                                     (ff_acc_per[2]) * 100,
                                                                     (ff_acc_per[3]) * 100))
    print('---------       test: Rare, gallery: Rare        ----------')
    print('|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|')
    print('|{}|{}   |{} |{:.2f} |{:.2f} |{:.2f} |{:.2f} |'.format("GMDB-rare    ",
                                                                  len(rare_gallery) / num_splits,
                                                                  len(rare_test) / num_splits,
                                                                  np.mean(rr_acc_per_list[:, 0]) * 100,
                                                                  np.mean(rr_acc_per_list[:, 1]) * 100,
                                                                  np.mean(rr_acc_per_list[:, 2]) * 100,
                                                                  np.mean(rr_acc_per_list[:, 3]) * 100))
    print('---------   test: Frequent, gallery: Frequent+Rare   ----------')
    print('|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|')
    print('|{}|{}  |{} |{:.2f} |{:.2f} |{:.2f} |{:.2f} |'.format("GMDB-rare    ",
                                                                 (len(gallery_df) + len(rare_gallery) / num_splits),
                                                                 len(test_df),
                                                                 np.mean(ffr_acc_per_list[:, 0]) * 100,
                                                                 np.mean(ffr_acc_per_list[:, 1]) * 100,
                                                                 np.mean(ffr_acc_per_list[:, 2]) * 100,
                                                                 np.mean(ffr_acc_per_list[:, 3]) * 100))
    print('---------   test: Rare, gallery: Frequent+Rare   ----------')
    print('|Test set     |Gallery |Test  |Top-1 |Top-5 |Top-10|Top-30|')
    print('|{}|{}  |{} |{:.2f} |{:.2f} |{:.2f} |{:.2f} |'.format("GMDB-rare    ",
                                                                 (len(gallery_df) + len(rare_gallery) / num_splits),
                                                                 len(rare_test) / num_splits,
                                                                 np.mean(rfr_acc_per_list[:, 0]) * 100,
                                                                 np.mean(rfr_acc_per_list[:, 1]) * 100,
                                                                 np.mean(rfr_acc_per_list[:, 2]) * 100,
                                                                 np.mean(rfr_acc_per_list[:, 3]) * 100))

    ff_acc_results = [str(ff_acc_per[i] * 100) for i in range(len(ff_acc_per))]
    rr_acc_results = (np.mean(rr_acc_per_list, axis=0) * 100).astype(str)
    ffr_acc_results = (np.mean(ffr_acc_per_list, axis=0) * 100).astype(str)
    rfr_acc_results = (np.mean(rfr_acc_per_list, axis=0) * 100).astype(str)

    with open(os.path.join(result_path, f"{session_suffix}results.csv"), 'w+') as f:
        f.write("Test set,Gallery set,Gallery size,Test size,Top-1,Top-5,Top-10,Top-30\n")
        f.write(f"Frequent,Frequent,{len(gallery_df)},{len(test_df)},{','.join(ff_acc_results)}\n")
        f.write(
            f"Rare,Rare,{len(rare_gallery) / num_splits},{len(rare_test) / num_splits},{','.join(rr_acc_results)}\n")
        f.write(
            f"Frequent,Frequent+Rare,{len(gallery_df) + len(rare_gallery) / num_splits},{len(test_df)},{','.join(ffr_acc_results)}\n")
        f.write(
            f"Rare,Frequent+Rare,{len(gallery_df) + len(rare_gallery) / num_splits},{len(rare_test) / num_splits},{','.join(rfr_acc_results)}\n")
