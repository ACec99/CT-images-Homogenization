"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import shutil
from collections import OrderedDict
from tqdm import tqdm

import SimpleITK as sitk
import json
import pandas as pd
import csv
import ast
from scipy import stats
import radiomics
from radiomics import featureextractor

import numpy as np
import torch
import gc
from skimage import filters

from metrics.fid import calculate_fid_for_all_tasks
#from metrics.lpips import calculate_lpips_given_images
from core.data_loader import get_eval_loader
from core import utils

import datetime
import logging


# Configure logging
logging.basicConfig(level=logging.INFO)

def check_mask_for_labels(mask):
    mask_array = sitk.GetArrayFromImage(mask)
    if not np.sum(mask_array) == 0:
        logging.warning('No labels found in this mask (i.e. nothing is segmented)!')
        return False
    else:
        #logging.info("Mask contains labels.")
        return True


@torch.no_grad()
def calculate_metrics(nets, args, step, mode, domains, edge_loss):
    print('Calculating evaluation metrics...')
    assert mode in ['latent', 'reference']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coords_dict, reference_dims = utils.extract_metadata(args)

    #domains = os.listdir(args.val_img_dir)
    #domains.sort()

    """domains = utils.sort_domains_by_size(args.val_img_dir)
    domains = domains[:args.num_domains]
    domains.sort()"""

    #num_domains = len(domains)
    #print('Number of domains: %d' % num_domains)

    extractor = featureextractor.RadiomicsFeatureExtractor()

    if 'shape2D' not in extractor.enabledFeatures:
        extractor.enabledFeatures['shape2D'] = []
    extractor.settings['force2D'] = True

    # disabilitare le feature classes che non ci interessano
    """extractor.enableFeatureClassByName('shape2D', enabled=False)
    extractor.enableFeatureClassByName('shape', enabled=False)"""

    # abilitare solamente l'estrazione delle feature di interesse
    #extractor.enableFeaturesByName(**enabledFeatures)

    if mode == 'latent':
        T_test_filename = os.path.join(args.eval_dir, 'valid_T_test_%.5i.csv' % step)
        t_test_dataframe = pd.DataFrame(columns=['feature_name', 'source_domain', 'domain_mapped', 't-test'])

    columns_names = ['feature_name', 'domain', 'values']
    feature_table_path = os.path.join(args.current_expr_dir, '%01d_feature_per_volume.csv' % (args.num_domains))
    features_table = pd.read_csv(feature_table_path, header=None, names=columns_names)
    # TODO DA DECOMMENTARE LA PROSSIMA RIGA
    #features_table['values'] = features_table['values'].apply(ast.literal_eval)

    edge_per_dom_translation = {}

    #lpips_dict = OrderedDict()
    for trg_idx, trg_domain in enumerate(domains):

        # calcola il dizionario con chiavi le radiomic features ed associata a ciascuna di esse la lista dei relativi valori per ogni volume

        features_table_trg_domain = features_table.loc[features_table['domain'] == trg_domain].copy()

        features_real_dict = {}

        # TODO COMMENTARE LE PROSSIME QUATTRO RIGHE
        for idx_table, row in features_table_trg_domain.iterrows():
            feature_name = row['feature_name']
            #if feature_name in relevant_feature_complete_names:
            features_real_dict[feature_name] = ast.literal_eval(row['values'])

        #print(features_real_dict)

        src_domains = [x for x in domains if x != trg_domain]

        if mode == 'reference':
            path_ref = os.path.join(args.val_img_dir, trg_domain)
            loader_ref = get_eval_loader(root=path_ref,
                                         lung_coords=coords_dict,
                                         ref_dims=reference_dims,
                                         min_bound=args.min_bound,
                                         max_bound=args.max_bound,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         drop_last=True,
                                         resize=args.resize_bool)

        for src_idx, src_domain in enumerate(src_domains):
            edge_values = []
            path_src = os.path.join(args.val_img_dir, src_domain)
            loader_src = get_eval_loader(root=path_src,
                                         lung_coords=coords_dict,
                                         ref_dims=reference_dims,
                                         min_bound=args.min_bound,
                                         max_bound=args.max_bound,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         drop_last=True,
                                         resize=args.resize_bool)

            ### CREAZIONE DELLA COPPIA DI DOMINI DA COMPARARE ###
            task = '%s_to_%s' % (src_domain, trg_domain)
            path_fake_tifs = os.path.join(args.tifs_dir, task)
            if os.path.exists(path_fake_tifs):
                shutil.rmtree(path_fake_tifs, ignore_errors=True)
            os.makedirs(path_fake_tifs)
            volumes_fake_dict = {}
            for i, x_src_complete in enumerate(tqdm(loader_src, total=len(loader_src))):
                torch.cuda.empty_cache()
                x_src , patient_src = x_src_complete
                N = x_src.size(0)
                x_src = x_src.to(device)
                y_trg = torch.tensor([trg_idx] * N).to(device)
                masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

                # generate 10 outputs from the same input
                if mode == 'latent':
                    z_trg = torch.randn(N, args.latent_dim).to(device)
                    s_trg = nets.mapping_network(z_trg, y_trg)
                else:
                    try:
                        x_ref , patient_ref = next(iter_ref).to(device)
                    except:
                        iter_ref = iter(loader_ref)
                        x_ref , patient_ref = next(iter_ref)

                    if x_ref.size(0) > N:
                        x_ref = x_ref[:N]
                    x_ref = x_ref.to(device)
                    s_trg = nets.style_encoder(x_ref, y_trg)

                x_fake = nets.generator(x_src, s_trg, masks=masks)
                binary_fake_list = []
                binary_real_list = []

                for k in range(N):

                    # save generated images to calculate FID later
                    filename_single_tif = os.path.join(
                        path_fake_tifs,
                        '%.4i_%.2i.tif' % (i, k))
                    utils.save_tif(x_fake[k], args.max_bound, args.min_bound, ncol=1,
                                   filename=filename_single_tif)


                    patientID = patient_src[k]

                    # bed removal
                    slice_fake = ((x_fake[k, 1, :, :].unsqueeze(0)).cpu()).numpy()
                    slice_fake = utils.RemoveBed(slice_fake)

                    slice_real = ((x_src[k, 1, :, :].unsqueeze(0)).cpu()).numpy()
                    slice_real = utils.RemoveBed(slice_real)

                    slice_fake = slice_fake.squeeze(0)
                    slice_real = slice_real.squeeze(0)

                    # save the new slice in correspondence to the owner (patient) inside a dictionary, in order to rebuild the volume
                    if patientID not in volumes_fake_dict:
                        volumes_fake_dict[patientID] = [slice_fake] #[x_fake[k, 1, :, :]]
                    else:
                        volumes_fake_dict[patientID].append(slice_fake) #(x_fake[k, 1, :, :])

                    # extraction of lungs (masks)
                    lung_fake = utils.segment_lung(slice_fake)
                    lung_real = utils.segment_lung(slice_real)

                    # APPLICATION OTSU ALGORITHM #
                    ## Application of Otsu's threshold
                    threshold_value_fake = filters.threshold_otsu(lung_fake)
                    threshold_value_real = filters.threshold_otsu(lung_real)
                    ## Thresholding the image to segment it
                    binary_fake = lung_fake > threshold_value_fake
                    binary_real = lung_real > threshold_value_real

                    # Append the binary images to the respective lists
                    binary_fake_list.append(torch.tensor(binary_fake, dtype=torch.int32))
                    binary_real_list.append(torch.tensor(binary_real, dtype=torch.int32))

                # Stack the lists into tensors
                binary_fake_tensor = torch.stack(binary_fake_list).to(device)  # Shape: (N, H, W)
                #binary_fake_tensor = binary_fake_tensor.unsqueeze(1) # Shape: (N, C, H, W)
                binary_real_tensor = torch.stack(binary_real_list).to(device)  # Shape: (N, H, W)
                #binary_real_tensor = binary_real_tensor.unsqueeze(1)  # Shape: (N, C, H, W)
                #slice_edge, _, _ = edge_loss(tensor_real, tensor_fake)
                similarity_measure = edge_loss(binary_fake_tensor, binary_real_tensor)
                similarity_measure = similarity_measure.mean()
                #edge_values.append(slice_edge.item())
                #print(f"slice_edge is: {similarity_measure}")
                edge_values.append(similarity_measure)

            #edge_values_mean = np.mean(np.array(edge_values))
            stacked_edge_values = torch.stack(edge_values)
            print(f"stacked_edge_values is: {stacked_edge_values}, with the shape: {stacked_edge_values.shape}")
            # Compute the average
            edge_values_mean = torch.mean(stacked_edge_values, dim=0)
            edge_per_dom_translation[task] = edge_values_mean.item()

            torch.cuda.empty_cache()
            ### RADIOMIC FEATURE COMPARISONS ###
            if mode == 'latent':
                print("Calculating radiomic features...")

                features_fake_dict = {}

                #print("il volumes_fake_dict è:", volumes_fake_dict)

                for patient in volumes_fake_dict:

                    features_volume_dict = {}
                    volume_fake = volumes_fake_dict[patient]

                    #print(len(volume_fake))

                    for slice_fake in volume_fake:
                        #slice_fake = slice_fake.squeeze()
                        #print(slice_fake.shape)
                        slice_fake = utils.denormalize(slice_fake, args.max_bound, args.min_bound)
                        #print(slice_fake.shape)
                        #slice_fake = (slice_fake.cpu()).numpy()
                        img_fake = sitk.GetImageFromArray(slice_fake)
                        mask = img_fake > -1024
                        #mask_check = check_mask_for_labels(mask)
                        #if mask_check:
                        try:
                            features_slice = extractor.execute(img_fake, mask)
                            for feature_name in features_slice.keys():
                                if feature_name.startswith("original_"):
                                    if feature_name not in features_volume_dict:
                                        features_volume_dict[feature_name] = [features_slice[feature_name].item()]
                                    else:
                                        features_volume_dict[feature_name].append(features_slice[feature_name].item())
                        except ValueError as e:
                            print(f"Error extracting features for patient {patient}: {str(e)}")
                            continue

                    for feature in features_volume_dict:
                        volume_feature = np.mean(np.array(features_volume_dict[feature]))
                        if feature not in features_fake_dict:
                            features_fake_dict[feature] = [volume_feature]
                        else:
                            features_fake_dict[feature].append(volume_feature)

                 #print(features_fake_dict)

                print("Calculating T-test...")

                for feature in features_fake_dict:
                    real_domain = features_real_dict[feature]
                    fake_domain = features_fake_dict[feature]
                    #_, p_value = stats.ttest_ind(real_domain, fake_domain)
                    _, p_value = stats.ttest_ind(real_domain, fake_domain, equal_var=False)
                    #print("ho calcolato il p_value e lo sto salvando")
                    new_line_dataframe = {'feature_name': feature, 'source_domain': src_domain , 'domain_mapped':trg_domain, 't-test': p_value}
                    t_test_dataframe = t_test_dataframe.append(new_line_dataframe, ignore_index=True)
        # delete dataloaders
        del loader_src
        if mode == 'reference':
            del loader_ref
            del iter_ref

    df = pd.DataFrame(list(edge_per_dom_translation.items()), columns=['Domain Translation', 'Edge Mean'])
    # Salva il DataFrame in un file CSV
    edge_csv_filename = os.path.join(args.eval_dir, 'edge_mean_per_domain_%.5i.csv' % step)
    df.to_csv(edge_csv_filename, index=False)

    print("File CSV creato con successo!")

    if mode == 'latent':
        t_test_dataframe.to_csv(T_test_filename, index=False)

    # calculate and report fid values
    calculate_fid_for_all_tasks(args, domains, step=step, mode=mode, lung_coords=coords_dict, ref_dims=reference_dims)

    folder_tifs_list = os.listdir(args.tifs_dir)
    for folder_tifs in folder_tifs_list:
        folder_tifs_path = os.path.join(args.tifs_dir, folder_tifs)
        shutil.rmtree(folder_tifs_path, ignore_errors=True)

    torch.cuda.empty_cache()

def obtain_samples(nets, args, domains):
    print('Extracting images trg-translated...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coords_dict, reference_dims = utils.extract_metadata(args)
    # shutil.rmtree(path_fake_imgs, ignore_errors=True)
    # shutil.rmtree(path_fake_grid, ignore_errors=True)
    for trg_idx, trg_domain in enumerate(domains):
        src_domains = [x for x in domains if x != trg_domain]
        path_ref = os.path.join(args.val_img_dir, trg_domain)
        loader_ref = get_eval_loader(root=path_ref,
                                     lung_coords=coords_dict,
                                     ref_dims=reference_dims,
                                     min_bound=args.min_bound,
                                     max_bound=args.max_bound,
                                     img_size=args.img_size,
                                     batch_size=args.val_batch_size,
                                     drop_last=True,
                                     resize=args.resize_bool)
        try:
            x_ref, patient_ref = next(iter_ref).to(device)
        except:
            iter_ref = iter(loader_ref)
            x_ref, patient_ref = next(iter_ref)
        x_ref = x_ref.to(device)
        for src_idx, src_domain in enumerate(src_domains):
            path_src = os.path.join(args.val_img_dir, src_domain)
            loader_src = get_eval_loader(root=path_src,
                                         lung_coords=coords_dict,
                                         ref_dims=reference_dims,
                                         min_bound=args.min_bound,
                                         max_bound=args.max_bound,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size,
                                         drop_last=True,
                                         resize=args.resize_bool)

            task = '%s_to_%s' % (src_domain, trg_domain)
            #x_src, patient_src = next(iter(loader_src))
            try:
                x_src, patient_src = next(iter_src).to(device)
            except:
                iter_src = iter(loader_src)
                x_src, patient_src = next(iter_src)
            N = x_src.size(0)
            x_src = x_src.to(device)
            #print(N)
            y_trg = torch.tensor([trg_idx] * N).to(device)
            z_trg = torch.randn(N, args.latent_dim).to(device)
            s_trg = nets.mapping_network(z_trg, y_trg)
            x_fake = nets.generator(x_src, s_trg, masks=None)
            filename_img = os.path.join(args.eval_dir, task + ".png")
            x_ref_1 = x_ref[0].unsqueeze(0)
            x_fake_1 = x_fake[0].unsqueeze(0)
            print("la dimensione di x_ref è", x_ref_1.shape)
            print("la dimensione di x_src è", x_fake_1.shape)
            couple_img = torch.cat([x_ref_1, x_fake_1], dim=0)
            print("la dimensione di couple_img è", couple_img.shape)
            utils.save_image(couple_img, args.max_bound, args.min_bound, ncol=1, save=True, filename=filename_img)