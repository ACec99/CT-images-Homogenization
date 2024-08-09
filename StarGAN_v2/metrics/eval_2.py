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

from metrics.fid import calculate_fid_for_all_tasks
from metrics.lpips import calculate_lpips_given_images
from core.data_loader import get_eval_loader
from core import utils

from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True, nb_workers=32)

def feature_extraction(row, args, extractor):
    features_volume_dict = {}
    volume_fake = row.to_list()
    volume_fake = [x for x in volume_fake if x is not None]

    for slice_fake in volume_fake:
        slice_fake = slice_fake[0].squeeze()
        ## PARTE FINTA
        # slice_fake = (slice_fake.cpu()).numpy()
        # features_slice = {f"f_{i}": val for i, val in enumerate(slice_fake.mean(axis=0))}
        ####
        slice_fake = utils.denormalize(slice_fake, args.max_bound, args.min_bound)
        #slice_fake = (slice_fake.cpu()).numpy()

        img_fake = sitk.GetImageFromArray(slice_fake)
        mask = img_fake > -1024
        features_slice = extractor.execute(img_fake, mask)
        ###
        for feature_name in features_slice.keys():
            if feature_name not in features_volume_dict:
                features_volume_dict[feature_name] = [features_slice[feature_name]]
            else:
                features_volume_dict[feature_name].append(features_slice[feature_name])

    features_volume_dict = pd.DataFrame({feature: np.mean(np.array(features_volume_dict[feature])) for feature in features_volume_dict}, index=[row.name])
    return features_volume_dict


@torch.no_grad()
def calculate_metrics(nets, args, step, mode, domains):
    print('Calculating evaluation metrics...')
    assert mode in ['latent', 'reference']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    coords_dict, reference_dims = utils.extract_metadata(args)

    #domains = os.listdir(args.val_img_dir)
    #domains.sort()

    """domains = utils.sort_domains_by_size(args.val_img_dir)
    domains = domains[:args.num_domains]
    domains.sort()"""

    num_domains = len(domains)
    print('Number of domains: %d' % num_domains)

    extractor = featureextractor.RadiomicsFeatureExtractor()

    if 'shape2D' not in extractor.enabledFeatures:
        extractor.enabledFeatures['shape2D'] = []
    extractor.settings['force2D'] = True

    """relevant_fr_path = os.path.join(args.current_expr_dir, '%01d_feature_per_volume.csv' % (args.num_domains))
    with open(relevant_fr_path, 'r') as rf_json_file:
        relevant_features = json.load(rf_json_file)"""

    t_test_orig_valid_data_path = os.path.join(args.current_expr_dir, '%01d_T_test_original_valid_data.csv' % (args.num_domains))
    t_test_orig_valid_data_table = pd.read_csv(t_test_orig_valid_data_path, header=None, names=['feature_name', 'dom1', 'dom2', 't_test'], skiprows=1)

    features_grouped = t_test_orig_valid_data_table.groupby("feature_name")
    relevant_features = []
    for f, df_feature in features_grouped:
        relevant_features.append(f)
        """t_test_values = df_feature['t_test'].tolist()
        t_test_verified = all(float(t_test) < 0.05 for t_test in t_test_values)
        if t_test_verified:
            relevant_features.append(f)"""
    #print("LE FEATURES RILEVANTI SONO LE SEGUENTI:", relevant_features)

    extractor_feature_classes = extractor.featureClassNames
    relevant_feature_classes = []
    relevant_feature_complete_names = []
    enabledFeatures = {}
    for feature in relevant_features:
        feature_splitted = feature.split('_')
        feature_class = feature_splitted[1]
        feature_name = feature_splitted[2]
        #feature_complete_name = 'original_' + feature_splitted[0] + '_' + feature_splitted[1]
        if feature_class not in relevant_feature_classes:
            relevant_feature_classes.append(feature_class)
        if feature not in relevant_feature_complete_names:
            relevant_feature_complete_names.append(feature)
        # costruzione del dizionario contenente le feature radiomiche di interesse (quelle che vogliamo estrarre --> non volgiamo estrarle tutte)
        if feature_class not in enabledFeatures:
            enabledFeatures[feature_class] = [feature_name]
        else:
            enabledFeatures[feature_class].append(feature_name)

    #print("LE CLASSI DELLE FEATURES RILEVANTI SONO:", relevant_feature_classes)

    # disabilitare le feature classes che non ci interessano
    for extr_feature_class in extractor_feature_classes:
        if extr_feature_class not in relevant_feature_classes:
            extractor.enableFeatureClassByName(extr_feature_class, enabled=False)

    # abilitare solamente l'estrazione delle feature di interesse
    extractor.enableFeaturesByName(**enabledFeatures)

    if mode == 'latent':
        T_test_filename = os.path.join(args.eval_dir, 'valid_T_test_%.5i.csv' % step)
        t_test_dataframe = pd.DataFrame(columns=['feature_name', 'source_domain', 'domain_mapped', 't-test'])

    lpips_dict = OrderedDict()
    for trg_idx, trg_domain in enumerate(domains):

        # calcola il dizionario con chiavi le radiomic features ed associata a ciascuna di esse la lista dei relativi valori per ogni volume
        columns_names = ['feature_name', 'domain', 'values']
        feature_table_path = os.path.join(args.current_expr_dir, '%01d_feature_per_volume.csv' % (args.num_domains))
        features_table = pd.read_csv(feature_table_path, header=None, names=columns_names)
        features_table_trg_domain = features_table.loc[features_table['domain'] == trg_domain]

        features_real_dict = {}

        for idx_table, row in features_table_trg_domain.iterrows():
            feature_name = row['feature_name']
            if feature_name in relevant_feature_complete_names:
                features_real_dict[feature_name] = ast.literal_eval(row['values'])

        print(features_real_dict)

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
                                         drop_last=True)

        for src_idx, src_domain in enumerate(src_domains):
            path_src = os.path.join(args.val_img_dir, src_domain)
            loader_src = get_eval_loader(root=path_src,
                                         lung_coords=coords_dict,
                                         ref_dims=reference_dims,
                                         min_bound=args.min_bound,
                                         max_bound=args.max_bound,
                                         img_size=args.img_size,
                                         batch_size=args.val_batch_size)

            ### CREAZIONE DELLA COPPIA DI DOMINI DA COMPARARE ###
            task = '%s_to_%s' % (src_domain, trg_domain)

            #path_fake_imgs = os.path.join(args.eval_dir, task)
            #path_fake_grid = os.path.join(args.eval_dir, task)
            path_fake_tifs = os.path.join(args.tifs_dir, task)
            shutil.rmtree(path_fake_tifs, ignore_errors=True)
            #shutil.rmtree(path_fake_grid, ignore_errors=True)
            os.makedirs(path_fake_tifs)
            #os.makedirs(path_fake_imgs)
            """if not os.path.exists(path_fake_grid):
                os.makedirs(path_fake_grid)"""

            lpips_values = []
            print('Generating images and calculating LPIPS for %s...' % task)
            volumes_fake_dict = {}
            for i, x_src_complete in enumerate(tqdm(loader_src, total=len(loader_src))):
                x_src , patient_src = x_src_complete
                #print(x_src.shape)
                #print(patient_src)
                N = x_src.size(0)
                x_src = x_src.to(device)
                y_trg = torch.tensor([trg_idx] * N).to(device)
                masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

                # generate 10 outputs from the same input
                group_of_images = []
                for j in range(args.num_outs_per_domain):
                    #print("sono entrato nel ciclo per generare j output per lo stesso x_src")
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
                    group_of_images.append(x_fake)

                    # save generated images to calculate FID later
                    for k in range(N):

                        if j == 0:
                            patientID = patient_src[k]
                            if patientID not in volumes_fake_dict:
                                volumes_fake_dict[patientID] = [x_fake[k].cpu().numpy()]
                            else:
                                volumes_fake_dict[patientID].append(x_fake[k].cpu().numpy())

                        """if mode == 'reference':

                            grid_img = torch.cat([x_src[k].unsqueeze(0), x_ref[k].unsqueeze(0), x_fake[k].unsqueeze(0)], dim=0)

                            filename_grid_img = os.path.join(
                                path_fake_grid,
                                'grid_%.4i_%.2i.png' % (i * args.val_batch_size + (k + 1), j + 1))
                            utils.save_image(grid_img, args.max_bound, args.min_bound, ncol=3,
                                             filename=filename_grid_img)

                        if mode == 'latent':

                            grid_img = torch.cat([x_src[k].unsqueeze(0), x_fake[k].unsqueeze(0)], dim=0)

                            filename_grid_img = os.path.join(
                                path_fake_grid,
                                'latent_grid_%.4i_%.2i.png' % (i * args.val_batch_size + (k + 1), j + 1))
                            utils.save_image(grid_img, args.max_bound, args.min_bound, ncol=2, save=True,
                                             filename=filename_grid_img)

                        filename_single_img = os.path.join(
                            path_fake_imgs,
                            '%.4i_%.2i.png' % (i * args.val_batch_size + (k + 1), j + 1))
                        utils.save_image(x_fake[k], args.max_bound, args.min_bound, ncol=1,
                                         filename=filename_single_img)"""

                        filename_single_tif = os.path.join(
                            path_fake_tifs,
                            '%.4i_%.2i.tif' % (i * args.val_batch_size + (k + 1), j + 1))
                        utils.save_tif(x_fake[k], args.max_bound, args.min_bound, ncol=3,
                                         filename=filename_single_tif)

                lpips_value = calculate_lpips_given_images(group_of_images)
                lpips_values.append(lpips_value)

            ### dentro a lpips_values ci sono 32 (numero di immagini presenti mel loader_src) lpips_values ###
            ### lpips_value rappresenta la media della distanza (differenza) di ciascuna immagine genera

            # calculate LPIPS for each task (e.g. dom1_i2dom_j)
            lpips_mean = np.array(lpips_values).mean()
            lpips_dict['LPIPS_%s/%s' % (mode, task)] = lpips_mean
            ### per ogni task (coppia di domini dove uno è la sorgente e l'altro è il target = stile di riferimento) viene calcolato un lpips ###

            ### RADIOMIC FEATURE COMPARISONS ###
            if mode == 'latent':
                print("sono entrato nell'if dove parallelizza")
                volume_fake_dict = pd.DataFrame.from_dict(volumes_fake_dict, orient="index")
                feature_extraction_function = lambda row: feature_extraction(row, args, extractor)
                # features_fake_dict = volume_fake_dict.parallel_apply(feature_extraction, args=args, extractor=extractor, axis=1)
                features_fake_dict = volume_fake_dict.parallel_apply(feature_extraction_function, axis=1)
                features_fake_dict = pd.concat(features_fake_dict.values).to_dict(orient="list")
                print("ho appena finito le operazioni parallelizzate")
                print(features_fake_dict)
                #features_fake_dict = {}

                #print("il volumes_fake_dict è:", volumes_fake_dict)


                """for patient in volumes_fake_dict:

                    features_volume_dict = {}
                    volume_fake = volumes_fake_dict[patient]

                    print(len(volume_fake))

                    for slice_fake in volume_fake:
                        slice_fake = slice_fake[0].squeeze()
                        #print(slice_fake.shape)
                        slice_fake = utils.denormalize(slice_fake, args.max_bound, args.min_bound)
                        slice_fake = (slice_fake.cpu()).numpy()
                        img_fake = sitk.GetImageFromArray(slice_fake)
                        mask = img_fake > -1024
                        features_slice = extractor.execute(img_fake, mask)
                        for feature_name in features_slice.keys():
                            if feature_name.startswith("original_"):
                                if feature_name not in features_volume_dict:
                                    features_volume_dict[feature_name] = [features_slice[feature_name].item()]
                                else:
                                    features_volume_dict[feature_name].append(features_slice[feature_name].item())

                    for feature in features_volume_dict:
                        volume_feature = np.mean(np.array(features_volume_dict[feature]))
                        if feature not in features_fake_dict:
                            features_fake_dict[feature] = [volume_feature]
                        else:
                            features_fake_dict[feature].append(volume_feature)

                print(features_fake_dict)"""

                for feature in features_fake_dict:
                    real_domain = features_real_dict[feature]
                    fake_domain = features_fake_dict[feature]
                    _, p_value = stats.ttest_ind(real_domain, fake_domain)
                    print("ho calcolato il p_value e lo sto salvando")
                    new_line_dataframe = {'feature_name': feature, 'source_domain': src_domain , 'domain_mapped':trg_domain, 't-test': p_value}
                    t_test_dataframe = t_test_dataframe.append(new_line_dataframe, ignore_index=True)

        # delete dataloaders
        del loader_src
        if mode == 'reference':
            del loader_ref
            del iter_ref

    if mode == 'latent':
        t_test_dataframe.to_csv(T_test_filename, index=False)

    # calculate the average LPIPS for all tasks
    lpips_mean = 0
    for _, value in lpips_dict.items():
        lpips_mean += value / len(lpips_dict)
    lpips_dict['LPIPS_%s/mean' % mode] = lpips_mean

    # report LPIPS values
    filename = os.path.join(args.eval_dir, 'LPIPS_%.5i_%s.json' % (step, mode))
    utils.save_json(lpips_dict, filename)

    # calculate and report fid values
    calculate_fid_for_all_tasks(args, domains, step=step, mode=mode, lung_coords=coords_dict, ref_dims=reference_dims)

    folder_tifs_list = os.listdir(args.tifs_dir)
    for folder_tifs in folder_tifs_list:
        folder_tifs_path = os.path.join(args.tifs_dir, folder_tifs)
        shutil.rmtree(folder_tifs_path, ignore_errors=True)