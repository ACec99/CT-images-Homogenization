"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
from os.path import join as ospj
import matplotlib.pyplot as plt
import json
import csv
import re
import glob
from shutil import copyfile

from tqdm import tqdm
import ffmpeg

import math
import scipy
from scipy import stats
import pandas as pd
import pydicom

import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.utils as vutils
from PIL import Image, ImageColor, ImageDraw, ImageFont

import random, torch, os, numpy as np
from skimage import filters, morphology, measure
from medpy.filter.smoothing import anisotropic_diffusion
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
import yaml

def extract_metadata(args):

    coords_dict = {}

    with open(args.coords_lungs_dir, newline='') as coords_csv:
        coords_reader = csv.reader(coords_csv)

        for i, row in enumerate(coords_reader):

            if i > 0:
                patientID = (row[0].split('.'))[0]

                start_x = row[1]
                start_y = row[2]

                end_x = row[4]
                end_y = row[5]

                coords_dict[patientID] = [start_x, start_y, end_x, end_y]

    reference_dims = {}

    with open(args.ref_dims_dir, "r") as dims_file:
        # Read the file line by line
        for line in dims_file:
            line_list = line.strip().split(':')
            axis = line_list[0]
            if re.search('scale_factor', axis) != None:
                ref_dim = float(line_list[1])
            else:
                ref_dim = int(line_list[1])
            reference_dims[axis] = ref_dim

    return coords_dict, reference_dims

def get_domains_sizes(directories):
    domains_sizes = {}
    domains = []
    for directory in directories:
        for idx, (root, dirs, files) in enumerate(os.walk(directory)):
            if len(dirs) > 0:
                domains = dirs
            else:
                current_domain = domains[idx-1]
                if current_domain not in domains_sizes:
                    domains_sizes[current_domain] = len(files)
                else:
                    domains_sizes[current_domain] += len(files)
    return domains_sizes

def sort_domains_by_size(directories):
    domains_sizes = get_domains_sizes(directories)
    #print("la size dei domini è:", domains_sizes)
    sorted_domains = sorted(domains_sizes.items(), key=lambda x: x[1], reverse=True)
    return [folder[0] for folder in sorted_domains]

def compute_T_tests(args, domains):

    VOLUMES_OF_INTEREST_PATH = args.val_img_dir

    """domains = sort_domains_by_size(VOLUMES_OF_INTEREST_PATH)
    domains = domains[:args.num_domains]
    domains.sort()"""

    volumes_of_interest = []
    for folder_dom in domains:
        FOLDER_DOM_PATH = os.path.join(VOLUMES_OF_INTEREST_PATH, folder_dom)
        folder_dom_list = os.listdir(FOLDER_DOM_PATH)
        for f_dcm in folder_dom_list:
            F_DCM_PATH = os.path.join(FOLDER_DOM_PATH, f_dcm)
            dcm_read = pydicom.dcmread(F_DCM_PATH)
            patientID = dcm_read.PatientID
            if patientID not in volumes_of_interest:
                volumes_of_interest.append(patientID)
    #print(volumes_of_interest)

    data = pd.read_csv(args.fr_per_slice)

    mask = data['patientID'].isin(volumes_of_interest)

    data = data[mask]
    #print(data['domain'])
    data['domain'] = data['domain'].str.replace('QX/i', 'QX-i', regex=False)
    #print(data['domain'])

    #print(data)

    features_mean = data.drop("slice_num", axis=1).groupby(['patientID', 'domain', 'feature_name']).mean()
    domains_grouped = features_mean.groupby('domain')

    feature_per_volume_csv = ospj(args.current_expr_dir, '%01d_feature_per_volume.csv' % (args.num_domains))

    with open(feature_per_volume_csv, "w", newline="") as f_csv:
        writer = csv.writer(f_csv)

        features_dict = {}
        domains = []
        for domain, df_domain in domains_grouped:
            # print("### QUESTO E' IL DOMINIO:", domain, "###")
            features_grouped = df_domain.groupby("feature_name")
            domains.append(domain)
            for f, df_feature in features_grouped:
                # print("### LA FEATURE E':", f, "###")
                # print(df_feature)
                fr_vector = [float(elem) for elem in df_feature['feature_value']]
                writer.writerow([f, domain, fr_vector])
                if f not in features_dict:
                    features_dict[f] = [fr_vector]
                else:
                    features_dict[f].append(fr_vector)

    #relevant_features = []
    T_test_csv = ospj(args.current_expr_dir, '%01d_T_test_original_valid_data.csv' % (args.num_domains))

    with open(T_test_csv, "w", newline="") as t_test_csv:
        writer = csv.writer(t_test_csv)

        ### csv header ###
        writer.writerow(["feature_name", "dom1", "dom2", "t-test"])

        for feature in features_dict:
            #print(len(features_dict[feature]))

            for i in range(len(features_dict[feature])):
                for j in range(i + 1, len(features_dict[feature])):
                    fr_dom_1 = features_dict[feature][i]
                    fr_dom_2 = features_dict[feature][j]

                    _, p_value = stats.ttest_ind(fr_dom_1, fr_dom_2, equal_var=False)
                    writer.writerow([feature, domains[i], domains[j], p_value])


def extract_date(date_str):
    parts = date_str.split('-')
    date_info = parts[0].split(':') + parts[1].split(':')
    return tuple(map(int, date_info))

def save_json(json_file, filename):
    with open(filename, 'w') as f:
        json.dump(json_file, f, indent=4, sort_keys=False)


def print_network(network, name):
    num_params = 0
    for p in network.parameters():
        num_params += p.numel()
    # print(network)
    print("Number of parameters of %s: %i" % (name, num_params))


def he_init(module):
    if isinstance(module, nn.Conv2d):
        #print('Initializing')
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)
    if isinstance(module, nn.Linear):
        #print('Initializing')
        nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def denormalize(x, max_bound, min_bound, format='hu'):
    out = ((x+1) * (max_bound - min_bound)/2) + min_bound
    if format == 'image':
        out = (out - min_bound) / (max_bound - min_bound) #* 255 # porto tutti i valori tra zero e uno
    #out = (x + 1) / 2
    #return out.clamp_(0, 1)
    return out


def save_image(x, max_bound, min_bound, ncol, save=False, filename=None):
    x = denormalize(x, max_bound, min_bound, 'image')
    if save == True:
        vutils.save_image(x.cpu(), filename, nrow=ncol, padding=0)
    else:
        grid = vutils.make_grid(x.cpu(), ncol)
        # Add 0.5 after unnormalizing to [0, 255] to round to the nearest integer
        img_np = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
        #img = Image.fromarray(ndarr)
        return img_np


def save_tif(x, max_bound, min_bound, ncol, filename):
    x = denormalize(x, max_bound, min_bound)
    x_npy = x.cpu().detach().numpy()
    tifffile.imwrite(filename, x_npy)


def save_grid(images, ticks, rows, cols, filename, titles=None):
    if rows == 1 and cols == 1:
        fig, ax = plt.subplots(1,1, figsize=(15,10))
        plt.imshow(images)
        ax.xaxis.set_ticks_position('top')
        ticks_x_names = ticks[0]
        xlims = ax.get_xlim()
        if len(ticks) > 1:
            xtick_locations = np.linspace(xlims[0], xlims[1], ((len(ticks_x_names) + 1) * 2) + 1)
            #print("le locations di xtick sono:", xtick_locations)
            xtick_labels = ["", "", ""]
        else:
            xtick_locations = np.linspace(xlims[0], xlims[1], (len(ticks_x_names) * 2) + 1)
            xtick_labels = [""]
        for i in range(len(ticks_x_names)):
            xtick_labels += ["", ticks_x_names[i]]
        print("le xticks_labels sono:", xtick_labels)
        ax.set_xticks(xtick_locations)
        ax.set_xticklabels(xtick_labels, fontsize=8, rotation=45)
        ax.tick_params(axis='x', which='both', length=0)
        if len(ticks) > 1:
            ticks_y_names = ticks[1]
            ylims = ax.get_ylim()
            ytick_locations = np.linspace(ylims[0], ylims[1], ((len(ticks_y_names)+1) * 2) + 1)
            ytick_labels = []
            for j in range(len(ticks_y_names)):
                ytick_labels += ["", ticks_y_names[j]]
            ytick_labels += ["", "", ""]
            ax.set_yticks(ytick_locations)
            ax.set_yticklabels(ytick_labels, fontsize=8)
            ax.tick_params(axis='y', which='both', length=0)
        else:
            ax.set_yticks([])

    else:
        fig, axes = plt.subplots(rows, cols, figsize=(16,rows*8))
        if cols == 1:
            axes = np.array(axes).reshape(-1)

        for i, ax in enumerate(axes.flat):
            if i < images.shape[0]:
                ax.xaxis.set_ticks_position('top')
                ax.spines[['right', 'left', 'bottom']].set_visible(False)
                # ax.set_xticks([])
                ax.imshow(images[i])
                if len(ticks) > 1:
                    ticks_x_names = ticks[i]
                else:
                    ticks_x_names = ticks[0]
                xlims = sorted(ax.get_xlim())
                xtick_locations = np.linspace(xlims[0], xlims[1], (len(ticks_x_names) * 2) + 1)
                xtick_labels = [""]
                for j in range(len(ticks_x_names)):
                    xtick_labels += ["", ticks_x_names[j]]
                # xtick_labels += [""]
                ax.set_xticks(xtick_locations)
                ax.set_xticklabels(xtick_labels, fontsize=5, rotation=45)
                ax.tick_params(axis='x', which='both', length=0)
                ax.set_yticks([])
                if titles is not None:
                    #ax.set_title(titles[i])
                    ax.set_ylabel(titles[i], fontsize=10)
            else:
                ax.axis('off')

        plt.subplots_adjust(top=0.9, bottom=0.1, wspace=0.3, hspace=0.3)

    plt.tight_layout()
    fig.savefig(filename)
    plt.close(fig) # close the figure to release the memory


@torch.no_grad()
#def translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename, domains):
def translate_and_reconstruct(nets, args, x_src, y_src, y_trg, z_trg, filename, domains):

    """domains = sort_domains_by_size(args.val_img_dir)
    domains = domains[:args.num_domains]
    domains.sort()"""

    src_domains = []
    trg_domains = []
    for y_s in y_src:
        src_domains.append(domains[y_s])
    for y_r in y_trg:
        trg_domains.append(domains[y_r])

    translations_text = []
    for idx in range(len(src_domains)):
        translations_text.append(src_domains[idx] + " --> \n" + trg_domains[idx])
    """text_images = np.array([create_text_image(translation, x_src.shape[2], 50, 40) for translation in translations_text])
    text_images = torch.from_numpy(text_images).to(x_src.device)

    print("la shape di text_images è:", text_images.shape)
    print("la shape di x_src è:", x_src.shape)"""

    N, C, H, W = x_src.size()
    #s_ref = nets.style_encoder(x_ref, y_ref)
    s_trg = nets.mapping_network(z_trg, y_trg)
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    #x_fake = nets.generator(x_src, s_ref, masks=masks)
    x_fake = nets.generator(x_src, s_trg, masks=masks)
    s_src = nets.style_encoder(x_src, y_src)
    masks = nets.fan.get_heatmap(x_fake) if args.w_hpf > 0 else None
    x_rec = nets.generator(x_fake, s_src, masks=masks)
    #x_concat = [x_src, x_ref, x_fake, x_rec]
    x_concat = [x_src, x_fake, x_rec]
    x_concat = torch.cat(x_concat, dim=0)
    img_concat = save_image(x_concat, args.max_bound, args.min_bound, N)
    table_imgs_list = img_concat
    #legend_list = [src_domains, trg_domains]
    labels = [translations_text]
    save_grid(table_imgs_list, labels, 1, 1, filename, None)
    del x_concat


@torch.no_grad()
def translate_using_latent(nets, args, x_src, y_src, y_trg_list, z_trg_list, filename, domains, edge_loss=None, filename_edge=None):
    # qui dentro c'è anche il confronto per quanto riguarda l'edge extraction
    N, C, H, W = x_src.size()
    latent_dim = z_trg_list[0].size(1)
    #x_concat = [x_src]
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    src_domains = []
    for y in y_src:
        src_domains.append(domains[y])

    """domains = sort_domains_by_size(args.val_img_dir)
    domains = domains[:args.num_domains]
    domains.sort()"""

    table_imgs_list = []
    table_imgs_edge_list = []
    # -------------------------------------------- #
    edge_loss_labels = []  # To store loss values
    # -------------------------------------------- #

    for i, y_trg in enumerate(y_trg_list):

        ### y_trg has batch_size dimension --> the element repeated batch_size times is a single domain (one specific domain) ###

        """z_many = torch.randn(10000, latent_dim).to(x_src.device) # tensor of dimension (1000, 16) placed on the same device of x_src 
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0]) # tensor of dimension (1000,) filled with the value of y_trg[0]
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True) # dimension = (1, 64) where 64 = style dimension
        s_avg = s_avg.repeat(N, 1) # dimension = (4, 64)"""

        # Creare immagini di testo inclinate
        """text_images = np.array([create_text_image(src_name, x_src.shape[2], 50, 40) for src_name in src_domains])
        text_images = torch.from_numpy(text_images).to(x_src.device)"""

        x_concat = [x_src]
        #x_concat_edge = []

        for z_trg in z_trg_list:

            ### z_trg has (batch_size, latent_dim) dimension ###

            s_trg = nets.mapping_network(z_trg, y_trg)
            #s_trg = torch.lerp(s_avg, s_trg, psi) # linear interpolation
            x_fake = nets.generator(x_src, s_trg, masks=masks)
            x_concat += [x_fake]
            """if edge_loss is not None and not x_concat_edge:
                loss_edge_list = []
                x_fake_1_ch = x_fake[:, 1, :, :].unsqueeze(1)
                x_real_1_ch = x_src[:, 1, :, :].unsqueeze(1)
                laplacian_x_fake_all = torch.zeros_like(x_fake_1_ch)
                laplacian_x_real_all = torch.zeros_like(x_real_1_ch)
                for b in range(x_fake_1_ch.shape[0]):
                    loss_edge, laplacian_x, laplacian_y = edge_loss(x_real_1_ch[b].unsqueeze(0), x_fake_1_ch[b].unsqueeze(0))
                    loss_edge_list.append(loss_edge.item())
                    laplacian_x_real_all[b] = laplacian_x
                    laplacian_x_fake_all[b] = laplacian_y
                x_concat_edge += [laplacian_x_real_all]
                x_concat_edge += [laplacian_x_fake_all]
                # ---------------------------------------- #
                labels = [f"{src_domains[j]} \n {loss_edge_list[j]}" for j in range(len(loss_edge_list))]
                #print(labels)
                edge_loss_labels.append(labels)
                # --------------------------------------- #"""

        # transform tensor in image
        x_concat = torch.cat(x_concat, dim=0)
        img_concat = save_image(x_concat, args.max_bound, args.min_bound, N)
        table_imgs_list.append(img_concat)
        """if edge_loss is not None:
            x_concat_edge = torch.cat(x_concat_edge, dim=0)
            img_concat_edge = save_image(x_concat_edge, args.max_bound, args.min_bound, N)
            table_imgs_edge_list.append(img_concat_edge)"""

    table_imgs_np = np.array(table_imgs_list)
    #print("la shape di table_imgs_np nell'using_latent è:", table_imgs_np.shape)
    nrows = int(np.ceil(len(domains)/2))

    labels = [src_domains]
    save_grid(table_imgs_np, labels, nrows, 2, filename, domains)
    """if edge_loss is not None:
        print(edge_loss_labels)
        table_imgs_edge_np = np.array(table_imgs_edge_list)
        #print("la shape di table_imgs_np nell'using_latent è:", table_imgs_np.shape)
        nrows = int(np.ceil(len(domains) / 2))
        
        # the following two code lines were already commented
        labels = [src_domains]
        save_grid(table_imgs_edge_np, labels, nrows, 2, filename_edge, domains)"""

        # ---------------------------------------------------------------------------------------------------- #
        # Combine src_domains with corresponding loss_edge values for labels
        #save_grid(table_imgs_edge_np, edge_loss_labels, nrows, 2, filename_edge, domains)
        # ---------------------------------------------------------------------------------------------------- #


@torch.no_grad()
def translate_using_reference(nets, args, x_src, y_src, x_ref, y_ref, filename, domains):

    """domains = sort_domains_by_size(args.val_img_dir)
    domains = domains[:args.num_domains]
    domains.sort()"""

    src_domains = []
    for y_s in y_src:
        src_domains.append(domains[y_s])

    trg_domains = []
    for y_r in y_ref:
        trg_domains.append(domains[y_r])

    """text_images_src = np.array([create_text_image(src_name, x_src.shape[2], 50, 40) for src_name in src_domains])
    text_images_src = torch.from_numpy(text_images_src).to(x_src.device)
    text_images_trg = np.array([create_text_image(trg_name, x_ref.shape[2], 50, 40) for trg_name in trg_domains])
    text_images_trg = torch.from_numpy(text_images_trg).to(x_src.device)"""

    N, C, H, W = x_src.size()
    wb = torch.ones(1, C, H, W).to(x_src.device)
    x_src_with_wb = torch.cat([wb, x_src], dim=0)

    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None

    ### REFERENCE STILE ###
    s_ref = nets.style_encoder(x_ref, y_ref) # dimension = (batch_size, 64)
    #######################
    
    s_ref_list = s_ref.unsqueeze(1).repeat(1, N, 1)
    
    ### PUT IN THE FIRST ROW THE SOURCE IMAGES, with an empty first image (in order to create a correct table)
    x_concat = [x_src_with_wb]
    
    ### for batch_size time ###
    for i, s_ref in enumerate(s_ref_list):
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        x_fake_with_ref = torch.cat([x_ref[i:i+1], x_fake], dim=0)
        x_concat += [x_fake_with_ref]

    x_concat = torch.cat(x_concat, dim=0)
    img_concat = save_image(x_concat, args.max_bound, args.min_bound, N+1)
    table_imgs_list = img_concat
    labels = [src_domains, trg_domains]
    save_grid(table_imgs_list, labels, 1, 1, filename, None)
    del x_concat


@torch.no_grad()
def debug_image(nets, args, edge_loss, inputs, step, domains):
    x_src, y_src = inputs.x_src, inputs.y_src
    x_ref, y_ref = inputs.x_ref, inputs.y_ref

    device = inputs.x_src.device
    N = inputs.x_src.size(0) # 4 (splitted batch size)

    torch.manual_seed(42)

    # translate and reconstruct (reference-guided)
    filename = ospj(args.sample_dir, '%06d_cycle_consistency.jpg' % (step))
    #translate_and_reconstruct(nets, args, x_src, y_src, x_ref, y_ref, filename, domains)
    y_trg = torch.randint(low=0, high=args.num_domains-1, size=(args.val_batch_size,)).to(device)
    z_trg = torch.randn(1, args.latent_dim).repeat(N, 1).to(device)
    translate_and_reconstruct(nets, args, x_src, y_src, y_trg, z_trg, filename, domains)

    # latent-guided image synthesis
    y_trg_list = [torch.tensor(y).repeat(N).to(device)
                  for y in range(min(args.num_domains, 7))] # dimension = (num_domains, batch_size)
    z_trg_list = torch.randn(args.num_outs_per_domain, 1, args.latent_dim).repeat(1, N, 1).to(device) # dimension = (10,4,16)
    #for psi in [0.5, 0.7, 1.0]:
    filename = ospj(args.sample_dir, '%06d_latent.jpg' % (step))
    #if args.edge_loss:
    filename_edge = ospj(args.edge_imgs_dir, '%06d_latent.jpg' % (step))
    translate_using_latent(nets, args, x_src, y_src, y_trg_list, z_trg_list, filename, domains, edge_loss, filename_edge)
    #translate_using_latent(nets, args, x_src, y_src, y_trg_list, z_trg_list, filename, domains)

    # reference-guided image synthesis
    filename = ospj(args.sample_dir, '%06d_reference.jpg' % (step))
    translate_using_reference(nets, args, x_src, y_src, x_ref, y_ref, filename, domains)


# ======================= #
# Video-related functions #
# ======================= #


def sigmoid(x, w=1):
    return 1. / (1 + np.exp(-w * x))


def get_alphas(start=-5, end=5, step=0.5, len_tail=10):
    return [0] + [sigmoid(alpha) for alpha in np.arange(start, end, step)] + [1] * len_tail


def interpolate(nets, args, x_src, s_prev, s_next):
    ''' returns T x C x H x W '''
    B = x_src.size(0)
    frames = []
    masks = nets.fan.get_heatmap(x_src) if args.w_hpf > 0 else None
    alphas = get_alphas()

    for alpha in alphas:
        s_ref = torch.lerp(s_prev, s_next, alpha)
        x_fake = nets.generator(x_src, s_ref, masks=masks)
        entries = torch.cat([x_src.cpu(), x_fake.cpu()], dim=2)
        frame = torchvision.utils.make_grid(entries, nrow=B, padding=0, pad_value=-1).unsqueeze(0)
        frames.append(frame)
    frames = torch.cat(frames)
    return frames


def slide(entries, margin=32):
    """Returns a sliding reference window.
    Args:
        entries: a list containing two reference images, x_prev and x_next, 
                 both of which has a shape (1, 3, 256, 256)
    Returns:
        canvas: output slide of shape (num_frames, 3, 256*2, 256+margin)
    """
    _, C, H, W = entries[0].shape
    alphas = get_alphas()
    T = len(alphas) # number of frames

    canvas = - torch.ones((T, C, H*2, W + margin))
    merged = torch.cat(entries, dim=2)  # (1, 3, 512, 256)
    for t, alpha in enumerate(alphas):
        top = int(H * (1 - alpha))  # top, bottom for canvas
        bottom = H * 2
        m_top = 0  # top, bottom for merged
        m_bottom = 2 * H - top
        canvas[t, :, top:bottom, :W] = merged[:, :, m_top:m_bottom, :]
    return canvas


@torch.no_grad()
def video_ref(nets, args, x_src, x_ref, y_ref, fname):
    video = []
    s_ref = nets.style_encoder(x_ref, y_ref)
    s_prev = None
    for data_next in tqdm(zip(x_ref, y_ref, s_ref), 'video_ref', len(x_ref)):
        x_next, y_next, s_next = [d.unsqueeze(0) for d in data_next]
        if s_prev is None:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue
        if y_prev != y_next:
            x_prev, y_prev, s_prev = x_next, y_next, s_next
            continue

        interpolated = interpolate(nets, args, x_src, s_prev, s_next)
        entries = [x_prev, x_next]
        slided = slide(entries)  # (T, C, 256*2, 256)
        frames = torch.cat([slided, interpolated], dim=3).cpu()  # (T, C, 256*2, 256*(batch+1))
        video.append(frames)
        x_prev, y_prev, s_prev = x_next, y_next, s_next

    # append last frame 10 time
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


@torch.no_grad()
def video_latent(nets, args, x_src, y_list, z_list, psi, fname):
    latent_dim = z_list[0].size(1)
    s_list = []
    for i, y_trg in enumerate(y_list):
        z_many = torch.randn(10000, latent_dim).to(x_src.device)
        y_many = torch.LongTensor(10000).to(x_src.device).fill_(y_trg[0])
        s_many = nets.mapping_network(z_many, y_many)
        s_avg = torch.mean(s_many, dim=0, keepdim=True)
        s_avg = s_avg.repeat(x_src.size(0), 1)

        for z_trg in z_list:
            s_trg = nets.mapping_network(z_trg, y_trg)
            s_trg = torch.lerp(s_avg, s_trg, psi)
            s_list.append(s_trg)

    s_prev = None
    video = []
    # fetch reference images
    for idx_ref, s_next in enumerate(tqdm(s_list, 'video_latent', len(s_list))):
        if s_prev is None:
            s_prev = s_next
            continue
        if idx_ref % len(z_list) == 0:
            s_prev = s_next
            continue
        frames = interpolate(nets, args, x_src, s_prev, s_next).cpu()
        video.append(frames)
        s_prev = s_next
    for _ in range(10):
        video.append(frames[-1:])
    video = tensor2ndarray255(torch.cat(video))
    save_video(fname, video)


def save_video(fname, images, output_fps=30, vcodec='libx264', filters=''):
    assert isinstance(images, np.ndarray), "images should be np.array: NHWC"
    num_frames, height, width, channels = images.shape
    stream = ffmpeg.input('pipe:', format='rawvideo', 
                          pix_fmt='rgb24', s='{}x{}'.format(width, height))
    stream = ffmpeg.filter(stream, 'setpts', '2*PTS')  # 2*PTS is for slower playback
    stream = ffmpeg.output(stream, fname, pix_fmt='yuv420p', vcodec=vcodec, r=output_fps)
    stream = ffmpeg.overwrite_output(stream)
    process = ffmpeg.run_async(stream, pipe_stdin=True)
    for frame in tqdm(images, desc='writing video to %s' % fname):
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def tensor2ndarray255(images):
    images = torch.clamp(images * 0.5 + 0.5, 0, 1)
    return images.cpu().numpy().transpose(0, 2, 3, 1) * 255

def RemoveBed(img_batch):
    """
    Applies the RemoveBed operation to a batch of images.

    Parameters:
        img_batch (numpy.ndarray): A batch of images with shape (batch_size, height, width).

    Returns:
        numpy.ndarray: A batch of processed images with the same shape as the input.
    """

    cleaned_batch = []

    for img in img_batch:

        # ensure the input image is copied to avoid modifying the original
        img = img.copy()

        nan_mask = np.isnan(img)

        # handle nan values
        if np.sum(nan_mask) > 0:
            # Use nearest-neighbor interpolation to fill NaNs
            filled_image = ndimage.generic_filter(img, np.nanmean, size=3, mode='mirror')

            # Replace NaNs with interpolated values
            img[nan_mask] = filled_image[nan_mask]

        #  Otsu thresholding to create a mask
        mask = img > filters.threshold_otsu(img)

        # label connected components
        label_image = measure.label(mask, background=0)

        # find the largest connected component
        props = measure.regionprops(label_image)
        largest_region = max(props, key=lambda x: x.area)
        mask = (label_image == largest_region.label).astype(np.uint8)

        # morphological operations
        mask = morphology.binary_erosion(mask)
        mask = scipy.ndimage.binary_fill_holes(mask)

        img[~mask] = -1

        cleaned_batch.append(img)

    return np.array(cleaned_batch)

@torch.no_grad()
def segment_lung(img):
    # function sourced from https://www.kaggle.com/c/data-science-bowl-2017#tutorial
    """
    This segments the Lung Image(Don't get confused with lung nodule segmentation)
    """
    img = img.copy()

    mean = np.mean(img)
    std = np.std(img)
    img = img - mean
    img = img / std

    """plt.hist(img.flatten(), bins=200)
    plt.show()"""

    middle = img[100:400, 100:400]
    mean = np.mean(middle)
    max = np.max(img)
    min = np.min(img)
    # remove the underflow bins
    img[img == max] = mean
    img[img == min] = mean

    # apply median filter
    img = median_filter(img, size=3)
    # apply anistropic non-linear diffusion filter- This removes noise without blurring the nodule boundary
    img = anisotropic_diffusion(img)

    kmeans = KMeans(n_clusters=2).fit(np.reshape(middle, [np.prod(middle.shape), 1]))
    centers = sorted(kmeans.cluster_centers_.flatten())
    threshold = np.mean(centers)
    thresh_img = np.where(img < threshold, 1.0, 0.0)  # threshold the image
    eroded = morphology.erosion(thresh_img, np.ones([4, 4]))
    dilation = morphology.dilation(eroded, np.ones([10, 10]))
    labels = measure.label(dilation)
    label_vals = np.unique(labels)
    regions = measure.regionprops(labels)
    good_labels = []
    for prop in regions:
        B = prop.bbox
        if B[2] - B[0] < 475 and B[3] - B[1] < 475 and B[0] > 40 and B[2] < 472:
            good_labels.append(prop.label)
    mask = np.ndarray([512, 512], dtype=np.int8)
    mask[:] = 0
    #
    #  The mask here is the mask for the lungs--not the nodes
    #  After just the lungs are left, we do another large dilation
    #  in order to fill in and out the lung mask
    #
    for N in good_labels:
        mask = mask + np.where(labels == N, 1, 0)
    # mask = morphology.dilation(mask,np.ones([2,2])) # one last dilation
    # mask consists of 1 and 0. Thus by mutliplying with the orginial image, sections with 1 will remain
    return mask * img

