"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import os
import argparse

import torch
import torch.nn as nn
import numpy as np
from torchvision import models
from scipy import linalg
from core.data_loader import get_eval_loader
from core import utils
import pandas as pd
import csv

def calculate_fid_for_all_tasks(args, domains, step, mode, lung_coords, ref_dims):
    print('Calculating FID for all tasks...')
    #fid_values = OrderedDict()
    fid_values = {}
    for trg_domain in domains:
        src_domains = [x for x in domains if x != trg_domain]

        for src_domain in src_domains:
            path_real = os.path.join(args.val_img_dir, trg_domain)
            if mode == 'initial':
                task = 'among_%s_%s' % (src_domain, trg_domain)
                path_fake = os.path.join(args.val_img_dir, src_domain)
            else:
                task = '%s_to_%s' % (src_domain, trg_domain)
                path_fake = os.path.join(args.tifs_dir, task)
            print('Calculating FID %s...' % task)
            fid_value = calculate_fid_given_paths(
                paths=[path_real, path_fake],
                lung_coords=lung_coords,
                ref_dims=ref_dims,
                min_bound=args.min_bound,
                max_bound=args.max_bound,
                img_size=args.img_size,
                batch_size=args.val_batch_size,
                resize=args.resize_bool)
            if mode == 'initial':
                fid_values['FID_%s/%s' % (trg_domain, src_domain)] = [fid_value]
            else:
                fid_values['FID_%s/%s' % (mode, task)] = [fid_value]

    # calculate the average FID for all tasks
    fid_mean = 0
    for _, value in fid_values.items():
        fid_mean += value[0] / len(fid_values)
    fid_values['FID_%s/mean' % mode] = [fid_mean]

    # report FID values
    if step is not None:
        filename = os.path.join(args.eval_dir, 'FID_%.5i_%s.csv' % (step, mode))
    else:
        filename = os.path.join(args.current_expr_dir, 'FID.csv')

    fid_df = pd.DataFrame(fid_values)
    fid_df.to_csv(filename, index=False)

    #utils.save_json(fid_values, filename)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x


class InceptionV3(nn.Module):
    def __init__(self):
        super().__init__()
        inception = models.inception_v3(pretrained=True)
        self.block1 = nn.Sequential(
            inception.Conv2d_1a_3x3, inception.Conv2d_2a_3x3,
            inception.Conv2d_2b_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block2 = nn.Sequential(
            inception.Conv2d_3b_1x1, inception.Conv2d_4a_3x3,
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.block3 = nn.Sequential(
            inception.Mixed_5b, inception.Mixed_5c,
            inception.Mixed_5d, inception.Mixed_6a,
            inception.Mixed_6b, inception.Mixed_6c,
            inception.Mixed_6d, inception.Mixed_6e)
        self.block4 = nn.Sequential(
            inception.Mixed_7a, inception.Mixed_7b,
            inception.Mixed_7c,
            nn.AdaptiveAvgPool2d(output_size=(1, 1)))

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return x.view(x.size(0), -1)


def frechet_distance(mu, cov, mu2, cov2):
    cc, _ = linalg.sqrtm(np.dot(cov, cov2), disp=False)
    dist = np.sum((mu -mu2)**2) + np.trace(cov + cov2 - 2*cc)
    return np.real(dist)


@torch.no_grad()
def calculate_fid_given_paths(paths, lung_coords, ref_dims, min_bound, max_bound, img_size=256, batch_size=50, resize=False):
    print('Calculating FID given paths %s and %s...' % (paths[0], paths[1]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    inception = InceptionV3().eval().to(device)
    loaders = [get_eval_loader(path, lung_coords, ref_dims, min_bound, max_bound, img_size, batch_size, resize) for path in paths]

    mu, cov = [], []
    for loader in loaders:
        actvs = []
        for x_complete in tqdm(loader, total=len(loader)):
            x , _ = x_complete
            actv = inception(x.to(device))
            actvs.append(actv)
        actvs = torch.cat(actvs, dim=0).cpu().detach().numpy()
        mu.append(np.mean(actvs, axis=0))
        cov.append(np.cov(actvs, rowvar=False))
        del actvs
    fid_value = frechet_distance(mu[0], cov[0], mu[1], cov[1])
    del mu
    del cov
    return fid_value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--paths', type=str, nargs=2, help='paths to real and fake images')
    parser.add_argument('--img_size', type=int, default=512, help='image resolution')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size to use')
    args = parser.parse_args()
    coords_dict, reference_dims = utils.extract_metadata(args)
    fid_value = calculate_fid_given_paths(args.paths, coords_dict, reference_dims, args.min_bound, args.max_bound, args.img_size, args.batch_size, args.resize_bool)
    print('FID: ', fid_value)

# python -m metrics.fid --paths PATH_REAL PATH_FAKE