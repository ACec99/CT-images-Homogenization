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
import csv
import re

from munch import Munch
from torch.backends import cudnn
import torch
import datetime

from core.data_loader import get_train_loader
from core.data_loader import get_test_loader
from core.solver import Solver
import core.utils as utils
from metrics.fid import calculate_fid_for_all_tasks


def str2bool(v):
    return v.lower() in ('true')


def subdirs(dname):
    return [d for d in os.listdir(dname)
            if os.path.isdir(os.path.join(dname, d))]


def main(args):
    print(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    # -------------- If we want to take domains in order of dimension -------------- #
    """directories = [args.train_img_dir, args.val_img_dir, args.src_dir]
    domains_sorted = utils.sort_domains_by_size(directories)
    domains = domains_sorted[:args.num_domains]"""
    # ----------------------------------------------------------------------------- #
    domains = [
        'GE MEDICAL SYSTEMS LightSpeed QX-i -- BONE',
        'SIEMENS Sensation 16 -- B30f'
        #'GE MEDICAL SYSTEMS LightSpeed16 -- STANDARD'
    ]

    domains.sort()
    print("I DOMINI SONO I SEGUENTI:", domains)
    utils.compute_T_tests(args, domains)
    coords_dict, reference_dims = utils.extract_metadata(args)

    solver = Solver(args, domains)

    if args.mode == 'train':
        #assert len(subdirs(args.train_img_dir)) == args.num_domains
        #assert len(subdirs(args.val_img_dir)) == args.num_domains
        loaders = Munch(src=get_train_loader(root=args.train_img_dir,
                                             lung_coords=coords_dict,
                                             ref_dims=reference_dims,
                                             min_bound=args.min_bound,
                                             max_bound=args.max_bound,
                                             domains=domains,
                                             which='source',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers,
                                             resize=args.resize_bool),
                        ref=get_train_loader(root=args.train_img_dir,
                                             lung_coords=coords_dict,
                                             ref_dims=reference_dims,
                                             min_bound=args.min_bound,
                                             max_bound=args.max_bound,
                                             domains=domains,
                                             which='reference',
                                             img_size=args.img_size,
                                             batch_size=args.batch_size,
                                             prob=args.randcrop_prob,
                                             num_workers=args.num_workers,
                                             resize=args.resize_bool),
                        val=get_test_loader(root=args.val_img_dir,
                                            lung_coords=coords_dict,
                                            ref_dims=reference_dims,
                                            min_bound=args.min_bound,
                                            max_bound=args.max_bound,
                                            domains=domains,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=True,
                                            num_workers=args.num_workers,
                                            resize=args.resize_bool))
        calculate_fid_for_all_tasks(args, domains, None, 'initial', coords_dict, reference_dims)
        solver.train(loaders)
    elif args.mode == 'sample':
        #assert len(subdirs(args.src_dir)) == args.num_domains
        #assert len(subdirs(args.ref_dir)) == args.num_domains
        loaders = Munch(src=get_test_loader(root=args.src_dir,
                                            lung_coords=coords_dict,
                                            ref_dims=reference_dims,
                                            min_bound=args.min_bound,
                                            max_bound=args.max_bound,
                                            domains=domains,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            resize=args.resize_bool),
                        ref=get_test_loader(root=args.ref_dir,
                                            lung_coords=coords_dict,
                                            ref_dims=reference_dims,
                                            min_bound=args.min_bound,
                                            max_bound=args.max_bound,
                                            domains=domains,
                                            img_size=args.img_size,
                                            batch_size=args.val_batch_size,
                                            shuffle=False,
                                            num_workers=args.num_workers,
                                            resize=args.resize_bool))

        solver.sample(loaders)
    elif args.mode == 'eval':
        solver.evaluate()
    elif args.mode == 'extractions':
        solver.print_samples()
    elif args.mode == 'align':
        from core.wing import align_faces
        align_faces(args, args.inp_dir, args.out_dir)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # model arguments
    parser.add_argument('--img_size', type=int, default=512,
                        help='Image resolution')
    parser.add_argument('--num_domains', type=int, default=2,
                        help='Number of domains')
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Latent vector dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, #1024,
                        help='Hidden dimension of mapping network')
    parser.add_argument('--style_dim', type=int, default=64,
                        help='Style code dimension')

    # directory for crop images
    parser.add_argument('--coords_lungs_dir', type=str, default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets/csv_files/lung_coordinates.csv',
                        help='Directory containing coordinates of the maximum box containing lungs for each scan')
    parser.add_argument('--ref_dims_dir', type=str, default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets/csv_files/reference_dimensions.txt',
                        help='Directory containing reference dimensions for cropping')
    

    # weight for objective functions
    parser.add_argument('--lambda_reg', type=float, default=1,
                        help='Weight for R1 regularization')

    # ------------------------------------------------------------------------ #
    parser.add_argument('--lambda_cyc', type=float, default=1,
                        help='Weight for cyclic consistency loss')
    parser.add_argument('--lambda_sty', type=float, default=1,
                        help='Weight for style reconstruction loss')
    parser.add_argument('--lambda_ds', type=float, default=1,
                        help='Weight for diversity sensitive loss')
    parser.add_argument('--lambda_adv', type=float, default=1,
                        help='Weight for adversarial loss')
    parser.add_argument('--lambda_tx', type=float, default=1,
                        help='Weight for texture loss')
    parser.add_argument('--lambda_edge', type=float, default=1,
                        help='Weight for edge loss')
    # ------------------------------------------------------------------------ #

    parser.add_argument('--ds_iter', type=int, default=100000,
                        help='Number of iterations to optimize diversity sensitive loss')
    parser.add_argument('--w_hpf', type=float, default=0,
                        help='weight for high-pass filtering')

    # training arguments
    parser.add_argument('--randcrop_prob', type=float, default=0.5,
                        help='Probabilty of using random-resized cropping')
    parser.add_argument('--total_iters', type=int, default=100000,
                        help='Number of total iterations')
    parser.add_argument('--resume_iter', type=int, default=0,
                        help='Iterations to resume training/testing')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='Batch size for validation')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for D, E and G')
    parser.add_argument('--f_lr', type=float, default=1e-6,
                        help='Learning rate for F')
    parser.add_argument('--beta1', type=float, default=0.0,
                        help='Decay rate for 1st moment of Adam')
    parser.add_argument('--beta2', type=float, default=0.99,
                        help='Decay rate for 2nd moment of Adam')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')
    parser.add_argument('--num_outs_per_domain', type=int, default=10,
                        help='Number of generated images per domain during sampling')

    parser.add_argument('--texture_loss', type=bool, default=False,
                        help='Whether to add texture loss in your model.')
    parser.add_argument('--edge_loss', type=bool, default=False,
                        help='Whether to add edge loss in your model.')
    parser.add_argument('--ssim', type=bool, default=False,
                        help='Whether to use SSIM as edge loss in your model.')
    parser.add_argument('--resize_bool', type=bool, default=False,
                        help='Whether to apply transformations in the dataloader.')

    # misc
    parser.add_argument('--mode', type=str, required=True,
                        choices=['train', 'sample', 'eval', 'align', 'extractions'],
                        help='This argument is used in solver')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=777,
                        help='Seed for random number generator')

    # directory for training
    parser.add_argument('--train_img_dir', type=str, default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/StarGAN_v2/StarGAN_dataset/train',
                        help='Directory containing training images')
    parser.add_argument('--val_img_dir', type=str, default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/StarGAN_v2/StarGAN_dataset/valid',
                        help='Directory containing validation images')

    parser.add_argument('--src_dir', type=str, default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/StarGAN_v2/StarGAN_dataset/test',
                        help='Directory containing input source images')
    parser.add_argument('--ref_dir', type=str, default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/StarGAN_v2/StarGAN_dataset/test',
                        help='Directory containing input reference images')
    parser.add_argument('--inp_dir', type=str, default='assets/representative/custom/female',
                        help='input directory when aligning faces')
    parser.add_argument('--out_dir', type=str, default='assets/representative/celeba_hq/src/female',
                        help='output directory when aligning faces')

    # directory for experiments
    parser.add_argument('--expr_dir', type=str, default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/StarGAN_v2/experiments',
                        help='Directory containing experiments')

    # directory of the checkpoints of interest
    parser.add_argument('--pretr_w_dir', type=str, default=None,
                        help='Directory of the checkpoints of interest')

    # directory of radiomic features of interest (radiomic features that lead to a T-test < 0.05 among couples of domains (containing volumes belonging to the validation set)
    #parser.add_argument('--fr_json_dir', type=str, default="/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets/NSCLC_Radiomics/csv_files/relevant_features_valid_data.json",
    #                    help='Directory of json file containing the relevant radiomic features')

    # directory of domains radiomic features (for each volume)
    #parser.add_argument('--domains_fr_dir', type=str, default="/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets/NSCLC_Radiomics/csv_files/feature_per_volume.csv")

    # directory of radiomic features for each slice (already preprocessed)
    parser.add_argument('--fr_per_slice', type=str, default="/mimer/NOBACKUP/groups/snic2022-5-277/assolito/csv_files/radiomic_features_per_slice.csv")

    # face alignment
    parser.add_argument('--wing_path', type=str, default='experiments/checkpoints/wing.ckpt')
    parser.add_argument('--lm_path', type=str, default='experiments/checkpoints/celeba_lm_mean.npz')

    # normalization dimensions
    parser.add_argument('--min_bound', type=float, default=-1024.0)
    parser.add_argument('--max_bound', type=float, default=400.0)

    # step size
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--show_every', type=int, default=1000)
    parser.add_argument('--sample_every', type=int, default=5000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=10000)

    args = parser.parse_args()

    if args.resume_iter > 0:
        current_experiment_path = args.pretr_w_dir

    else:

        t = datetime.datetime.now()
        minute = str(t.minute)
        if t.minute < 10:
            minute = '0' + minute
        daytime = str(t.year) + ':' + str(t.month) + ':' + str(t.day) + '-' + str(t.hour) + ':' + minute

        current_experiment_path = args.expr_dir + '/' + daytime
        if not os.path.exists(current_experiment_path):
            os.makedirs(current_experiment_path)

    samples_folder_path = current_experiment_path + '/samples'
    if not os.path.exists(samples_folder_path):
        os.makedirs(samples_folder_path)

    edge_folder_path = current_experiment_path + '/edge_images'
    if not os.path.exists(edge_folder_path):
        os.makedirs(edge_folder_path)

    checkpoints_folder_path = current_experiment_path + '/checkpoints'
    if not os.path.exists(checkpoints_folder_path):
        os.makedirs(checkpoints_folder_path)

    eval_folder_path = current_experiment_path + '/eval'
    if not os.path.exists(eval_folder_path):
        os.makedirs(eval_folder_path)

    """grid_folder_path = current_experiment_path + '/grid'
    if not os.path.exists(grid_folder_path):
        os.makedirs(grid_folder_path)"""

    tifs_folder_path = current_experiment_path + '/tifs'
    if not os.path.exists(tifs_folder_path):
        os.makedirs(tifs_folder_path)

    losses_folder_path = current_experiment_path + '/losses'
    if not os.path.exists(losses_folder_path):
        os.makedirs(losses_folder_path)

    results_folder_path = current_experiment_path + '/results'
    if not os.path.exists(results_folder_path):
        os.makedirs(results_folder_path)

    parser.add_argument('--current_expr_dir', type=str, default=current_experiment_path,
                        help='Directory for saving generated images')
    parser.add_argument('--sample_dir', type=str, default=samples_folder_path,
                        help='Directory for saving generated images')
    parser.add_argument('--edge_imgs_dir', type=str, default=edge_folder_path,
                       help='Directory for saving generated edge images')
    parser.add_argument('--checkpoint_dir', type=str, default=checkpoints_folder_path,
                        help='Directory for saving network checkpoints')
    parser.add_argument('--eval_dir', type=str, default=eval_folder_path,
                        help='Directory for saving metrics, i.e., FID and LPIPS')
    """parser.add_argument('--grid_dir', type=str, default=grid_folder_path,
                        help='Directory for saving grid with source, reference and generated images')"""
    parser.add_argument('--tifs_dir', type=str, default=tifs_folder_path,
                        help='Directory for saving tif files representing the created fake images')
    parser.add_argument('--loss_monitoring_dir', type=str, default=losses_folder_path,
                        help='Directory for saving plots and history of losses')
    parser.add_argument('--result_dir', type=str, default=results_folder_path,
                        help='Directory for saving generated images and videos')


    args = parser.parse_args()

    main(args)
