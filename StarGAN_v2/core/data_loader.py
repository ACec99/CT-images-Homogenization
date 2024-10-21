"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np
import tifffile

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

import pydicom
from  scipy.ndimage import zoom , rotate

from core import utils

def listdir(folder_name):
    dicom_files_paths = list(chain(*[list(Path(folder_name).    rglob('*.' + ext))
                                    for ext in ['dcm', 'png', 'tif']]))
    return dicom_files_paths


class DefaultDataset(data.Dataset):
    def __init__(self, root, lung_coords, ref_dims, min_bound, max_bound, img_size, transform=None):
        self.samples_paths = listdir(root)
        self.samples_paths.sort()
        self.transform = transform
        self.targets = None
        self.lung_coords = lung_coords
        self.ref_dims = ref_dims
        self.min_bound = min_bound
        self.max_bound = max_bound
        self.img_size = img_size

    def __getitem__(self, index):
        f_path = self.samples_paths[index] # a single dicom file
        extension = f_path.suffix

        if extension == '.dcm':

            # read dicom file
            dcm_read = pydicom.dcmread(f_path)

            # take the pixels array (image)
            image = dcm_read.pixel_array
            image = transofrm_image(dcm_read, image, self.lung_coords, self.ref_dims)

            image = 2*((image - self.min_bound) / (self.max_bound - self.min_bound))-1
            #image = utils.RemoveBed(image)

            if self.transform is not None:
                image = self.transform(image)

            image = torch.tensor(image, dtype=torch.float32)
            image = image.unsqueeze(dim=0)
            #T = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC)
            if self.transform is not None:
                image = self.transform(image)
            image = torch.cat([image,image,image],dim=0)
            patientID = dcm_read.PatientID
            #print(image.shape)
        else:
            image = tifffile.imread(f_path)
            image = 2 * ((image - self.min_bound) / (self.max_bound - self.min_bound)) - 1
            #image = utils.RemoveBed(image)

            image = torch.tensor(image, dtype=torch.float32)
            patientID = os.path.basename(f_path)
            #image = image.unsqueeze(dim=0)
            #image = torch.cat([image,image,image],dim=0)

        return image , patientID

    def __len__(self):
        return len(self.samples_paths)

class SourceDataset(data.Dataset):
    def __init__(self, root, lung_coords, ref_dims, min_bound, max_bound, domains, transform=None):
        self.domains = domains
        self.samples_paths, self.targets = self._make_dataset(root)
        self.transform = transform
        self.lung_coords = lung_coords
        self.ref_dims = ref_dims
        self.min_bound = min_bound
        self.max_bound = max_bound

    def _make_dataset(self, root):
        """domains = utils.sort_domains_by_size(root)
        domains = domains[:self.num_domains]
        print("i domini sono i seguenti:", domains)"""
        #domains = os.listdir(root)
        #print(domains)
        dcm_file_names, labels = [], []
        for idx, domain in enumerate(self.domains):
            class_dir = os.path.join(root, domain)
            #print(class_dir)
            cls_dicom_file_names = listdir(class_dir)
            #print(len(cls_dicom_file_names))
            dcm_file_names += cls_dicom_file_names
            labels += [idx] * len(cls_dicom_file_names)
        return dcm_file_names, labels

    def __getitem__(self, index):
        f_dcm_path = self.samples_paths[index]
        label = self.targets[index]

        # read dicom file
        dcm_read = pydicom.dcmread(f_dcm_path)

        # take the pixels array from the first dicom file (image1)
        image = dcm_read.pixel_array
        image = transofrm_image(dcm_read,image,self.lung_coords,self.ref_dims)
        image = 2 * ((image - self.min_bound) / (self.max_bound - self.min_bound)) - 1
        #image = utils.RemoveBed(image) # filter the image in order to remove the bed and have only lungs

        image = torch.tensor(image, dtype=torch.float32)
        
        image = image.unsqueeze(dim=0)
        #T = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC)
        if self.transform is not None:
            image = self.transform(image)
        image = torch.cat([image,image,image],dim=0)
        #print(image.shape)
        
        #print("la shape nel dataloader è:", image.shape)
        
        return image, label

    def __len__(self):
        return len(self.targets)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, lung_coords, ref_dims, min_bound, max_bound, domains, transform=None):
        self.domains = domains
        self.samples_paths, self.targets = self._make_dataset(root)
        self.transform = transform
        self.lung_coords = lung_coords
        self.ref_dims = ref_dims
        self.min_bound = min_bound
        self.max_bound = max_bound

    def _make_dataset(self, root):
        """domains = utils.sort_domains_by_size(root)
        domains = domains[:self.num_domains]
        print(domains)"""
        #domains = os.listdir(root)
        dcm_file_names, dcm_file_names_2, labels = [], [], []
        for idx, domain in enumerate(self.domains):
            class_dir = os.path.join(root, domain)
            cls_dicom_file_names = listdir(class_dir)
            dcm_file_names += cls_dicom_file_names
            dcm_file_names_2 += random.sample(cls_dicom_file_names, len(cls_dicom_file_names))
            labels += [idx] * len(cls_dicom_file_names)
        return list(zip(dcm_file_names, dcm_file_names_2)), labels # the zip function creates tuples containing one element from the first list and
                                                                   # one from the second list

    def __getitem__(self, index):
        f_dcm_path, f_dcm_2_path = self.samples_paths[index]
        label = self.targets[index]

        # read dicom file
        dcm_read = pydicom.dcmread(f_dcm_path)
        dcm_2_read = pydicom.dcmread(f_dcm_2_path)

        # take the pixels array from the first dicom file (image1)
        image1 = dcm_read.pixel_array
        image2 = dcm_2_read.pixel_array

        image1 = transofrm_image(dcm_read,image1,self.lung_coords,self.ref_dims)
        image2 = transofrm_image(dcm_2_read,image2,self.lung_coords,self.ref_dims)

        image1 = 2 * ((image1 - self.min_bound) / (self.max_bound - self.min_bound)) - 1
        #image1 = utils.RemoveBed(image1)
        image2 = 2 * ((image2 - self.min_bound) / (self.max_bound - self.min_bound)) - 1
        #image2 = utils.RemoveBed(image2)
        
        image1 = torch.tensor(image1, dtype=torch.float32)
        image2 = torch.tensor(image2, dtype=torch.float32)

        #T = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC)
        
        image1 = image1.unsqueeze(dim=0)
        image2 = image2.unsqueeze(dim=0)
        if self.transform is not None:
            image1 = self.transform(image1)
            image2 = self.transform(image2)
        #print(image1.shape)

        image1 = torch.cat([image1, image1, image1], dim=0)
        image2 = torch.cat([image2,image2,image2],dim=0)
        #print(image2.shape)
        
        return image1, image2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def cut_image(image, coords, ref_dims, scale_factor):

    img_width = image.shape[0]
    img_height = image.shape[1]
    
    ref_width = int(int(ref_dims['x']) * ref_dims['scale_factor_x'])
    ref_height = int(int(ref_dims['y']) * ref_dims['scale_factor_y'])
    
    # we must have square (width and height must be the same)
    if ref_width > ref_height:
        ref_height = ref_width
    elif ref_height > ref_width:
        ref_width = ref_height
    
    ### STARTING AND ENDING POINT OF X-AXIS (LENGTH OF THE LUNG) ###
    
    start_x = int(int(coords[0]) * scale_factor[0])
    end_x = int(int(coords[2]) * scale_factor[0])
    
    dim_x = (end_x - start_x) + 1
    
    if dim_x != ref_width:
        diff_width = ref_width - dim_x
        #print("diff_width:", diff_width)
        half_diff_width = diff_width // 2 
        #print("half_diff_width:", half_diff_width)
        int_or_not_w = diff_width % 2
        if int_or_not_w == 0:
            new_start_x = start_x - half_diff_width
            if new_start_x < 0:
                start_x = 0
                end_x = end_x + half_diff_width + abs(new_start_x)
            else:
                new_end_x = end_x + half_diff_width
                if new_end_x >= img_width:
                    surplus_x = new_end_x - (img_width - 1)
                    start_x = new_start_x - surplus_x
                    end_x = img_width - 1
                else:
                    start_x = new_start_x
                    end_x = new_end_x
        else:
            new_start_x = start_x - (half_diff_width + 1)
            if new_start_x < 0:
                start_x = 0
                end_x = end_x + half_diff_width + abs(new_start_x)
            else:
                new_end_x = end_x + half_diff_width
                if new_end_x >= img_width:
                    surplus_x = new_end_x - (img_width - 1)
                    start_x = new_start_x - surplus_x
                    end_x = img_width - 1
                else:
                    start_x = new_start_x
                    end_x = new_end_x
    
    
    ### STARTING AND ENDING POJNT OF Y-AXIS (HEIGHT OF THE LUNG) ###
    
    start_y = int(int(coords[1]) * scale_factor[1])
    end_y = int(int(coords[3]) * scale_factor[1])
    
    dim_y = (end_y - start_y) + 1
    
    if dim_y != ref_height:
        diff_heights = ref_height - dim_y
        half_diff_heights = diff_heights // 2 
        int_or_not_h = diff_heights % 2
        if int_or_not_h == 0:
            new_start_y = start_y - half_diff_heights
            if new_start_y < 0:
                start_y = 0
                end_y = end_y + half_diff_heights + abs(new_start_y)
            else:
                new_end_y = end_y + half_diff_heights
                if new_end_y >= img_height:
                    surplus_y = new_end_y - (img_height - 1)
                    start_y = new_start_y - surplus_y
                    end_y = img_height - 1
                else:
                    start_y = new_start_y
                    end_y = new_end_y
        else:
            new_start_y = start_y - (half_diff_heights + 1)
            if new_start_y < 0:
                start_y = 0
                end_y = end_y + half_diff_heights + abs(new_start_y)
            else:
                new_end_y = end_y + half_diff_heights
                if new_end_y >= img_height:
                    surplus_y = new_end_y - (img_height - 1)
                    start_y = new_start_y - surplus_y
                    end_y = img_height - 1
                else:
                    start_y = new_start_y
                    end_y = new_end_y
    
    ### CROP IMAGE ###
    sub_array = image[start_x:end_x, start_y:end_y]
    crop_img = np.full((ref_height, ref_width), -1024)
    # ----------------------------------------------------------------------- #
    """crop_img = np.zeros((ref_height, ref_width), dtype=sub_array.dtype)"""
    # ----------------------------------------------------------------------- #
    insert_start_row = (crop_img.shape[0] - sub_array.shape[0]) // 2
    insert_start_col = (crop_img.shape[1] - sub_array.shape[1]) // 2

    # Inserire il sub_array nell'array grande
    crop_img[insert_start_row:insert_start_row + sub_array.shape[0],
    insert_start_col:insert_start_col + sub_array.shape[1]] = sub_array
    
    return crop_img

def padding(img):
    outer_shape = (512, 512)
    # inner_shape = image_crop.shape
    inner_shape = img.shape
    outer_array = np.full(outer_shape, -1024)
    # inner_array = image_crop
    inner_array = img

    # Calcoliamo gli indici di slicing per inserire l'array interno al centro dell'array esterno
    start_index = tuple((np.array(outer_shape) - np.array(inner_shape)) // 2)
    end_index = tuple(start_index + np.array(inner_shape))

    # Inseriamo l'array interno al centro dell'array esterno
    outer_array[start_index[0]:end_index[0], start_index[1]:end_index[1]] = inner_array

    return outer_array


def transofrm_image(dcm_slice, image, coords, reference_dims):

    image = image.astype(np.int16)
    image[image == -2000] = 0

    # convert to Hounsfield units (HU)
    intercept = dcm_slice.RescaleIntercept
    slope = dcm_slice.RescaleSlope
    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    image = np.array(image, dtype=np.int16)
    
    # resample
    new_spacing = [1.0, 1.0]
    new_spacing = np.array(new_spacing)
    spacing = []
    for num in dcm_slice.PixelSpacing:
        fnum = float(num)
        spacing.append(fnum)
    spacing = np.array(spacing)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    image_resampled = zoom(image, real_resize_factor, mode='nearest')
    patientID = dcm_slice.PatientID
    coords_patient = coords[patientID]
    image_crop = cut_image(image_resampled, coords_patient, reference_dims, real_resize_factor)
    image_padded = padding(image_crop)

    return image_padded


def get_train_loader(root, lung_coords,ref_dims, min_bound, max_bound, domains,
                     which='source', img_size=256, batch_size=8, prob=0.5, num_workers=4, resize=False):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    T = None
    if resize == True:
        T = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC)

    if which == 'source':
        dataset = SourceDataset(root,lung_coords,ref_dims, min_bound, max_bound, domains, transform=T)
    elif which == 'reference':
        dataset = ReferenceDataset(root,lung_coords,ref_dims, min_bound, max_bound, domains, transform=T)
    else:
        raise NotImplementedError

    sampler = _make_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, lung_coords, ref_dims, min_bound, max_bound,
                    img_size=256, batch_size=32,shuffle=True, num_workers=4, drop_last=False, resize=False):
    print('Preparing DataLoader for the evaluation phase...')

    T = None
    if resize == True:
        T = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC)

    dataset = DefaultDataset(root,lung_coords,ref_dims, min_bound, max_bound, img_size, transform=T)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, lung_coords, ref_dims, min_bound, max_bound, domains, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4, resize=False):
    print('Preparing DataLoader for the generation phase...')

    T = None
    if resize == True:
        T = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.BICUBIC)

    dataset = SourceDataset(root,lung_coords,ref_dims, min_bound, max_bound, domains, transform=T)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode=''):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_inputs(self):
        try:
            x, y = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x, y = next(self.iter)
        return x, y

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def __next__(self):
        x, y = self._fetch_inputs()
        #print(f"la shape di x è {x.shape}, dove la size in 0 è {x.size(0)}")
        if self.mode == 'train':
            x_ref, x_ref2, y_ref = self._fetch_refs()
            z_trg = torch.randn(x.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.size(0), self.latent_dim)
            inputs = Munch(x_src=x, y_src=y, y_ref=y_ref,
                           x_ref=x_ref, x_ref2=x_ref2,
                           z_trg=z_trg, z_trg2=z_trg2)
        elif self.mode == 'val':
            x_ref, y_ref = self._fetch_inputs()
            inputs = Munch(x_src=x, y_src=y,
                           x_ref=x_ref, y_ref=y_ref)
        elif self.mode == 'test':
            inputs = Munch(x=x, y=y)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})