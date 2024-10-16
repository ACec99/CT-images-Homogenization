import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pydicom
import os
import csv
import re
import pickle

import scipy.ndimage
from  scipy.ndimage import zoom , rotate
import matplotlib.pyplot as plt

from mask_creation import create_lungs_mask
from preprocessing_base_functions import *

def extraxt_masks(seg_path, patients_dict):
    sorted_keys_scans = sorted(patients_dict.keys())
    patients_created_mask = []

    for patientID in sorted_keys_scans:

        mask_folder_path = os.path.join(seg_path, patientID + ".npy")
        if not os.path.isfile(mask_folder_path):
            path_scan = patients_CT_scans_paths[patientID]
            scan = load_scan(path_scan)
            #num_slices = len(scan)
            lung_mask = create_lungs_mask(patientID, scan)
            # lung_mask, _ = resample(lung_mask, scan, [3.0,1.0,1.0])
            np.save(mask_folder_path, lung_mask)
            patients_created_mask.append(patientID)

    return patients_created_mask

if __name__ == "__main__":

    SEG_MASKS_PATH = "/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets/segmentation_masks"
    LIDC_IDRI_CTs_PATH = "/mimer/NOBACKUP/groups/snic2022-5-277/assolito/LIDC_IDRI_CTs"
    LIDC_IDRI_CTs_list = os.listdir(LIDC_IDRI_CTs_PATH)

    patients_CT_scans_paths = {}
    for patient in LIDC_IDRI_CTs_list:
        patient_dicom_files_path = os.path.join(LIDC_IDRI_CTs_PATH, patient)
        patients_CT_scans_paths[patient] = patient_dicom_files_path

    patients_created_mask = extraxt_masks(SEG_MASKS_PATH, patients_CT_scans_paths)
    print(patients_created_mask)