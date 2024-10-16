import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import csv
import re
import pydicom
from preprocessing_base_functions import *


def find_max_resample_factor(dataset_path):
    print("I'm searching the maximum resample factor in the LIDC-IDRI dataset")
    LIDC_IDRI_list = os.listdir(dataset_path)
    max_resample_factor = []
    for patient in LIDC_IDRI_list:
        patient_dcms_path = os.path.join(dataset_path, patient)
        patient_scan = load_scan(patient_dcms_path)
        patient_volume = get_pixels_hu(patient_scan)
        patient_volume_resampled, resample_factor = resample(patient_volume, patient_scan, [3.0,1.0,1.0])
        if max_resample_factor == [] or resample_factor[1] > max_resample_factor[1]:
            max_resample_factor = resample_factor

    return max_resample_factor

def extract_reference_dims(file_path):
    print("I'm building the dictionary containing the reference dimensions")
    reference_dims = {}
    with open(file_path, "r") as dims_file:
        # Read the file line by line
        for line in dims_file:
            line_list = line.strip().split(':')
            axis = line_list[0]
            if re.search('scale_factor', axis) != None:
                ref_dim = float(line_list[1])
            else:
                ref_dim = int(line_list[1])
            reference_dims[axis] = ref_dim
    return reference_dims

def find_lung_coordinates(lung_coords_path, masks_path): # modifica successivamente i parametri con args (argparse)
    #lung_coords_pd = pd.read_csv(lung_coords_path, names=['patient_ID', 'Start_X', 'Start_Y', 'Start_Z', 'End_X', 'End_Y', 'End_Z', 'Half_Z'], skiprows=1)

    masks_names_list = os.listdir(masks_path)

    max_all_scans_lung_x = 0
    max_all_scans_lung_y = 0
    max_all_scans_lung_z = 0

    for mask_name in masks_names_list:
        patientID = (mask_name.split('.'))[0]
        #if patientID not in lung_coords_pd['patient_ID'].tolist():
        MASK_FOLDER = os.path.join(masks_path, mask_name)

        mask = np.load(MASK_FOLDER)

        max_scan_lung_x = 0
        max_scan_lung_y = 0
        max_scan_lung_z = 0

        max_start_lung_x = 0
        max_start_lung_y = 0
        max_start_lung_z = 0

        max_end_lung_x = 0
        max_end_lung_y = 0
        max_end_lung_z = 0

        half_lung_z = 0
        max_tot_pixels = 0

        found_lung = False

        for idx_slice, mask_slice in enumerate(mask):

            max_slice_lung_x = 0
            start_coord_lung_x = float('inf')
            end_coord_lung_x = 0

            max_slice_lung_y = 0
            start_coord_lung_y = 0
            end_coord_lung_y = 0

            upper_lung = False

            tot_pixels = 0

            for row in range(mask_slice.shape[0]):
                current_lung_x = np.count_nonzero(mask_slice[row] == True)
                tot_pixels += current_lung_x
                if current_lung_x > 0:

                    if upper_lung == False:
                        start_coord_lung_y = row
                        upper_lung = True
                    end_coord_lung_y = row  # in this way, it increments every time and at the end we have both
                    # the starting and the ending point of the lung y-axis, avoiding every
                    # error due to possible discontinuites in the lung image

                    lung_coord_x = np.where(mask_slice[row] == True)[0]
                    if lung_coord_x[0] < start_coord_lung_x:
                        start_coord_lung_x = lung_coord_x[0]
                    if lung_coord_x[-1] > end_coord_lung_x:
                        end_coord_lung_x = lung_coord_x[-1]

            max_slice_lung_x = (end_coord_lung_x - start_coord_lung_x) + 1

            if tot_pixels > max_tot_pixels:
                max_tot_pixels = tot_pixels
                half_lung_z = idx_slice

            if found_lung == False and tot_pixels > 0:
                found_lung = True
                max_start_lung_z = idx_slice
            elif found_lung == True and tot_pixels == 0:
                found_lung = False
                max_end_lung_z = idx_slice - 1

            max_slice_lung_y = (end_coord_lung_y - start_coord_lung_y) + 1
            if max_slice_lung_y > max_scan_lung_y:
                max_scan_lung_y = max_slice_lung_y
                max_start_lung_y = start_coord_lung_y
                max_end_lung_y = end_coord_lung_y

            if max_slice_lung_x > 0:
                max_scan_lung_z += 1

            if max_slice_lung_x > max_scan_lung_x:
                max_scan_lung_x = max_slice_lung_x
                max_start_lung_x = start_coord_lung_x
                max_end_lung_x = end_coord_lung_x

        if max_scan_lung_x > max_all_scans_lung_x:
            max_all_scans_lung_x = max_scan_lung_x

        if max_scan_lung_y > max_all_scans_lung_y:
            max_all_scans_lung_y = max_scan_lung_y

        # non vado a vedere la profondità (asse z) perchè sto analizzando tutte maschere ricreate (LIDC-IDRI dataset non formisce maschere dei polmoni)
        """if max_scan_lung_z > max_all_scans_lung_z and patientID not in reproduced_mask_patients_list:
            max_all_scans_lung_z = max_scan_lung_z"""

        # DECOMMENTALO POI
        #lung_coords_pd.loc[len(lung_coords_pd)] = [patientID, max_start_lung_x, max_start_lung_y, max_start_lung_z,
        #                                           max_end_lung_x, max_end_lung_y, max_end_lung_z, half_lung_z]

    #lung_coords_pd.to_csv(lung_coords_path, index=False)
    return max_all_scans_lung_x, max_all_scans_lung_y, max_all_scans_lung_z


if __name__ == '__main__':

    ### PATHS ###
    DATASET_FOLDER = "/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets"
    MASKS_FOLDER = DATASET_FOLDER + "/segmentation_masks"
    LUNG_COORDS_CSV_PATH = os.path.join(DATASET_FOLDER, "csv_files/lung_coordinates.csv")
    REF_DIMS_PATH = os.path.join(DATASET_FOLDER, "csv_files/reference_dimensions.txt")
    LIDC_IDRI_PATH = "/mimer/NOBACKUP/groups/snic2022-5-277/assolito/LIDC_IDRI_CTs"
    ###

    reference_dims = extract_reference_dims(REF_DIMS_PATH)
    print(f'the dictionary containing the reference dimensions is:', reference_dims)
    # modify, if necessary, the scale factor
    std_resample_factor = [reference_dims[axis] for axis in reference_dims if re.search('scale_factor', axis)]
    print(f'the standard resample factor is {std_resample_factor}')
    LIDC_IDRI_max_resample_factor = find_max_resample_factor(LIDC_IDRI_PATH)

    if LIDC_IDRI_max_resample_factor[1] > std_resample_factor[1]:
        #std_resample_factor = LIDC_IDRI_max_resample_factor
        reference_dims['scale_factor_z'] = LIDC_IDRI_max_resample_factor[0]
        reference_dims['scale_factor_x'] = LIDC_IDRI_max_resample_factor[1]
        reference_dims['scale_factor_y'] = LIDC_IDRI_max_resample_factor[2]

    max_all_scans_lung_x, max_all_scans_lung_y, max_all_scans_lung_z = find_lung_coordinates(LUNG_COORDS_CSV_PATH, MASKS_FOLDER)

    if max_all_scans_lung_x > reference_dims['x']:
        reference_dims['x'] = max_all_scans_lung_x
    if max_all_scans_lung_y > reference_dims['y']:
        reference_dims['y'] = max_all_scans_lung_y

    with open(REF_DIMS_PATH, "w") as file:
        file.write(f"x: {reference_dims['x']} \n y: {reference_dims['y']} \n z: {reference_dims['z']} \n scale_factor_x: {reference_dims['scale_factor_x']} \n scale_factor_y: {reference_dims['scale_factor_y']} \n scale_factor_z: {reference_dims['scale_factor_z']}")




