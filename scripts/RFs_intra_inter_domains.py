import SimpleITK as sitk
import numpy as np
import os
import csv
import argparse
import pandas as pd
import radiomics
from radiomics import featureextractor
import matplotlib.pyplot as plt
import math
from scipy import stats
import re
from sklearn.model_selection import ShuffleSplit
from preprocessing_base_functions import *

def extract_pixel_spacing(args, patient):
    if patient.split('-')[0] == 'LUNG1':
        PATIENT_FOLDER_PATH = os.path.join(args.NSCLC_dataset, patient)
    else:
        PATIENT_FOLDER_PATH = os.path.join(args.LIDC_IDRI_dataset, patient)
    patient_folder_list = os.listdir(PATIENT_FOLDER_PATH)
    #for num_slice, f_dcm in enumerate(patient_folder_list):
    f_dcm = patient_folder_list[0]
    DCM_PATH = os.path.join(PATIENT_FOLDER_PATH, f_dcm)
    dcm_read = pydicom.dcmread(DCM_PATH)
    pixel_spac_np = list(float(value) for value in dcm_read.PixelSpacing)
    spacing_patient = np.array(pixel_spac_np, dtype=np.float32)
    return spacing_patient

def coords_extraction(args):
    coords_dict = {}
    with open(args.lung_coords_csv, newline='') as coords_csv:
        coords_reader = csv.reader(coords_csv)
        next(coords_reader)
        for row in coords_reader:
            #print(row)
            patientID = (row[0].split('.'))[0]
            start_x = row[1]
            start_y = row[2]
            end_x = row[4]
            end_y = row[5]
            coords_dict[patientID] = [start_x, start_y, end_x, end_y]
    return coords_dict

def reference_dimensions_extractions(args):
    reference_dims = {}

    with open(args.ref_dims_csv, "r") as dims_file:
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


def domains_selection(args):
    columns_names = ["patient_ID", "body_part", "scanner", "kernel", "slice_thickness"]
    #NSCLC_patients_metadata_pd = pd.read_csv(args.NSCLC_metadata, names=columns_names, skiprows=1)
    LIDC_IDRI_patients_metadata_pd = pd.read_csv(args.LIDC_IDRI_metadata, names=columns_names, skiprows=1)
    patients_metadata = LIDC_IDRI_patients_metadata_pd #pd.concat([NSCLC_patients_metadata_pd, LIDC_IDRI_patients_metadata_pd], axis=0)
    if args.domain_type == 'kernel':
        patients_metadata = patients_metadata[['patient_ID', 'kernel']]
    else:
        patients_metadata = patients_metadata[['patient_ID', 'scanner', 'kernel']]

    domains_dict = {}
    for _, row in patients_metadata.iterrows():
        if args.domain_type == 'kernel':
            domain = row['kernel']
        else:
            domain = row['scanner'] + ' -- ' + row['kernel']
        patient = row['patient_ID']
        if domain not in domains_dict:
            domains_dict[domain] = [patient]
        else:
            domains_dict[domain].append(patient)

    domains_dict_support = domains_dict.copy()

    for dom in domains_dict_support:
        if len(domains_dict[dom]) > 90 or len(domains_dict[dom]) < 55:
            del domains_dict[dom]

    return domains_dict

def RFs_extraction(args, group, extractor, coords_dict, ref_dims_dict, RFs_per_slice = None, dom=None):
    radiomic_features_group_dict = {}
    for patient in group:
        if patient.split('-')[0] == 'LUNG1':
            PATIENT_FOLDER_PATH = os.path.join(args.NSCLC_dataset, patient)
        else:
            PATIENT_FOLDER_PATH = os.path.join(args.LIDC_IDRI_dataset, patient)
        pixel_spacing = extract_pixel_spacing(args, patient)
        if pixel_spacing[0] < 0.8:
            patient_folder_list = os.listdir(PATIENT_FOLDER_PATH)
            radiomic_features_patient_dict = {}
            for num_slice, f_dcm in enumerate(patient_folder_list):
                DCM_PATH = os.path.join(PATIENT_FOLDER_PATH, f_dcm)

                # read dicom file
                dcm_read = pydicom.dcmread(DCM_PATH)
                # take the pixels array from the first dicom file (image1)
                # dcm_px = dcm_read.pixel_array
                px_slices = get_pixels_hu([dcm_read])

                # resample
                image_resampled, real_resize_factor = resample(px_slices, [dcm_read], [0.8, 0.8]) #[1.0, 1.0])
                coords_patient = coords_dict[patient]
                image_crop = cut_image(image_resampled, coords_patient, ref_dims_dict, real_resize_factor)

                # transform the pixel array into a SimpleITK image
                img = sitk.GetImageFromArray(image_crop)
                mask = img > -1024
                try:
                    features_slice = extractor.execute(img, mask)
                    for feature_name in features_slice.keys():
                        if feature_name.startswith("original_"):  # Filter only radiomic features
                            if RFs_per_slice is not None:
                                RFs_per_slice.append([patient, num_slice, dom, feature_name, features_slice[feature_name]])
                            if feature_name in radiomic_features_patient_dict:
                                radiomic_features_patient_dict[feature_name].append(features_slice[feature_name].item())
                            else:
                                radiomic_features_patient_dict[feature_name] = [features_slice[feature_name].item()]
                except ValueError as e:
                    print(f"Error extracting features for patient {patient}: {str(e)}")
                    continue

            ### now, we have to obtain the mean of each list computed, in order to have for each feature one value for each volume ###
            for feature in radiomic_features_patient_dict:
                radiomic_features_patient_dict[feature] = np.array(radiomic_features_patient_dict[feature])
                volume_feature = np.mean(radiomic_features_patient_dict[feature])
                if feature not in radiomic_features_group_dict:
                    radiomic_features_group_dict[feature] = [volume_feature]
                else:
                    radiomic_features_group_dict[feature].append(volume_feature)

    return radiomic_features_group_dict

def RFs_dict_creation(RFs_csv):
    data = pd.read_csv(RFs_csv)

    #mask = data['patientID'].isin(volumes_of_interest)

    #data = data[mask]

    # print(data)

    features_mean = data.drop("slice_num", axis=1).groupby(['patientID', 'domain', 'feature_name']).mean()
    domains_grouped = features_mean.groupby('domain')

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
            if f not in features_dict:
                features_dict[f] = [fr_vector]
            else:
                features_dict[f].append(fr_vector)

    return features_dict, domains

def T_test(args, g1_RFs_dict, g2_RFs_dict, T_test_dict, domains):
    print("sono nella funzione T test")

    domains_comparison = []

    for f in g1_RFs_dict:

        ### COMPUTE THE T-TEST FOR THE FEATURE F ###

        if args.type == 'intra':
            RF_g1 = g1_RFs_dict[f]
            RF_g2 = g2_RFs_dict[f]

            _, p_value = stats.ttest_ind(RF_g1, RF_g2)

            if f not in T_test_dict:
                T_test_dict[f] = [p_value]
            else:
                T_test_dict[f].append(p_value)

        else:
            for i in range(len(g1_RFs_dict[f])):
                for j in range(i + 1, len(g1_RFs_dict[f])):
                    comparison = domains[i] + '_vs_' + domains[j]
                    if comparison not in domains_comparison:
                        domains_comparison.append(comparison)
                    #print(domains_comparison)
                    fr_dom_1 = g1_RFs_dict[f][i]
                    fr_dom_2 = g1_RFs_dict[f][j]

                    _, p_value = stats.ttest_ind(fr_dom_1, fr_dom_2)

                    if f not in T_test_dict:
                        T_test_dict[f] = [p_value]
                    else:
                        T_test_dict[f].append(p_value)

    return domains_comparison

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--NSCLC_dataset', type=str,
                        default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets/NSCLC_Radiomics/NSCLC_Radiomics_CTs',
                        help='Directory of NSCLC (only CTs) dataset')
    parser.add_argument('--LIDC_IDRI_dataset', type=str,
                        default='/mimer/NOBACKUP/groups/snic2022-5-277/assolito/LIDC_IDRI_CTs',
                        help='Directory of LIDC-IDRI (only CTs) dataset')
    parser.add_argument('--NSCLC_metadata', type=str,
                        default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets/csv_files/NSCLC_patients_metadata.csv',
                        help='Directory of NSCLC metadata')
    parser.add_argument('--LIDC_IDRI_metadata', type=str,
                        default='/mimer/NOBACKUP/groups/snic2022-5-277/assolito/csv_files/LIDC_IDRI_patients_metadata.csv',
                        help='Directory of LIDC-IDRI metadata')
    parser.add_argument('--csv_folder', type=str,
                        default='/mimer/NOBACKUP/groups/snic2022-5-277/assolito/csv_files',
                        help='Directory of csv folder')
    parser.add_argument('--lung_coords_csv', type=str,
                            default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets/csv_files/lung_coordinates.csv',
                            help='Directory of lung coordinates csv')
    parser.add_argument('--ref_dims_csv', type=str,
                                default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets/csv_files/reference_dimensions.txt',
                                help='Directory of reference dimensions csv')
    parser.add_argument('--rf_per_slice', type=bool,
                                default=False,
                                help='It indicates if radiomic features for each slice are already available or not')
    parser.add_argument('--type', type=str,
                                    default='intra',
                                    choices=['intra', 'inter'],
                                    help='used to understand if we want to compare features among different domains or among different patients inside the same domain')
    parser.add_argument('--domain_type', type=str,
                        default='kernel',
                        choices=['kernel', 'scanner-kernel'],
                        help='how aggregate data in domains')

    args = parser.parse_args()

    if args.rf_per_slice:
        RFs_csv = os.path.join(args.csv_folder, 'radiomic_features_per_slice.csv')
        features_dict, domains = RFs_dict_creation(RFs_csv)
        #print(features_dict.keys())
        T_test_dict = {}
        domains_comparison = T_test(args, features_dict, None, T_test_dict, domains)
        print(domains_comparison)
        domains = domains_comparison
    else:
        domains_dict = domains_selection(args)
        print(domains_dict)

        coords_dict = coords_extraction(args)
        reference_dims_dict = reference_dimensions_extractions(args)

        # definition of the radiomic features extractor
        extractor = featureextractor.RadiomicsFeatureExtractor()
        if 'shape2D' not in extractor.enabledFeatures:
            extractor.enabledFeatures['shape2D'] = []
        extractor.settings['force2D'] = True

        # split each domain in two groups and compare their FRs to understand if there is homogeneity inside the same domain (understand if there are FRs that depend on
        # the specific patient)
        RFs_per_slice = []
        T_test_dict = {}
        radiomic_features_dict = {}
        domains = []
        for domain in domains_dict:
            domains.append(domain)
            patients = np.array(domains_dict[domain])
            if args.type == 'intra':
                group_1 = []
                group_2 = []
                rs = ShuffleSplit(n_splits=1, test_size=.5, random_state=6)
                for i, (group_1_index, group_2_index) in enumerate(rs.split(domains_dict[domain])):
                    group_1, group_2 = patients[group_1_index], patients[group_2_index]
                print("patients in group_1 are:", group_1)
                print("patients in group_2 are:", group_2)
                group_1_RFs = RFs_extraction(args, group_1, extractor, coords_dict, reference_dims_dict)
                group_2_RFs = RFs_extraction(args, group_2, extractor, coords_dict, reference_dims_dict)
                T_test(args, group_1_RFs, group_2_RFs, T_test_dict, domains)
            else:
                dom_RFs = RFs_extraction(args, domains_dict[domain], extractor, coords_dict, reference_dims_dict, RFs_per_slice, domain)
                for f in dom_RFs:
                    if f not in radiomic_features_dict:
                        radiomic_features_dict[f] = [dom_RFs[f]]
                    else:
                        radiomic_features_dict[f].append(dom_RFs[f])
        if RFs_per_slice:
            radiomic_features_per_slice_csv = os.path.join(args.csv_folder, "radiomic_features_per_slice_no_outliers_(scanner_kernel_dataset).csv")

            with open(radiomic_features_per_slice_csv, "w", newline="") as file_csv:
                writer = csv.writer(file_csv)

                ### csv header ###
                writer.writerow(["patientID", "slice_num", "domain", "feature_name", "feature_value"])
                writer.writerows(RFs_per_slice)

        #print(T_test_dict)
        if not T_test_dict:
            domains_comparison = T_test(args, radiomic_features_dict, None, T_test_dict, domains)
            domains = domains_comparison
    T_test_pd = pd.DataFrame(T_test_dict, index=domains)
    T_test_pd = T_test_pd.T
    T_test_csv = os.path.join(args.csv_folder, "T_test_" + args.type + "_domains_no_outliers_(scanner_kernel_dataset).csv")
    T_test_pd.to_csv(T_test_csv)





