import os
import shutil
import csv
import re
import random
import pandas as pd
import numpy as np
import argparse
import pydicom

# from iterstrat.ml_stratifiers import IterativeStratification

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

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

def domains_selection_per_patient(args):
    domains_dict = {}
    if args.stargan_data:
        for folder in os.listdir(args.StarGAN_dataset):
            folder_path = os.path.join(args.StarGAN_dataset, folder)
            for dom_folder in os.listdir(folder_path):
                kernel_domain = dom_folder.split(' -- ')[1]
                dom_folder_path = os.path.join(folder_path, dom_folder)
                for patient_file in os.listdir(dom_folder_path):
                    patient_slice = patient_file.split('.')[0]
                    patient = patient_slice.rsplit('-', 1)[0]
                    domains_dict[patient] = kernel_domain
    else:
        columns_names = ["patient_ID", "body_part", "scanner", "kernel", "slice_thickness"]
        NSCLC_patients_metadata_pd = pd.read_csv(args.NSCLC_metadata, names=columns_names, skiprows=1)
        LIDC_IDRI_patients_metadata_pd = pd.read_csv(args.LIDC_IDRI_metadata, names=columns_names, skiprows=1)
        patients_metadata = pd.concat([NSCLC_patients_metadata_pd, LIDC_IDRI_patients_metadata_pd], axis=0)

        dom_selectors = (args.domain_type).split('-')
        cols_to_consider = ['patient_ID'] + dom_selectors
        patients_metadata = patients_metadata[cols_to_consider]

        for _, row in patients_metadata.iterrows():
            if row['kernel'] != "-":
                domain = row['kernel']
                if len(row) == 3:
                    scanner = row['scanner']
                    scanner = scanner.replace('/', '-')
                    domain_temp = scanner + " -- " + domain
                    domain = domain_temp
                patient = row['patient_ID']
                domains_dict[patient] = domain

    domains_dict_support = domains_dict.copy()

    count_domain_dict = {}
    for p in domains_dict:
        domain = domains_dict[p]
        if domain not in count_domain_dict:
            count_domain_dict[domain] = 1
        else:
            count_domain_dict[domain] += 1
    print(count_domain_dict)

    #domains = [domain for domain in domains_dict]
    return count_domain_dict, domains_dict

def split(patients, domains):

    X_train = []
    y_train = []

    X_test = []
    y_test = []

    X_valid = []
    y_valid = []

    skf_train_test = StratifiedShuffleSplit(n_splits=1, test_size=0.1, train_size=0.9, random_state=0)
    for i, (train_idx, test_idx) in enumerate(skf_train_test.split(patients, domains)):
        X_train, X_test = patients[train_idx], patients[test_idx]
        y_train, y_test = domains[train_idx], domains[test_idx]

    skf_train_valid = StratifiedShuffleSplit(n_splits=1, test_size=0.2, train_size=0.8, random_state=0)
    for i, (train_idx, valid_idx) in enumerate(skf_train_valid.split(X_train, y_train)):
        X_train, X_valid = X_train[train_idx], X_train[valid_idx]
        y_train, y_valid = y_train[train_idx], y_train[valid_idx]

    return X_train, X_test, X_valid, y_train, y_test, y_valid


def dataset_suddivision(args, count_domain_dict, domains_dict, new_folder_path):

    patients = []
    domains = []
    for patient in domains_dict:
        dom = domains_dict[patient]
        #if count_domain_dict[dom] >= args.min_num_patients:  # in this way we keep out domains that contain few patients
        #print(args.max_num_patients
        # , args.min_num_patients)
        if args.min_num_patients <= count_domain_dict[dom] <= args.max_num_patients: # because I want to take a precise interval of domains in order to have a more homogeneous dataset
            pixel_spacing = extract_pixel_spacing(args, patient)
            if pixel_spacing[0] < 0.8:
                patients.append(patient)
                domains.append(dom)
    patients = np.array(patients)
    domains = np.array(domains)

    print(patients)
    print(domains)

    X_train, X_test, X_valid, y_train, y_test, y_valid = split(patients, domains)

    print(X_train, len(X_train))
    print(y_train, len(y_train))

    #df_split_path = os.path.join(args.csv_folder,"patients_splitted_by_kernel.csv")
    #df_split_path = os.path.join(args.csv_folder,"patients_splitted_by_scanner_kernel.csv")
    df_split_path = os.path.join(args.csv_folder,"patients_splitted_by_scanner_kernel_px_reduced.csv")

    with open(df_split_path, "w", newline="") as file_csv:
        writer = csv.writer(file_csv)

        writer.writerow(["Patient_ID", "folder"])

        for idx_tr, patient_tr in enumerate(X_train):
            writer.writerow([patient_tr, 'train'])

            domain = domains_dict[patient_tr]

            STARGAN_DOMAIN_PATH = new_folder_path + "/train/" + domain
            if not os.path.exists(STARGAN_DOMAIN_PATH):
                os.makedirs(STARGAN_DOMAIN_PATH)

            if patient_tr.split('-')[0] == 'LUNG1':
                PATIENT_DCMs_PATH = os.path.join(args.NSCLC_dataset, patient_tr)
            else:
                PATIENT_DCMs_PATH = os.path.join(args.LIDC_IDRI_dataset, patient_tr)

            patient_dcms_list = os.listdir(PATIENT_DCMs_PATH)
            for f_dcm in patient_dcms_list:
                F_DCM_PATH = os.path.join(PATIENT_DCMs_PATH, f_dcm)
                shutil.copy(F_DCM_PATH, STARGAN_DOMAIN_PATH)
                # rename the files in order to have in their name the correspondent patient
                old_name_path = os.path.join(STARGAN_DOMAIN_PATH, f_dcm)
                f_dcm_split = f_dcm.split('.')
                f_dcm_name = f_dcm_split[0].split('-')
                new_name = patient_tr + "-" + f_dcm_name[1] + ".dcm"
                new_name_path = os.path.join(STARGAN_DOMAIN_PATH, new_name)
                os.rename(old_name_path, new_name_path)

        for idx_te, patient_te in enumerate(X_test):
            writer.writerow([patient_te, 'test'])

            domain = domains_dict[patient_te]

            STARGAN_DOMAIN_PATH = new_folder_path + "/test/" + domain
            if not os.path.exists(STARGAN_DOMAIN_PATH):
                os.makedirs(STARGAN_DOMAIN_PATH)

            if patient_te.split('-')[0] == 'LUNG1':
                PATIENT_DCMs_PATH = os.path.join(args.NSCLC_dataset, patient_te)
            else:
                PATIENT_DCMs_PATH = os.path.join(args.LIDC_IDRI_dataset, patient_te)

            patient_dcms_list = os.listdir(PATIENT_DCMs_PATH)
            for f_dcm in patient_dcms_list:
                F_DCM_PATH = os.path.join(PATIENT_DCMs_PATH, f_dcm)
                shutil.copy(F_DCM_PATH, STARGAN_DOMAIN_PATH)
                # rename the files in order to have in their name the correspondent patient
                old_name_path = os.path.join(STARGAN_DOMAIN_PATH, f_dcm)
                f_dcm_split = f_dcm.split('.')
                f_dcm_name = f_dcm_split[0].split('-')
                new_name = patient_te + "-" + f_dcm_name[1] + ".dcm"
                new_name_path = os.path.join(STARGAN_DOMAIN_PATH, new_name)
                os.rename(old_name_path, new_name_path)

        for idx_val, patient_val in enumerate(X_valid):
            writer.writerow([patient_val, 'valid'])

            domain = domains_dict[patient_val]

            STARGAN_DOMAIN_PATH = new_folder_path + "/valid/" + domain
            if not os.path.exists(STARGAN_DOMAIN_PATH):
                os.makedirs(STARGAN_DOMAIN_PATH)

            if patient_val.split('-')[0] == 'LUNG1':
                PATIENT_DCMs_PATH = os.path.join(args.NSCLC_dataset, patient_val)
            else:
                PATIENT_DCMs_PATH = os.path.join(args.LIDC_IDRI_dataset, patient_val)

            patient_dcms_list = os.listdir(PATIENT_DCMs_PATH)
            for f_dcm in patient_dcms_list:
                F_DCM_PATH = os.path.join(PATIENT_DCMs_PATH, f_dcm)
                shutil.copy(F_DCM_PATH, STARGAN_DOMAIN_PATH)
                # rename the files in order to have in their name the correspondent patient
                old_name_path = os.path.join(STARGAN_DOMAIN_PATH, f_dcm)
                f_dcm_split = f_dcm.split('.')
                f_dcm_name = f_dcm_split[0].split('-')
                new_name = patient_val + "-" + f_dcm_name[1] + ".dcm"
                new_name_path = os.path.join(STARGAN_DOMAIN_PATH, new_name)
                os.rename(old_name_path, new_name_path)

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
    parser.add_argument('--min_num_patients', type=int,
                        default=70,
                        help='the minimum number of patients in a domain in order to make the domain considerable')
    parser.add_argument('--max_num_patients', type=int,
                            default=float('inf'),
                            help='the maximum number of patients in a domain in order to make the domain considerable')

    # ------------------ TEMPORANEO (to obtain kernel subdivision from the scanner-kernel one already existing -------------- #
    parser.add_argument('--stargan_data', type=bool,
                        default=False,
                        help='whether we are taking data from a model dataset already created')
    parser.add_argument('--StarGAN_dataset', type=str,
                        default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/StarGAN_v2/StarGAN_dataset_expanded',
                        help='Directory of StarGAN dataset with scanner-kernel domains')
    # ----------------------------------------------------------------------------------------------------------------------- #

    parser.add_argument('--domain_type', type=str,
                        default='kernel',
                        choices=['kernel', 'scanner-kernel'],
                        help='how aggregate data in domains')

    args = parser.parse_args()
    
    count_domain_dict, domains_dict = domains_selection_per_patient(args)

    #new_folder_path = '/mimer/NOBACKUP/groups/snic2022-5-277/assolito/StarGAN_kernel_dataset'
    #new_folder_path = '/mimer/NOBACKUP/groups/snic2022-5-277/assolito/StarGAN_scanner_kernel_dataset'
    new_folder_path = '/mimer/NOBACKUP/groups/snic2022-5-277/assolito/StarGAN_scanner_kernel_px_reduced_dataset'


    if not os.path.exists(new_folder_path):
        os.mkdir(new_folder_path)

    dataset_suddivision(args, count_domain_dict, domains_dict, new_folder_path)




