import numpy as np
import pydicom
import os
import csv
import argparse
import pandas as pd
from RFs_intra_inter_domains import domains_selection
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
                        default='/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/datasets/csv_files',
                        help='Directory of csv folder')

    args = parser.parse_args()

    domains_dict = domains_selection(args)

    spacing_details_dict = {}
    #count_patients_per_domain = {}
    for domain in domains_dict:
        spacing_LIDC_IDRI_domain = []
        spacing_NSCLC_domain = []
        #count_LIDC_IDRI = 0
        #count_NSCLC = 0
        for patient in domains_dict[domain]:
            spacing_patient = extract_pixel_spacing(args, patient)

            if patient.split('-')[0] == 'LUNG1':
                #count_NSCLC += 1
                spacing_NSCLC_domain.append(spacing_patient)
            else:
                #count_LIDC_IDRI += 1
                # ------------------ added the condition to remove all the volumes with a pixel spacing >= 0.8 in order to understand the maximum pixel spacing without outliers ------------------ #
                if spacing_patient[0] < 0.8:
                    spacing_LIDC_IDRI_domain.append(spacing_patient)
                # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- #
        # ------------- in the case in which there are no NSCLC patients ---------------- #
        if spacing_NSCLC_domain == []:
            spacing_NSCLC_domain = [[0.0, 0.0]]
        if spacing_LIDC_IDRI_domain == []:
            spacing_LIDC_IDRI_domain = [[0.0, 0.0]]
        # -------------------------------------------------------------------------------- #
        #print(f"domain {domain} spacing list for LIDC_IDRI is {spacing_LIDC_IDRI_domain} ")
        #print("\n \n \n")

        spacing_NSCLC_domain = np.array(spacing_NSCLC_domain)
        spacing_LIDC_IDRI_domain = np.array(spacing_LIDC_IDRI_domain)

        # ------------- max value of spacing for each dataset type --------------- #
        spacing_LIDC_IDRI_domain_max = np.max(spacing_LIDC_IDRI_domain, axis=0)
        spacing_NSCLC_domain_max = np.max(spacing_NSCLC_domain, axis=0)
        # ------------------------------------------------------------------------ #

        # ------------- min value of spacing for each dataset type --------------- #
        spacing_LIDC_IDRI_domain_mean = np.mean(spacing_LIDC_IDRI_domain, axis=0)
        spacing_NSCLC_domain_mean = np.mean(spacing_NSCLC_domain, axis=0)
        # ------------------------------------------------------------------------ #

        spacing_details_dict[domain] = {'max_LIDC_IDRI':spacing_LIDC_IDRI_domain_max, 'max_NSCLC': spacing_NSCLC_domain_max ,'mean_LIDC_IDRI':spacing_LIDC_IDRI_domain_mean, 'mean_NSCLC':spacing_NSCLC_domain_mean}

        """if 'LIDC_IDRI' not in count_patients_per_domain:
            count_patients_per_domain['LIDC_IDRI'] = [count_LIDC_IDRI]
        else:
            count_patients_per_domain['LIDC_IDRI'].append(count_LIDC_IDRI)

        if 'NSCLC' not in count_patients_per_domain:
            count_patients_per_domain['NSCLC'] = [count_NSCLC]
        else:
            count_patients_per_domain['NSCLC'].append(count_NSCLC)"""

    df_spacing = pd.DataFrame(spacing_details_dict).T
    #df_count = pd.DataFrame(count_patients_per_domain, index=domains_dict.keys())
    #print(df)
    csv_spacing_path = os.path.join(args.csv_folder, 'spacing_details_no_outliers.csv')
    df_spacing.to_csv(csv_spacing_path)
    """csv_count_path = os.path.join(args.csv_folder, 'datasets_occurrences.csv')
    df_count.to_csv(csv_count_path)"""






