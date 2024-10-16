import pydicom
import os
import csv
import re
import pandas as pd
import shutil
from tqdm import tqdm

def delete_xml(folder_path):
    folder_list = os.listdir(folder_path)
    clean_list = []
    for file in folder_list:
        if not file.endswith(".xml"):
            #os.remove(os.path.join(folder_path, file)) # per ora non voglio eliminarli dalla cartella, voglio solo escluderli dalla lista (però nel dataset finale non li vorrò)
            clean_list.append(file)
    return clean_list


def copy_tree_with_extension(src, dst):
    if not os.path.exists(dst):
        os.mkdir(dst)

    for file in os.listdir(src):
        if file.endswith('.dcm'):
            src_file = os.path.join(src, file)
            dst_file = os.path.join(dst, file)
            shutil.copy2(src_file, dst_file)

### crea la cartella del dataset LIDC_IDRI con all'interno solo i pazienti scanzionati con i kernel desiderati (domini con più di 100 volumi)
### per ciascuno dei quali sono inseriti solo i CT dicom files
def extract_and_save_CTs(csv_path, input_path, output_path, kof, columns_names, output_csv):
    features_table = pd.read_csv(csv_path, header=None, names=columns_names)
    # print(features_table.head())
    CT_table = features_table.loc[features_table['Modality'] == 'CT']

    with open(output_csv, "w", newline="") as file_csv:
        writer = csv.writer(file_csv)
        writer.writerow(["patient_ID", "body_part", "scanner", "kernel", "slice_thickness"])

        for index, row in CT_table.iterrows():
            #total_patients += 1
            #contrast_check = False
            patient_ID = row['Subject ID']
            file_location_path_orig = row['Download Timestamp']
            file_location_path = re.sub('(^.)', '', file_location_path_orig)
            file_location_path = file_location_path.replace("\\", "/")
            # file_location_path = file_location_path.replace("\\")
            dicom_files_path = input_path + file_location_path
            list_dicom_files = delete_xml(dicom_files_path)
            f_dicom = list_dicom_files[0]
            dicom_content = pydicom.dcmread(dicom_files_path + "/" + f_dicom)

            try:
                kernel = dicom_content.ConvolutionKernel
                if kernel in kof:
                    contrast_check = False
                    for elem in dicom_content:
                        tag_name = (elem.name).lower()
                        if re.search('contrast', tag_name) is not None:
                            #print(patient_ID, tag_name)
                            contrast_check = True
                            break
                    if contrast_check == False:
                        body_part = dicom_content.BodyPartExamined if 'BodyPartExamined' in dicom_content else 'Unknown'
                        scanner_brand = dicom_content.Manufacturer
                        scanner_type = dicom_content.ManufacturerModelName
                        scanner = scanner_brand + " " + scanner_type
                        slice_thickness = dicom_content.SliceThickness if 'SliceThickness' in dicom_content else 'Unknown'
                        writer.writerow([patient_ID, body_part, scanner, kernel, slice_thickness])
                        """patient_out_path = os.path.join(output_path, patient_ID)
                        #os.mkdir(patient_out_path)
                        #shutil.copytree(dicom_files_path, patient_out_path)
                        copy_tree_with_extension(dicom_files_path, patient_out_path)"""

            except (ValueError, TypeError) as e:
                print(f"Error: {e}")


if __name__ == '__main__':
    #LIDC_IDRI_folder_path = "/mimer/NOBACKUP/groups/snic2022-5-277/assolito/LIDC_IDRI_dataset"
    CSV_FILES = "/mimer/NOBACKUP/groups/snic2022-5-277/assolito/csv_files"
    LIDC_IDRI_metadata_path = CSV_FILES + "/LIDC_IDRI_dataset_metadata.csv"

    LIDC_IDRI_CTs_dataset = "/mimer/NOBACKUP/groups/snic2022-5-277/assolito/LIDC_IDRI_CTs"
    if not os.path.exists(LIDC_IDRI_CTs_dataset):
        os.mkdir(LIDC_IDRI_CTs_dataset)


    ### number of volums (patients) for each kernel type (domain) ###
    # STANDAR: 203
    # BONE: 220
    # B30f: 178
    # B19f: 142
    kernels_of_interest = ['STANDARD', 'BONE', 'B30f', 'B19f']
    columns_names = ['Series UID', 'Collection', '3PA', 'Data Description URI', 'Subject ID', 'Study UID',
                     'Study Description', 'Study Date', 'Series Description', 'Manufacturer', 'Modality',
                     'SOP Class Name', 'SOP Class UID', 'Number of Images', 'File Size', 'File Location',
                     'Download Timestamp', 'Not indicated']

    LIDC_IDRI_patients_metadata = CSV_FILES + "/LIDC_IDRI_patients_metadata.csv"

    extract_and_save_CTs(LIDC_IDRI_metadata_path, LIDC_IDRI_folder_path, LIDC_IDRI_CTs_dataset, kernels_of_interest, columns_names, LIDC_IDRI_patients_metadata)









