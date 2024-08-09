import os
import shutil

DATASET_PATH = '/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/StarGAN_v2/StarGAN_dataset'

dataset_list =  os.listdir(DATASET_PATH)

elems_per_domains_counter = {}

for folder in dataset_list:
    FOLDER_PATH = os.path.join(DATASET_PATH, folder)
    for domain_folder in os.listdir(FOLDER_PATH):
        DOMAIN_FOLDER_PATH = os.path.join(FOLDER_PATH, domain_folder)
        dcm_files = os.listdir(DOMAIN_FOLDER_PATH)
        if domain_folder not in elems_per_domains_counter:
            elems_per_domains_counter[domain_folder] = len(dcm_files)
        else:
            elems_per_domains_counter[domain_folder] += len(dcm_files)

# take only the two biggest domains
domains_considered = sorted(elems_per_domains_counter, key=elems_per_domains_counter.get, reverse=True)[:2]

dat_train = '/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/StarGAN_v2/StarGAN_dataset_2/train'
dat_test = '/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/StarGAN_v2/StarGAN_dataset_2/test'
dat_valid = '/mimer/NOBACKUP/groups/naiss2023-6-336/assolito/StarGAN_v2/StarGAN_dataset_2/valid'

for folder in dataset_list:
    FOLDER_PATH = os.path.join(DATASET_PATH, folder)
    for domain_folder in os.listdir(FOLDER_PATH):
        DOMAIN_FOLDER_PATH = os.path.join(FOLDER_PATH, domain_folder)
        if domain_folder in domains_considered:
            if folder == 'train':
                DESTINATION_FOLDER = dat_train + '/' + domain_folder
            elif folder == 'test':
                DESTINATION_FOLDER = dat_test + '/' + domain_folder
            else:
                DESTINATION_FOLDER = dat_valid + '/' + domain_folder
            shutil.copytree(DOMAIN_FOLDER_PATH, DESTINATION_FOLDER)


