import pandas as pd
import numpy as np
import os
import splitfolders

import random
import shutil

random.seed(42)

data_path = '/zhome/a7/0/155527/Desktop/s204161/fagprojekt/EasyOCR-master/trainer/all_data'

path_train_folder = '/zhome/a7/0/155527/Desktop/s204161/fagprojekt/EasyOCR-master/trainer/all_data/clean_cropped_en_train'
path_val_folder = '/zhome/a7/0/155527/Desktop/s204161/fagprojekt/EasyOCR-master/trainer/all_data/clean_cropped_en_val'

path_train_csv = '/zhome/a7/0/155527/Desktop/s204161/fagprojekt/EasyOCR-master/trainer/all_data/clean_cropped_en_train/labels.csv'
path_val_csv =  '/zhome/a7/0/155527/Desktop/s204161/fagprojekt/EasyOCR-master/trainer/all_data/clean_cropped_en_val/labels.csv'



train_csv = pd.read_csv(path_train_csv, sep='\t', header = 0)
val_csv = pd.read_csv(path_val_csv, sep='\t', header = 0)

data_csv = pd.concat([train_csv,val_csv],ignore_index=True)

train_test_ratio = 0.9

num_splits = 7

N = len(data_csv)

for split in range(num_splits):

    train_name = 'clean_cropped_en_train_split_' + str(split)
    val_name = 'clean_cropped_en_val_split_' + str(split)

    train_save_f = data_path + '/' + train_name
    val_save_f = data_path + '/' + val_name
    isExist_t = os.path.exists(train_save_f)
    isExist_v = os.path.exists(val_save_f)

    if not isExist_t:
        os.makedirs(train_save_f)
    if not isExist_v:
        os.makedirs(val_save_f)

    split_train_list = []
    split_val_list = []
    num_train_samples = int(N*train_test_ratio)
    train_set = random.sample(range(N), num_train_samples)
    for im in range(N):
        info = list(data_csv.iloc[im])
        image_ID = info[0]
        if im in train_set:
            split_train_list.append(info)
            if im < len(train_csv):
                shutil.copy(path_train_folder + '/' + image_ID, train_save_f + '/' + image_ID)
            else: shutil.copy(path_val_folder + '/' + image_ID, train_save_f+ '/' + image_ID)
        else:
            split_val_list.append(info)
            if im < len(train_csv):
                shutil.copy(path_train_folder + '/' + image_ID, val_save_f + '/' + image_ID)
            else: shutil.copy(path_val_folder + '/' + image_ID, val_save_f + '/' + image_ID)


    split_train_csv = pd.DataFrame(split_train_list, dtype='string')
    split_val_csv = pd.DataFrame(split_val_list, dtype='string')
    split_train_csv.columns = ['filename', 'words']
    split_val_csv.columns = ['filename', 'words']
    split_train_csv.to_csv(train_save_f + '/labels.csv', header=True, index=False, sep='\t', mode='w')
    split_val_csv.to_csv(val_save_f + '/labels.csv', header=True, index=False, sep='\t', mode='w')




print('done')