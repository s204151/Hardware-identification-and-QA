# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:18:33 2022

Almost all of the data is unlabeled :(

"""
import shutil
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
import sys
path = 'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/meter_number_images_2'
os.chdir(path)

csv_file = pd.read_csv('labels.csv')



#csv_file.at[21,'meter_number'] = csv_file.at[21,'meter_number'].split(' ')[-1]
#csv_file.at[23,'meter_number'] = csv_file.at[23,'meter_number'].split(' ')[-1]
#csv_file.at[24,'meter_number'] = csv_file.at[24,'meter_number'].split(' ')[-1]
#csv_file.at[25,'meter_number'] = csv_file.at[25,'meter_number'].split(' ')[-1]
#csv_file.at[26,'meter_number'] = csv_file.at[26,'meter_number'].split(' ')[-1]

paths = np.asarray(csv_file['image_path'])
meter_numbers = np.asarray(csv_file['meter_number'])



path_train, path_test, meter_train, meter_test = train_test_split(paths, meter_numbers, test_size = 0.2, random_state=42)

#-----------------------------------------------------
#now that we have chosen what our test and train data is, we get the data into the format which EasyOCR uses.

path = 'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/meter_number_images_2/photos'
Image.MAX_IMAGE_PIXELS = None
os.chdir(path)

train_list = []
test_list = []
OCR_train_path = 'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/en_train_2/'
OCR_test_path = 'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/en_val_2/'

def todataX(path,csv_file):
    i_num = 0
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            full_path_to_picture = os.path.join(root, name)
            path_to_picture = os.path.join(root, name).split('meter_number_images_2/')[1]
            if '\\' in path_to_picture:
                path_to_picture = path_to_picture.replace('\\','/')
            #path_to_dir = os.path.join(root, name).rsplit('/',1)[0]
            #dir_name = os.path.join(root, name).rsplit('/',2)[1]
            if path_to_picture in path_train:
                new_id = 'ID' +  str(2062 + int(csv_file[csv_file['image_path'] == path_to_picture]['Unnamed: 0'].item())) + '.' + path_to_picture.split('.')[-1]

                label = 'empty'
                train_list.append([new_id,label])
                #train_list.append(['file:///C:/Users/khali/Desktop/fagprojekt/en_train/' + new_id, label])
                #to copy image to train/test folder
                shutil.copy(full_path_to_picture, OCR_train_path + new_id)
            elif path_to_picture in path_test:
                new_id = 'ID' + str(2062 + int(csv_file[csv_file['image_path'] == path_to_picture]['Unnamed: 0'].item())) + '.' + path_to_picture.split('.')[-1]
                label = 'empty'
                test_list.append([new_id, label])
                #test_list.append(['file:///C:/Users/khali/Desktop/fagprojekt/en_val/' + new_id, label])
                shutil.copy(full_path_to_picture, OCR_test_path + new_id)

            i_num += 1
            print(i_num)


    return train_list,test_list

train_list, test_list = todataX(path,csv_file)

train_df = pd.DataFrame(train_list,dtype = 'string')
test_df = pd.DataFrame(test_list, dtype = 'string')
#train_df.columns = ['filename', 'words']
#test_df.columns = ['filename', 'words']
train_df.columns = ['image_url', 'words']
test_df.columns = ['image_url', 'words']

train_df.to_csv(r'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/en_train_2/labels.csv', header=True, index=False, sep=',')
test_df.to_csv(r'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/en_train_2/labels.csv', header=True, index=False, sep=',')

#train_df.to_csv(r'C:/Users/khali/Desktop/fagprojekt/en_train/labels.csv', header=True, index=False, sep='\t')
#test_df.to_csv(r'C:/Users/khali/Desktop/fagprojekt/en_val/labels.csv', header=True, index=False, sep='\t')




#print(len(np.unique(shapes)))
#print(np.unique(shapes))


#np.save('/zhome/a7/0/155527/Desktop/s204161/fagprojekt/path_to_data.npy', path_data)
#np.save('/zhome/a7/0/155527/Desktop/s204161/fagprojekt/data_shapes.npy', shapes)
#np.save('/zhome/a7/0/155527/Desktop/s204161/fagprojekt/data_names.npy', names)

