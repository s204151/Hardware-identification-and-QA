# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:18:33 2022

@author: khalil
"""
import shutil
import numpy as np
import os
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split
from distutils.dir_util import copy_tree
import sys
OCR_train_path = 'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/en_train/'
OCR_test_path = 'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/en_val/'

path_1_1 = 'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/meter_number/meter_number/'
path_1_2 = 'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/meter_number/meter_number/photos'
path_2_1 = 'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/meter_number_images_2'
path_2_2 = 'C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/meter_number_images_2/photos'

os.chdir(path_1_1)

csv_file = pd.read_csv('labels.csv')

##### ome minor Data-cleaning to remove skap.gif and other bad/mislabeled pictures
#bad_gif_path ='photos/584c6349-1176-4efe-b96d-2f5c40faa538/Skap.GIF'
csv_file = csv_file.drop(1935,axis = 0) #label_missing
csv_file = csv_file.drop(1711,axis = 0) #Bad gif
csv_file = csv_file.drop(1529,axis = 0) #label_missing
csv_file = csv_file.drop(1040,axis = 0) #label_missing
csv_file = csv_file.drop(563,axis = 0) #Bad picture
csv_file = csv_file.drop(440,axis = 0) #Bad picture
csv_file = csv_file.drop(442,axis = 0) #Bad picture

#Der ændres i labels på række-ID 1822-1823 fra: '5706567272466825 Måleid: 707057500075152926' til 707057500075152926'
csv_file.at[1822,'meter_number'] = csv_file.at[1822,'meter_number'].split(' ')[-1]
csv_file.at[1823,'meter_number'] = csv_file.at[1823,'meter_number'].split(' ')[-1]
csv_file.at[1153,'meter_number'] = csv_file.at[1153,'meter_number'].split(' ')[0]
#Række-ID 435,440,442 står der 7359992897705033 / 7359992897705200 på alle 3, og ved manuel aflæsning
#af billederne, inspiceres 7359992897705200 på 435, at der ikke er taget billede af stregkoden i 440 og 442.
csv_file.at[435,'meter_number'] = csv_file.at[435,'meter_number'].split('/')[1]
csv_file.at[435,'meter_number'] = csv_file.at[1070,'meter_number'].split('/')[0]

paths = np.asarray(csv_file['image_path'])
meter_numbers = np.asarray(csv_file['meter_number'])
##Even after cleaning up a bit, we can see that there is still some noisy/bad observations where it is unclear what the
#real label is.
#for i in meter_numbers:
#    if i.isdigit() != True:
#        print(i)
#sys.exit("done")

#We note that of 2059 the meter_numbers, 1317 only seem to appear once/are unique, which makes it difficult to stratitfy
# our train_test_split along meter_numbers, since we would need at least 2 observations from each class to stratify...
counts = np.unique(meter_numbers, return_counts=True)[1]
num_classes = len(counts)
total_observations = sum(counts)
max_label_length = np.unique(meter_numbers)[-1]

path_train, path_test, meter_train, meter_test = train_test_split(paths, meter_numbers, test_size = 0.2, random_state=42)

#-----------------------------------------------------
#now that we have chosen what our test and train data is, we get the data into the format which EasyOCR uses.

Image.MAX_IMAGE_PIXELS = None
os.chdir(path_1_2)



def todataX(path,csv_file):
    i_num = 0
    train_list = []
    test_list = []
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            full_path_to_picture = os.path.join(root, name)
            path_to_picture = os.path.join(root, name).split('meter_number/')[2]
            if '\\' in path_to_picture:
                path_to_picture = path_to_picture.replace('\\','/')
                full_path_to_picture = full_path_to_picture.replace('\\', '/')
            #path_to_dir = os.path.join(root, name).rsplit('/',1)[0]
            #dir_name = os.path.join(root, name).rsplit('/',2)[1]
            if path_to_picture in path_train:
                new_id = 'ID' + str(csv_file[csv_file['image_path'] == path_to_picture]['Unnamed: 0'].item()) + '.' + path_to_picture.split('.')[-1]
                label = csv_file[csv_file['image_path'] == path_to_picture]['meter_number']
                label = label.item()
                label = str(label)
                train_list.append([new_id, label])
                #train_list.append(['file:///C:/Users/khali/Desktop/fagprojekt/en_train/' + new_id, label])
                ###to copy image to train/test folder - uncomment this if copy over.
                #shutil.copy(full_path_to_picture, OCR_train_path + new_id)
            elif path_to_picture in path_test:
                new_id = 'ID' + str(csv_file[csv_file['image_path'] == path_to_picture]['Unnamed: 0'].item()) + '.' + path_to_picture.split('.')[-1]
                label = csv_file[csv_file['image_path'] == path_to_picture]['meter_number']
                label = label.item()
                label = str(label)
                test_list.append([new_id, label])
                #test_list.append(['file:///C:/Users/khali/Desktop/fagprojekt/en_val/' + new_id, label])
                ###to copy image to train/test folder - uncomment this if copy over.
                #shutil.copy(full_path_to_picture, OCR_test_path + new_id)
            i_num += 1
            print(i_num, new_id)


    return train_list,test_list

train_list_1, test_list_1 = todataX(path_1_2,csv_file)
#
# train_df = pd.DataFrame(train_list,dtype = 'string')
# test_df = pd.DataFrame(test_list, dtype = 'string')
# #train_df.columns = ['filename', 'words']
# #test_df.columns = ['filename', 'words']
# train_df.columns = ['image_url', 'words']
# test_df.columns = ['image_url', 'words']
#
# train_df.to_csv(r'C:/Users/khali/Desktop/fagprojekt/en_train/labels.csv', header=True, index=False, sep=',')
# test_df.to_csv(r'C:/Users/khali/Desktop/fagprojekt/en_val/labels.csv', header=True, index=False, sep=',')

#train_df.to_csv(r'C:/Users/khali/Desktop/fagprojekt/en_train/labels.csv', header=True, index=False, sep='\t')
#test_df.to_csv(r'C:/Users/khali/Desktop/fagprojekt/en_val/labels.csv', header=True, index=False, sep='\t')

# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:18:33 2022

Almost all of the data in 2. batch is unlabeled :(

"""
os.chdir(path_2_1)
#exit
csv_file = pd.read_csv('labels.csv')

paths = np.asarray(csv_file['image_path'])
meter_numbers = np.asarray(csv_file['meter_number'])

path_train, path_test, meter_train, meter_test = train_test_split(paths, meter_numbers, test_size = 0.2, random_state=42)

#-----------------------------------------------------
#now that we have chosen what our test and train data is, we get the data into the format which EasyOCR uses.
Image.MAX_IMAGE_PIXELS = None
os.chdir(path_2_2)

def todataX(path,csv_file):
    i_num = 0
    train_list = []
    test_list = []
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
                ###to copy image to train/test folder - uncomment this if copy over.
                #shutil.copy(full_path_to_picture, OCR_train_path + new_id)
            elif path_to_picture in path_test:
                new_id = 'ID' + str(2062 + int(csv_file[csv_file['image_path'] == path_to_picture]['Unnamed: 0'].item())) + '.' + path_to_picture.split('.')[-1]
                label = 'empty'
                test_list.append([new_id, label])
                #test_list.append(['file:///C:/Users/khali/Desktop/fagprojekt/en_val/' + new_id, label])
                ###to copy image to train/test folder - uncomment this if copy over.
                #shutil.copy(full_path_to_picture, OCR_test_path + new_id)
            i_num += 1
            print(i_num, new_id)
    return train_list,test_list

train_list_2, test_list_2 = todataX(path_2_2,csv_file)

train_df = pd.DataFrame(train_list_1+train_list_2,dtype = 'string')
test_df = pd.DataFrame(test_list_1+test_list_2, dtype = 'string')
#train_df.columns = ['filename', 'words']
#test_df.columns = ['filename', 'words']
train_df.columns = ['image_url', 'words']
test_df.columns = ['image_url', 'words']

train_df.to_csv('C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/en_train/labels.csv', header=True, index=False, sep=',')
test_df.to_csv('C:/Users/khali/OneDrive - Danmarks Tekniske Universitet/Fagprojekt/en_val/labels.csv', header=True, index=False, sep=',')

#train_df.to_csv(r'C:/Users/khali/Desktop/fagprojekt/en_train/labels.csv', header=True, index=False, sep='\t')
#test_df.to_csv(r'C:/Users/khali/Desktop/fagprojekt/en_val/labels.csv', header=True, index=False, sep='\t')


#print(len(np.unique(shapes)))
#print(np.unique(shapes))

#np.save('/zhome/a7/0/155527/Desktop/s204161/fagprojekt/path_to_data.npy', path_data)
#np.save('/zhome/a7/0/155527/Desktop/s204161/fagprojekt/data_shapes.npy', shapes)
#np.save('/zhome/a7/0/155527/Desktop/s204161/fagprojekt/data_names.npy', names)

