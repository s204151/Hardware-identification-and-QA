
from ast import literal_eval
import numpy as np
import pandas as pd

data_dir = "/zhome/a7/0/155527/Desktop/s204161/fagprojekt/EasyOCR-master/trainer/all_data/clean_en_train/"

def load_data():
    train_csv = pd.read_csv(data_dir + "labels.csv", sep ="\t", header = 0)

    for i in range(len(train_csv)):
        train_csv['BBs'][i] = literal_eval \
            (train_csv['BBs'][i].replace("\n" ,"").replace(" " ,"").replace("array(" ,"").replace(")" ,""))
        if len(train_csv['BBs'][i]) > 1:
            all_BBs = []
            for j in range(len(train_csv['BBs'][i])):
                BB = [val for sublist in train_csv['BBs'][i][j] for val in sublist]
                all_BBs.append(BB)
            train_csv['BBs'][i] = np.stack(all_BBs, axis = 0)
        elif len(train_csv['BBs'][i]) > 0:
            train_csv['BBs'][i] = np.array([[val for sublist in train_csv['BBs'][i][0] for val in sublist]])

    meterbox = train_csv['BBs']
    image = train_csv['filename']
    imgtxt = train_csv['words']

    return train_csv


train_csv = load_data()
import os
import cv2

for i in range(len(ims)):
    img_path = os.path.join(data_dir, ims[i])
    image = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if image.shape[0] < 768 or image.shape[1] < 768:
        print(image.shape, "image:",ims[i],"index:",i)
        print('oke')


print('done')