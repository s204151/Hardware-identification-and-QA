import cv2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.transforms import Affine2D
from PIL import Image, ExifTags
from ast import literal_eval
import numpy as np


def orient_w_metadata(img):
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = img._getexif()
        if exif != None:
            orient_num = exif[orientation]
            if orient_num == 3:
                img = img.rotate(180, expand=True)
            elif orient_num == 6:
                img = img.rotate(270, expand=True)
            elif orient_num == 8:
                img = img.rotate(90, expand=True)
        else: orient_num = 0
    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        orient_num = 0

    return img, orient_num


image_path = 'C:/Users/khali/Desktop/fagprojekt/en_train/'
path_to_csv = 'C:/Users/khali/Desktop/fagprojekt/annotations_up_to_ID900.csv'

def process_BB(path_to_csv):
    label_df = pd.read_csv(path_to_csv)
    image_urls = label_df['image_url']
    BB_df = list(label_df['bbox'])

    labeled_image_IDs = []
    BBs = []
    for i in range(len(image_urls)):
        try:
            BBs.append(literal_eval(BB_df[i]))
            labeled_image_IDs.append(image_urls[i].split('-', 1)[-1])
            # make python see the string of with BB info as a dictionary with literal_eval
        except ValueError:
            print(f'Warning! The {i}. entry in the dataFrame does not contain any bounding_boxes. It is excluded.')
    return labeled_image_IDs, BBs

labeled_image_IDs, BBs = process_BB(path_to_csv)

some_image_ID = labeled_image_IDs[12]
some_BB = BBs[12]

def get_corner_points(BB,xy_pairwise = False):
    BB = BB[0] # Bemærk at dette index gør at vi kun indexer den 1. bounding boks som er på et billede.
    rect_x = BB['x'] * BB['original_width'] / 100
    rect_y = BB['y'] * BB['original_height'] / 100
    rect_width = BB['width'] * BB['original_width'] / 100
    rect_height = BB['height'] * BB['original_height'] / 100
    deg = BB['rotation']

    angle = (deg * np.pi) / 180
    x1, y1 = int(rect_x), int(rect_y)
    x2, y2 = int(rect_x + rect_width * np.cos(angle)), int(rect_y + rect_width * np.sin(angle))
    x3, y3 = int(rect_x + rect_width * np.cos(angle) - rect_height * np.sin(angle)), \
             int(rect_y + rect_width * np.sin(angle) + rect_height * np.cos(angle))
    x4, y4 = int(rect_x - rect_height * np.sin(angle)), int(rect_y + rect_height * np.cos(angle))

    if xy_pairwise == False:
        return [x1,x2,x3,x4],[y1,y2,y3,y4]
    elif xy_pairwise == True:
        cnt= np.array([
            [[x1, y1]],
            [[x2, y2]],
            [[x3, y3]],
            [[x4, y4]]
        ])
        return cnt
#Function to check if bounding_boxes are properly transformed.
def plot_BB(image_ID, BB):
    image = Image.open(image_path + image_ID)
    oriented_image, orient_num = orient_w_metadata(image)
    xs,ys = get_corner_points(BB,xy_pairwise=False)

    BB = BB[0] # Bemærk at dette index gør at vi kun indexer den 1. bounding boks som der er.
    rect_x = BB['x'] * BB['original_width'] / 100
    rect_y = BB['y'] * BB['original_height'] / 100
    rect_width = BB['width'] * BB['original_width'] / 100
    rect_height = BB['height'] * BB['original_height'] / 100
    deg = BB['rotation']

    fig, ax = plt.subplots()
    ax.imshow(oriented_image)
    # Create a Rectangle patch
    rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=2, edgecolor='r', facecolor='none')
    plt.scatter(xs,ys, color = ['blue','orange','yellow','green'])

    if BB['rotation'] != 0:
        rot_rect = plt.Rectangle((rect_x, rect_y), rect_width, rect_height, linewidth=2,
                                 edgecolor='r', facecolor='none',
                                 transform=Affine2D().rotate_deg_around(*(rect_x,rect_y), BB['rotation'])+ax.transData)
        ax.add_patch(rot_rect)
    else:
        ax.add_patch(rect)

    plt.show()
    return

#plot_BB(some_image_ID,some_BB)


def crop_from_BB(labeled_image_IDs,image_path,BBs):
    # img = Image.open(image_path + labeled_image_IDs[12])
    # rotated_img = np.array(orient_w_metadata(img)[0])
    for i in range(len(labeled_image_IDs)):
        image_ID = labeled_image_IDs[i]
        BB = BBs[i]

        img = cv2.imread(image_path + image_ID)
        # four corner points for padded shoe
        cnt = get_corner_points(BB, xy_pairwise=True)

        #print("shape of cnt: {}".format(cnt.shape))
        rect = cv2.minAreaRect(cnt)
        print("rect: {}".format(rect))

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        width = int(rect[1][0])
        height = int(rect[1][1])

        src_pts = box.astype("float32")
        dst_pts = np.array([[0, height - 1],
                            [0, 0],
                            [width - 1, 0],
                            [width - 1, height - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(img, M, (width, height))
        if height > width:
            c_img = cv2.rotate(warped, cv2.cv2.ROTATE_90_CLOCKWISE)
        else: c_img = warped

        cv2.imwrite('cropped_en_train/c' + image_ID, c_img)
    #    cv2.imshow('okay', warped)
    return


crop_from_BB(labeled_image_IDs,image_path,BBs)



