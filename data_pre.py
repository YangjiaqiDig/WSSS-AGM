import pandas as pd 
import shutil
import os
from PIL import Image
import numpy as np

IMAGES_DIR = "train/"
IMAGES_LABELED_DIR = "train_label"
LABELS_DIR = "train_label.csv"
pd.set_option('display.max_rows', None)

def normalize_label(value):
    if value == 'y':
        return 1
    return 0

def get_dic_img_label(img, df, filename, dic):
    dic["image"] = img
    row = df.loc[df['Image'] == filename]
    dic["edema"] = normalize_label(row["Edema"].values[0])
    dic["dril"] = normalize_label(row["DRIL"].values[0])
    dic["ez"] = normalize_label(row["EZ loss"].values[0])
    dic["rpe"] = normalize_label(row["RPE changes"].values[0])

    return dic

df = pd.read_csv(LABELS_DIR)
df = df[df['DRIL'].notna()]
print("total number of labeled", len(df))
labeled_file_name = df["Image"]
for idx, filename in enumerate(labeled_file_name):
    dic = {}
    img_path = os.path.join(IMAGES_DIR, '{}.jpeg'.format(filename))
    image = Image.open(img_path)
    image_arr = np.asarray(image)
    print(filename, image_arr.shape)
    dic = get_dic_img_label(image_arr, df, filename, dic)
    np.save("{0}/{1}.npy".format(IMAGES_LABELED_DIR, idx), dic)
    shutil.copy(img_path, "{0}/{1}_{2}.jpeg".format(IMAGES_LABELED_DIR, idx, filename))
    