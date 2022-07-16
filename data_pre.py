import glob
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

def get_dic_img_detail_lable(img, df, filename, dic):
    # img,SRF,IRF,EZ attenuated,EZ disrupted,HRD,RPE,Retinal Traction,Definite DRIL,Questionable DRIL
    dic["image"] = img
    row = df.loc[df['img'] == filename]
    dic["srf"] = row["SRF"].values[0]
    dic["irf"] = row["IRF"].values[0]
    dic["ezAtt"] = row["EZ attenuated"].values[0]
    dic["ezDis"] = row["EZ disrupted"].values[0]
    dic["hrd"] = row["HRD"].values[0]
    dic["rpe"] = row["RPE"].values[0]
    dic["rt"] = row["Retinal Traction"].values[0]
    dic["dril"] = row["Definite DRIL"].values[0]
    dic["qDril"] = row["Questionable DRIL"].values[0]
    return dic

def prepare_dme_dataset():
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

def prepare_dr_dataset():
    df = pd.read_csv('dr_labels.csv')
    print("total number of labeled", len(df))
    labeled_file_name = df["img"]
    for idx, filename in enumerate(labeled_file_name):
        dic = {}
        img_path = os.path.join('dr_dataset', '{}'.format(filename))
        image = Image.open(img_path)
        image_arr = np.asarray(image)
        print(filename, image_arr.shape)
        dic = get_dic_img_detail_lable(image_arr, df, filename, dic)
        np.save("{0}/{1}.npy".format('train_dr', idx), dic)
        shutil.copy(img_path, "{0}/{1}_{2}".format('train_dr', idx, filename))

def prepare_resc_dataset():
    root_dir = "RESC/train/label_images"
    # rename all the sub-folders to unique shorter name, skip the code here
    
    # flatten the images
    subfolders = glob.glob("{}/*".format(root_dir))
    for sub_folder in subfolders:
        print(sub_folder)
        patient_id = sub_folder.split('/')[-1]
        images = os.listdir(sub_folder)
        for image in images:
            old_path = "{}/{}".format(sub_folder, image)
            new_path = "{}/{}_{}".format(root_dir, patient_id, image)
            shutil.move(old_path, new_path)

if __name__ == "__main__":
    # prepare_dr_dataset()
    prepare_resc_dataset()