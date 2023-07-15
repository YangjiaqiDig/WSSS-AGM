import os, glob
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
# from medpy.filter.smoothing import anisotropic_diffusion
import cv2 as cv
import torchvision.transforms as T
import pandas as pd
import shutil

from utils.metrics import scores
from utils.utils import convert_our_dataset_labels

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


def edge_detection():
    img = cv.imread('cam_test/9_DR10.jpeg',0)
    edges = cv.Canny(img,100,200)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.savefig('tst3.jpeg')
    return

def overlap():
    """
    visdom
    """
    image = Image.open('ganomaly/output/ganomaly/anomaly_dme_rescale/test/images/001_fake.png')
    image_arr_f = np.array(image)
    edges = cv.Canny(image_arr_f,300,50)
    # image_arr_f[image_arr_f < 85] = 0
    im = Image.fromarray(edges)
    im.save("your_file_f.jpeg")
    
    image_r = Image.open('ganomaly/output/ganomaly/anomaly_dme_rescale/test/images/001_real.png')
    image_arr_r = np.array(image_r)
    edges = cv.Canny(image_arr_r,300,300)
    # image_arr_r[image_arr_r < 85] = 0
    im = Image.fromarray(edges)
    im.save("your_file_r.jpeg")

    
    image_diff = np.abs(image_arr_f - image_arr_r)    
    # image_diff[image_diff<127] = 0
    
    """
    save images
    """
    # image_diff[image_diff<100] = 0
    # image_diff[image_diff>=250] = 255
    im = Image.fromarray(image_diff)
    im.save("your_file.jpeg")
    # image_residual = torch.abs(image_rec - image)

def conencted_component_removal():
    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever
    # min_size = 1500
    imgs_dirs = glob.glob("examples/ganomaly/test/1.abnormal/*")
    for img_dir in imgs_dirs:
        img_name = img_dir.split('/')[-1]
        img = cv.imread(img_dir,0)
        # img[img == 255] = 0
        kernel = np.ones((5,5),np.uint8)
        erosion = cv.erode(img,kernel,iterations = 1)
        # img[img == 255] = 0
        # img[(erosion == 255 )  ] = 0 #& (img >= 250)
        dilatation = cv.dilate(erosion,kernel,iterations = 5)
        img[(dilatation == 255 ) ] = 0 
        cv.imwrite('background_rm/{}'.format(img_name), img)
    return

def background_mask(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.zeros(image.shape, np.uint8)

    img_median = cv.medianBlur(gray, 5)
    ret, gray = cv.threshold(img_median, 60,255, cv.THRESH_BINARY)

    closing_kernel = np.ones((15,15),np.uint8)
    closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, closing_kernel, iterations=1)

    contours, _ = cv.findContours(closing.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_area = sorted(contours, key=cv.contourArea)[-1:]
    cv.drawContours(mask, largest_area, 0, (255,255,255), cv.FILLED)

    return mask[:,:,0]

def generate_background_mask_for_GAN(image_tensor):
    # define a transform to convert a tensor to PIL image
    transform = T.ToPILImage()

    # convert the tensor to PIL image using above transform
    image = transform(image_tensor)
    
    image = image.filter(ImageFilter.MinFilter(3))
    image = ImageEnhance.Contrast(image).enhance(2)

    image = np.array(image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.zeros(image.shape, np.uint8)
    
    gray = cv.medianBlur(gray, 5)
    ret, gray = cv.threshold(gray, 30,255, cv.THRESH_BINARY)

    closing_kernel = np.ones((15,15),np.uint8)
    closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, closing_kernel, iterations=1)

    contours, _ = cv.findContours(closing.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_area = sorted(contours, key=cv.contourArea)[-1:]
    cv.drawContours(mask, largest_area, 0, (255,255,255), cv.FILLED)

    return mask

def generate_background_mask(image_path):
    img = np.array(cv.imread(image_path))
    image = img.copy()
    image[image > 250] = 0
    image = Image.fromarray(image)
    
    image = image.filter(ImageFilter.MinFilter(3))
    image = ImageEnhance.Contrast(image).enhance(2)

    image = np.array(image)
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    mask = np.zeros(image.shape, np.uint8)
    
    gray = cv.medianBlur(gray, 5)
    ret, gray = cv.threshold(gray, 30,255, cv.THRESH_BINARY)

    closing_kernel = np.ones((15,15),np.uint8)
    closing = cv.morphologyEx(gray, cv.MORPH_CLOSE, closing_kernel, iterations=1)

    contours, _ = cv.findContours(closing.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largest_area = sorted(contours, key=cv.contourArea)[-1:]
    cv.drawContours(mask, largest_area, 0, (255,255,255), cv.FILLED)
    dst = cv.bitwise_and(img, mask)
    
    return mask#dst

def remove_background():
    imgs_dirs = glob.glob("datasets/our_dataset/dataset_DME/4/images/*")#
    count = 0
    for img_dir in imgs_dirs:
        img_name = img_dir.split('/')[-1]
        imgOrg = cv.imread(img_dir,0)
        img = cv.medianBlur(imgOrg, 3)
        ret, binary = cv.threshold(img, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
        th2 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_MEAN_C,\
            cv.THRESH_BINARY,11,2)
        th3 = cv.adaptiveThreshold(img,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,\
                    cv.THRESH_BINARY,11,2)
        # kernel2 = cv.getStructuringElement(cv.MORPH_RECT, (1,1)) #(3,3)
        # erosion = cv.erode(binary,kernel2,iterations = 1)
        # bin_clo = cv.dilate(th2,kernel2,iterations = 1)
        nb_components, labels, stats, centroids = cv.connectedComponentsWithStats(th2, connectivity=8)
        output = imgOrg.copy()#np.zeros((img.shape[0], img.shape[1], 3))
        color_output = np.zeros((img.shape[0], img.shape[1], 3))

        max_labels = []
        sizes = stats[:, -1]
        # print(nb_components, sizes)
        max_size = 500
        half_num_pixels = output.size / 2

        for i in range(1, nb_components):
            mask = labels == i
            color_output[:,:,0][mask] = np.random.randint(0,255)
            color_output[:,:,1][mask] = np.random.randint(0,255)
            color_output[:,:,2][mask] = np.random.randint(0,255)
            if sizes[i] > max_size and sizes[i] < half_num_pixels and len([i for i in output[mask] if i > 240]) / len(output[mask]) > 0.95:
                max_labels.append(i)
                # max_size = sizes[i]
            # break
        # print(sizes, max_labels)
        if len(max_labels):
            count += 1
            for max_label in max_labels:
                output[labels == max_label] = 0
            # print(max_label)
            # print(output.shape)
            cv.imwrite('datasets/our_dataset/dataset_DME/4/images_backrm/{}'.format(img_name), output)
        # output[output >= 240] = 0
        # cv.imwrite('test.png', output)
    print(count)
    return

def random_seperate_test():
    labels_table = pd.read_csv('datasets/our_dataset/labels.csv')
    labels_table['EZ'] = (labels_table['EZ attenuated'] + labels_table['EZ disrupted'])
    labels_table['patient'] = labels_table['img'].apply(lambda row: '-'.join(row.split('-')[:2])) 
    
    combined_df = labels_table.groupby(['patient']).agg({'SRF':'sum','IRF':'sum', 'EZ':'sum', 'HRD':'sum','RPE':'sum','Retinal Traction':'sum','Definite DRIL':'sum'}).reset_index()
    selected = labels_table[labels_table['img'].str.contains('2388519|15307|3882196|3565572|4240465|224974|3491563|1072015|DR91|DR69|DR10.jpeg')]
    selected = selected.append(selected.sum(numeric_only=True), ignore_index=True)
    print(selected)
    combined_df = combined_df.sort_values(by=[ 'Definite DRIL'], ascending=False)
    print(combined_df.iloc[:30])
    #3882196|3565572|4240465|224974|2205167|3491563|DR91|DR10

def generate_mask_datasets():
    list_of_data = glob.glob("datasets/our_dataset/original/train/*")
    for item in list_of_data:
        image_name = item.split('/')[-1].split('.')[0]
        res = generate_background_mask(item)
        cv.imwrite('datasets/our_dataset/mask/train/{}.png'.format(image_name), res)

    list_of_test = glob.glob("datasets/our_dataset/original/test/*")
    for item in list_of_test:
        image_name = item.split('/')[-1].split('.')[0]
        res = generate_background_mask(item)
        cv.imwrite('datasets/our_dataset/mask/test/{}.png'.format(image_name), res)

def generate_mask_general(dirs, save_dir):
    for dir_i in dirs:
        list_of_data = glob.glob(dir_i)
        # print(list_of_data)
        for item in list_of_data:
            image_name = item.split('/')[-1]
            res = generate_background_mask(item)
            cv.imwrite('{}/{}'.format(save_dir, image_name), res)

    
cat_mapping = {1: 'IRF', 2: 'SRF', 3: 'HRD', 4: 'EZ disrupted', 5: 'RPE'}
def generate_from_coco(image_id, coco, gt_labels, cat_ids, expert):
    
    img = coco.imgs[image_id]
    file_name = img['file_name']
    
    found_gt = gt_labels[gt_labels['img']==file_name].to_dict('records')[0]    
    anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    
    anns_df = pd.DataFrame(anns)
    anns_df = anns_df[['category_id', 'area']]
    anns_df = anns_df.groupby('category_id').agg('sum').reset_index().to_dict('records')
    anns_img = np.zeros((img['height'],img['width']))
    for item in anns_df:
        col_name = '{}_{}'.format(expert, cat_mapping[item['category_id']])
        area_name = '{}_area_{}'.format(expert, cat_mapping[item['category_id']])
        found_gt[col_name] = 1
        found_gt[area_name] = item['area']
           
    # mask = coco.annToMask(anns[0])
    for ann in anns:
        anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])
        # mask += coco.annToMask(ann)
    save_mask = Image.fromarray((anns_img / 5 * 255).astype(np.uint8))
    i_name = file_name.split('.')[0]
    save_mask.save('our_data_analysis/annotation_v2/{}_{}.png'.format(i_name, expert))
    return found_gt

def genearte_annotation_for_our_dataset(gt_labels):
    from pycocotools.coco import COCO
    import numpy as np
    # "categories":[{"id":1,"name":"IRF"},{"id":2,"name":"SRF"},{"id":3,"name":"HRD"},{"id":4,"name":"EZ disruption"},{"id":5,"name":"RPE "}]}
    coco_1 = COCO('our_data_analysis/mina_v2.json')
    coco_2 = COCO('our_data_analysis/meera_v2.json')
    cat_ids = coco_1.getCatIds() # 1,2,3,4,5
    res_list = []
    for image_id in range(1, 101):
        # new_labeled = {'SRF_new': 0, 'IRF_new': 0, 'EZ disrupted_new': 0, 'HRD_new': 0}
        updated_res = generate_from_coco(image_id, coco_1, gt_labels, cat_ids, 'mina')
        res_list.append(updated_res)
    new_df = pd.DataFrame(res_list)
    
    res_list = []
    for image_id in range(1, 101):
        # new_labeled = {'SRF_new': 0, 'IRF_new': 0, 'EZ disrupted_new': 0, 'HRD_new': 0}
        updated_res = generate_from_coco(image_id, coco_2, new_df, cat_ids, 'meera')
        res_list.append(updated_res)
    new_df = pd.DataFrame(res_list)
    new_df = new_df.fillna(0)
    new_df.to_csv('our_data_analysis/annot_analysis_v2.csv')

def analyze_annotations():
    from sklearn import metrics
    import matplotlib.pyplot as plt
    df = pd.read_csv('our_data_analysis/annot_analysis_v2.csv', index_col=0)
    confusion_matrix = metrics.confusion_matrix(df['IRF'], df['mina_IRF'])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])
    # print(cm_display)
    import pdb; pdb.set_trace()
    print(df[(df['mina_SRF']==df['meera_SRF']) & (df['mina_IRF']==df['meera_IRF']) & (df['mina_EZ disrupted']==df['meera_EZ disrupted']) & (df['mina_HRD']==df['meera_HRD']) & (df['mina_RPE']==df['meera_RPE'])].shape)
    # return cm_display

def save_human_label():
    labels_table = pd.read_csv('our_data_analysis/annot_analysis_v2.csv')
    mina_label = labels_table.loc[:, (labels_table.columns.str.contains(r'mina|img')) & ~(labels_table.columns.str.contains('_area_'))]
    mina_label.columns = mina_label.columns.str.replace('mina_', '')
    meera_label = labels_table.loc[:, (labels_table.columns.str.contains(r'meera|img')) & ~(labels_table.columns.str.contains('_area_'))]
    meera_label.columns = meera_label.columns.str.replace('meera_', '')
    
    mina_label.to_csv('datasets/our_dataset/mina_labels.csv')
    meera_label.to_csv('datasets/our_dataset/meera_labels.csv')
    return


def get_intersection_area(ex1, ex2, id_num):
    ex1_copy, ex2_copy = ex1.copy(), ex2.copy()
    ex1_copy[ex1_copy != id_num] = 0
    ex1_copy[ex1_copy == id_num] = 1
    ex2_copy[ex2_copy != id_num] = 0
    ex2_copy[ex2_copy == id_num] = 1
    score = scores([ex1_copy], [ex2_copy], n_class=2)
    # import pdb; pdb.set_trace()
    return score['Class IoU'][1]

def union_exps_annotations(ex1, ex2, updated_img, id_num):
    updated_img[(ex1==id_num) | (ex2==id_num)] = id_num
    return updated_img

def intersect_exps_annotations(ex1, ex2, updated_img):
    updated_img[(ex1==51) & (ex2==51)] = 51
    updated_img[(ex1==102) & (ex2==102)] = 102
    updated_img[(ex1==153) & (ex2==153)] = 153
    updated_img[(ex1==204) & (ex2==204)] = 204
    return updated_img


def intersect_human_annots():
    list_of_annotations = glob.glob('datasets/our_dataset/annotation_v2/*')
    mina_list = sorted([x for x in list_of_annotations if 'mina' in x])
    meera_list = sorted([x for x in list_of_annotations if 'meera' in x])
    updated_img_labels = []
    # "categories":[{"id":51,"name":"IRF"},{"id":102,"name":"SRF"},{"id":153,"name":"HRD"},{"id":204,"name":"EZ disruption"},{"id":255,"name":"RPE"}]}
    # ['SRF', 'IRF', 'EZ disrupted', 'HRD', 'BackGround']
    mina_collect = []
    meera_collect = []
    for mina, meera in zip(mina_list, meera_list):
        curr_img_label = {'IRF': 0, 'SRF': 0, 'EZ disrupted': 0, 'HRD': 0}
        assert mina.split('_')[0] == meera.split('_')[0]
        mina_img = np.array(Image.open(mina))
        meera_img = np.array(Image.open(meera))
        img_name = mina.split('/')[-1].split('_')[0] + '.jpeg'
        curr_img_label['img'] = img_name
        
        unioned_image = np.zeros_like(mina_img)
        unioned_image = intersect_exps_annotations(mina_img, meera_img, unioned_image)
        if 51 in unioned_image:
            curr_img_label['IRF'] = 1
        if 102 in unioned_image:
            curr_img_label['SRF'] = 1
        if 153 in unioned_image:
            curr_img_label['HRD'] = 1
        if 204 in unioned_image:
            curr_img_label['EZ disrupted'] = 1
        # if 51 in mina_img and 51 in meera_img and 51 not in unioned_image:
        #     import pdb; pdb.set_trace()
        updated_img_labels.append(curr_img_label)
        mina_img_norm = convert_our_dataset_labels(mina_img.copy())
        meera_img_norm = convert_our_dataset_labels(meera_img.copy())
        mina_collect.append(mina_img_norm)
        meera_collect.append(meera_img_norm)
        
        save_mask = Image.fromarray((unioned_image).astype(np.uint8))
        save_mask.save('datasets/our_dataset/annot_combine/' + img_name.replace('.jpeg', '.png'))
    import pdb; pdb.set_trace
    pd.DataFrame(updated_img_labels).to_csv('datasets/our_dataset/combine_labels.csv')
    print(scores(mina_collect, meera_collect, n_class=5))
    # import pdb; pdb.set_trace()
    return


def combine_human_annots():
    list_of_annotations = glob.glob('datasets/our_dataset/annotation_v2/*')
    mina_labels = pd.read_csv('datasets/our_dataset/mina_labels.csv', index_col=0)
    meera_labels = pd.read_csv('datasets/our_dataset/meera_labels.csv', index_col=0)
    mina_list = sorted([x for x in list_of_annotations if 'mina' in x])
    meera_list = sorted([x for x in list_of_annotations if 'meera' in x])
    inter_img_label = {'IRF': 0, 'SRF': 0, 'EZ': 0, 'HRD': 0}
    inter_areas = {'IRF': 0, 'SRF': 0, 'EZ': 0, 'HRD': 0}
    updated_img_labels = []
    # "categories":[{"id":51,"name":"IRF"},{"id":102,"name":"SRF"},{"id":153,"name":"HRD"},{"id":204,"name":"EZ disruption"},{"id":255,"name":"RPE"}]}
    # ['SRF', 'IRF', 'EZ disrupted', 'HRD', 'BackGround']
    for mina, meera in zip(mina_list, meera_list):
        curr_img_label = {'IRF': 0, 'SRF': 0, 'EZ disrupted': 0, 'HRD': 0}
        assert mina.split('_')[0] == meera.split('_')[0]
        mina_img = np.array(Image.open(mina))
        meera_img = np.array(Image.open(meera))
        assert mina_img.shape == meera_img.shape
        img_name = mina.split('/')[-1].split('_')[0] + '.jpeg'
        curr_img_label['img'] = img_name
        mina_img_l = mina_labels[mina_labels['img']==img_name].iloc[0]
        meera_img_l = meera_labels[meera_labels['img']==img_name].iloc[0]
        unioned_image = np.zeros_like(mina_img)
        if mina_img_l['EZ disrupted'] and meera_img_l['EZ disrupted']:
            assert 204 in mina_img and 204 in meera_img
            curr_img_label['EZ disrupted'] = 1
            inter_areas['EZ'] += get_intersection_area(mina_img, meera_img, 204)
            inter_img_label['EZ'] += 1
            unioned_image = union_exps_annotations(mina_img, meera_img, unioned_image, 204)
        if mina_img_l['HRD'] and meera_img_l['HRD']:
            assert 153 in mina_img and 153 in meera_img
            curr_img_label['HRD'] = 1
            inter_areas['HRD'] +=  get_intersection_area(mina_img, meera_img, 153)
            inter_img_label['HRD'] += 1
            unioned_image = union_exps_annotations(mina_img, meera_img, unioned_image, 153)
        if mina_img_l['SRF'] and meera_img_l['SRF']:
            assert 102 in mina_img and 102 in meera_img
            curr_img_label['SRF'] = 1
            inter_areas['SRF'] += get_intersection_area(mina_img, meera_img, 102)
            inter_img_label['SRF'] += 1
            unioned_image = union_exps_annotations(mina_img, meera_img, unioned_image, 102)
        if mina_img_l['IRF'] and meera_img_l['IRF']:
            assert 51 in mina_img and 51 in meera_img
            curr_img_label['IRF'] = 1
            inter_areas['IRF'] += get_intersection_area(mina_img, meera_img, 51)
            inter_img_label['IRF'] += 1
            unioned_image = union_exps_annotations(mina_img, meera_img, unioned_image, 51)
        updated_img_labels.append(curr_img_label)
        # if 255 in mina_img or 255 in meera_img:
            # import pdb; pdb.set_trace()
        assert len(np.unique(unioned_image)) == len([k for k, v in curr_img_label.items() if v == 1]) + 1
        save_mask = Image.fromarray((unioned_image).astype(np.uint8))
        save_mask.save('datasets/our_dataset/annot_combine/' + img_name.replace('.jpeg', '.png'))
    inter_areas['EZ'] /= inter_img_label['EZ']
    inter_areas['HRD'] /= inter_img_label['HRD']
    inter_areas['SRF'] /= inter_img_label['SRF']
    inter_areas['IRF'] /= inter_img_label['IRF']
    print(inter_img_label)
    pd.DataFrame(updated_img_labels).to_csv('datasets/our_dataset/combine_labels.csv')
    return inter_areas



if __name__ == "__main__":
    # overlap()
    # filter_noise()
    # remove_background()
    # random_seperate_test()
    # dirs = ["datasets/oct_kaggle/train/0.normal/*"]
    # generate_mask_general(dirs, "datasets/oct_kaggle/normal_mask")
    # import os
    # images = os.listdir('datasets/our_dataset/original/test/')
    # lesion_images = [x for x in images if 'NORMAL' not in x]
    # first_labels = pd.read_csv('datasets/our_dataset/labels.csv')
    # first_labels = first_labels[first_labels['img'].isin(lesion_images)].drop(['Retinal Traction', 'Definite DRIL', 'Questionable DRIL'], axis=1)
    # print(first_labels.sum())

    # genearte_annotation_for_our_dataset(first_labels)
    
    # analyze_annotations()
    # save_human_label()
    intersect_human_annots()
