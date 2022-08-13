import os, glob
from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
# from medpy.filter.smoothing import anisotropic_diffusion
import cv2 as cv

import pandas as pd

def filter_noise():
    for filename in os.listdir('/ssd1/jiaqi/retinal_project/examples/ganomaly/anomaly_dme_rescale'):
        print(filename)
        img_path = os.path.join('/ssd1/jiaqi/retinal_project/examples/ganomaly/anomaly_dme_rescale', '{}'.format(filename))
        image = Image.open(img_path)
        
        # orig_path = os.path.join('datasets/our_dataset/test/1.abnormal', '{}'.format(filename))
        # orig = Image.open(orig_path)
        # r, g, b = image.split()
        # b[b<255] =0

        im2 = image.filter(ImageFilter.MinFilter(3))
        # im = orig.filter(ImageFilter.MinFilter(3))
        
        # im3 = image.filter(ImageFilter.ModeFilter(3))  
        im2.save('{}'.format(filename))
        # im.save('org.jpeg')
        
        import pdb; pdb.set_trace()
        # image_arr = np.asarray(im3)
        # result = anisotropic_diffusion(image_arr)
        # print(result.shape)
        # imgplotr = plt.imshow(image_arr)
        # plt.savefig('tst3.jpeg')
        # plt.show()

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
    return;

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

def generate_mask_resc():
    list_of_data = glob.glob("datasets/2015_BOE_Chiu/segment_annotation/images/*")
    # print(list_of_data)
    for item in list_of_data:
        image_name = item.split('/')[-1]
        res = generate_background_mask(item)
        cv.imwrite('datasets/2015_BOE_Chiu/segment_annotation/mask/{}'.format(image_name), res)

    # list_of_test = glob.glob("RESC/valid/original_images/*")
    # for item in list_of_test:
    #     image_name = item.split('/')[-1]
    #     res = generate_background_mask(item)
    #     cv.imwrite('RESC/mask/valid/{}'.format(image_name), res)

def generate_healthy_gan_resc():
    return

if __name__ == "__main__":
    # overlap()
    # filter_noise()
    # remove_background()
    # random_seperate_test()
    generate_mask_resc()