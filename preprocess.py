import shutil
import os
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion
import cv2 as cv
import torchvision.utils as vutils

def filter_noise():
    for filename in os.listdir('dataset_DR/images'):
        print(filename)
        img_path = os.path.join('dataset_DR/images', '{}'.format(filename))
        image = Image.open(img_path)
        # r, g, b = image.split()
        # b[b<255] =0

        im2 = image.filter(ImageFilter.MinFilter(3))
        im3 = image.filter(ImageFilter.ModeFilter(3))  
        image_arr = np.asarray(im3)
        # result = anisotropic_diffusion(image_arr)
        # print(result.shape)
        imgplotr = plt.imshow(image_arr)
        plt.savefig('tst3.jpeg')
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
    image = Image.open('ganomaly/output/ganomaly/anomaly_dme_2/test/images/001_fake.png')
    image_arr_f = np.array(image)
    # edges = cv.Canny(image_arr_f,300,50)
    # image_arr_f[image_arr_f < 85] = 0
    # im = Image.fromarray(edges)
    # im.save("your_file_f.jpeg")
    
    image_r = Image.open('ganomaly/output/ganomaly/anomaly_dme_2/test/images/001_real.png')
    image_arr_r = np.array(image_r)
    # edges = cv.Canny(image_arr_r,300,300)
    # image_arr_r[image_arr_r < 85] = 0
    # im = Image.fromarray(edges)
    # im.save("your_file_r.jpeg")

    
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


if __name__ == "__main__":
    overlap()