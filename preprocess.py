import shutil
import os
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from medpy.filter.smoothing import anisotropic_diffusion

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
    sss