import shutil
import os, glob
from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
# from medpy.filter.smoothing import anisotropic_diffusion
import cv2 as cv
import torchvision.utils as vutils

def filter_noise():
    for filename in os.listdir('/ssd1/jiaqi/retinal_project/examples/ganomaly/anomaly_dme_rescale'):
        print(filename)
        img_path = os.path.join('/ssd1/jiaqi/retinal_project/examples/ganomaly/anomaly_dme_rescale', '{}'.format(filename))
        image = Image.open(img_path)
        
        # orig_path = os.path.join('our_dataset/test/1.abnormal', '{}'.format(filename))
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
    import subprocess
    from wand.image import Image
    # img = cv.imread('examples/ganomaly/test/1.abnormal/DME-9721607-1.jpeg',0)
    # ny = Image(filename ='examples/ganomaly/test/1.abnormal/DME-9925591-1.jpeg')
    # objects = ny.connected_components()
    # # print(objects)
    # ny.edge(radius = 1)
    # ny.save(filename="edge new york.jpg")
    # cmd = 'convert examples/ganomaly/test/1.abnormal/DME-9925591-1.jpeg -fuzz 30% -trim +repage 0_trim.png'
    # subprocess.check_output(cmd, shell=True, universal_newlines=True)
    # ss
    # with Image(filename='examples/ganomaly/test/test/DME-30521-41.jpeg') as img:
    #     objects = img.connected_components()
    #     print(objects)
    #     for cc_obj in objects:
    #         print("{0._id}: {0.size} {0.offset}".format(cc_obj))
    # cmd = 'convert examples/ganomaly/test/1.abnormal/DME-9721607-1.jpeg -fuzz 30% -trim +repage 0_trim.png'
    # subprocess.check_output(cmd, shell=True, universal_newlines=True)
    #find all your connected components (white blobs in your image)
    # nb_components, output, stats, centroids = cv.connectedComponentsWithStats(img, connectivity=8)
    # # #connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # # #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    # sizes = stats[:, -1]
    # print(nb_components)
    # # print(output)
    # # print(nb_components)

    # max_label = 1
    # max_size = sizes[0]
    # # print(max_size)
    # for i in range(1, nb_components):
    #     if sizes[i] > max_size:
    #         max_label = i
    #         # max_size = sizes[i]
    # img2 = np.zeros(output.shape)
    # img2[output == max_label] = 255
    # cv.imwrite('test.png', output)
    # print(img2)

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

def remove_background():
    imgs_dirs = glob.glob("our_dataset/dataset_DME/4/images/*")#
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
            cv.imwrite('our_dataset/dataset_DME/4/images_backrm/{}'.format(img_name), output)
        # output[output >= 240] = 0
        # cv.imwrite('test.png', output)
    print(count)
    return;

if __name__ == "__main__":
    # overlap()
    # filter_noise()
    remove_background()