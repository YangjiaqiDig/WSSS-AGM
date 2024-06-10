# Weakly Supervised Semantic Segmentation on retina OCT images
## Anomaly-Guided ([AGM](https://github.com/YangjiaqiDig/WSSS-AGM/tree/master/anomaly_guided))
🎉 `05/2024` Paper is accepted in Medical Image Analysis (MedAI): 

[Anomaly-guided weakly supervised lesion segmentation on retinal OCT images](https://www.sciencedirect.com/science/article/abs/pii/S1361841524000641)

## Fully-supervised
semantic segmentation with pixel-level annotation on RESC dataset, for comparison
## Structure-Guided (Ongoing)

## Download data
1. The datasets, RESC and Duke, are used by all projects in this repo.

2. The datasets can be downloaded from this Google Drive [link](https://drive.google.com/drive/folders/1IdQUW4zpfnXRsq_8OWdEH90bWR8c9Cod?usp=sharing)

3. The RESC and Duke are .tar.gz files which need to be extracted: ```tar -xzf datasets.tar.gz```. The structure of /your_dir/datasets/ should be organized as follows:
```
---2015_BOE_Chiu/
       --segment_annotation
            --gan_healthy
            --images
            --labels
            --mask
       --train.csv
       --valid.csv

---oct_kaggle/
       --normal_mask
       --origin
            --test/DME
            --train/DME
       --test
            --0.normal
            --1.abnormal
       --train
            --0.normal
            --gan_healthy

---RESC/
       --mask
            --train
            --valid
       --train
            --gan_healthy
            --label_images
            --original_images
       --valid
            --gan_healthy
            --label_images
            --original_images
       --resc_cls_labels.npy
```
