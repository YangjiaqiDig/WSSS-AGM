# Anomaly-guided weakly supervised lesion segmentation on retinal OCT images


Pytorch code for the paper : "Anomaly-guided weakly supervised lesion segmentation on retinal OCT images".


<img src="figures/overview.png" width="800" height="285"/>


## Reqirements

```
# create conda env
conda create -n wsss-agm python=3.9
conda activate wsss-agm

# install packages
pip install torch torchvision torchaudio
pip install opencv-python tqdm tensorboard cython
pip install grad-cam
```

## Preparing Datasets
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
## Weights Download
Same Google Drive as Datasets: [link](https://drive.google.com/drive/folders/1p0ruQ7V8YUuBfU87bizpT4WYtrFeckSK)

AGM weights: resc & duke

GANomaly pretrained weight: gan
## Training
```
python run_train.py
```

## Contact

For further questions or details, please post an issue or directly reach out to Jiaqi Yang (jyang2@gradcenter.cuny.edu)

## Acknowledgement
We used [GANomaly](https://github.com/samet-akcay/ganomaly) to generate normal OCT images and used [pytorch_grad_cam](https://github.com/jacobgil/pytorch-grad-cam/tree/61e9babae8600351b02b6e90864e4807f44f2d4a) to generate GradCAM results. Thanks for their wonderful works.


## Citation
If you find this project helpful for your research, please consider citing the following BibTeX entry.
```
@article{yang2024anomaly,
  title={Anomaly-guided weakly supervised lesion segmentation on retinal OCT images},
  author={Yang, Jiaqi and Mehta, Nitish and Demirci, Gozde and Hu, Xiaoling and Ramakrishnan, Meera S and Naguib, Mina and Chen, Chao and Tsai, Chia-Ling},
  journal={Medical Image Analysis},
  volume={94},
  pages={103139},
  year={2024},
  publisher={Elsevier}
}
```