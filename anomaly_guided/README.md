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

## Preparing Datasets (coming soon)
RESC 

Duke

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