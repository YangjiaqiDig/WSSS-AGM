# OCT_Retinal_Project
Retinal Classification

LABELS = ['srf', 'irf', 'ezAtt', 'ezDis', 'hrd', 'rpe', 'rt', 'dril']
Edema: 0.9266381766381766 0.9036087369420625 0.9266381766381766
Val loss: 0.17094962520343362 Val acc: {'acc': 0.9549763033175356, 'f1m': 0.9407582938388621, 'f1mi': 0.9549763033175356}

gan_and_str: source code for GANomaly and P-Net structure

remove background: Gan - normal images: 14606 / 26315   our dataset: ~55%


Interesting to show: 
- outputs/naive_aug_gan/fold-6/iteration/DME-3157783-98/epoch24__IRF_EZ_HRD_BackGround.jpg
- outputs/naive_aug_gan/fold-6/iteration/DME-3289980-7/epoch24__IRF_EZ_BackGround.jpg
- outputs/naive_aug_gan/fold-6/iteration/DME-3342858-2/epoch24__IRF_EZ_BackGround.jpg

Labeling mistake correction:
DME-3157783-106 -> add IRF
Another one on 1st ppt, need double check


Ablation work flow and model buildup tryings


## Experiment process and save direct
- origin

