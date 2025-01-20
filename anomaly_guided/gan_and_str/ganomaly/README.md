# Instruction
The AGM's GANomaly pretrained was implemented based on the original GANomaly codebase, and the original GANomaly README is [here](https://github.com/YangjiaqiDig/WSSS-AGM/blob/master/anomaly_guided/gan_and_str/ganomaly/README_GANomaly.md)

# Train within AGM repo

1. Create the virtual environment via conda
    ```
    conda create -n ganomaly python=3.7
    ```
2. Activate the virtual environment.
    ```
    conda activate ganomaly
    ```
3. Install the dependencies under anomaly_guided/gan_and_str/ganomaly.
   ```
   conda install -c intel mkl_fft
   pip install --user --requirement requirements.txt
   ```

4. Training on OCT (can modify the options.py file)

    ``` 
    python train.py \
        --dataset oct                         \
        --display                               # optional if you want to visualize     
    ```

5. To train the model on a OCT dataset, the file structure in datasets/ is:

    ```
    oct_kaggle
    ├── test
    │   ├── 0.normal
    │   │   └── normal_tst_img_0.jpeg
    │   │   └── normal_tst_img_1.jpeg
    │   │   ...
    │   │   └── normal_tst_img_n.jpeg
    │   ├── 1.abnormal
    │   │   └── abnormal_tst_img_0.jpeg
    │   │   └── abnormal_tst_img_1.jpeg
    │   │   ...
    │   │   └── abnormal_tst_img_m.jpeg
    ├── train
    │   ├── 0.normal
    │   │   └── normal_tst_img_0.jpeg
    │   │   └── normal_tst_img_1.jpeg
    │   │   ...
    │   │   └── normal_tst_img_t.jpeg

    ```
