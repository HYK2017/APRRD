# APR-RD: Complemental Two Steps for Self-Supervised Real Image Denoising

## Introduction
The project provides the official PyTorch implementation with pretrained models for the paper "APR-RD: Complemental Two Steps for Self-Supervised Real Image Denoising" (AAAI 2025).    

<p align="center"><img src="figure/Methodology.png" width="950"></p>

## Abstract
Recent advancements in self-supervised denoising have made it possible to train models without needing a large amount of noisy-clean image pairs. A significant development in this area is the use of blind-spot networks (BSNs), which use single noisy images as training pairs by masking some input information to prevent noise transmission to the network output. Researchers have shown that BSNs are capable of reconstructing clean pixels from various types of independent pixel-wise degradations, such as synthetic additive white Gaussian noise (AWGN). However, unlike synthetic noise, real noise often contains highly correlated components which can induce noise transmission and reduce the performance of BSNs. To address the spatial correlation of real noise, we propose the Adjacent Pixel Replacer (APR), which decorrelates noise without a downsampling process that is widely adopted in previous research. The dissimilarity in our APR-generated pairs serves as relatively different noise components during training. Hence, it enables the BSN to block noise transmission while utilizing clean information effectively. As a result, BSN can utilize denser information to reconstruct the corresponding center pixel. We also propose Recharged Distillation (RD) to enhance high-frequency textures without additional network modifications. This method selectively refines clean information from recharged noisy pixels during distillation. Extensive experimental results demonstrate that our proposed method outperforms the existing state-of-the-art self-supervised denoising methods in real sRGB space.

## Environment
- Ubuntu 20.04.5 LTS
- Python 3.8.10
- Pytorch 1.12.1 (CUDAtoolkit=11.3)
- RTX 2080ti * 2ea

## Code Structure
```
├─ codes/
│    ├─ dataset/
│    │    └─ README.md/            # Download each dataset and organize them according to the given instructions.
│    │    
│    ├─ network/                 
│    │    ├─ __init__.py   
│    │    ├─ baseline_network.py   # Network modules of AT-BSN and NBSN
│    │    └─ multi_target_T        # Inference module for obtaining multi-target T from the trained AT-BSN.
│    │
│    └─ utils/
│         ├─ __init__.py
│         ├─ dataloader.py    # SIDD Medium loader for training / SIDD Val, Bench, and DND Bench loaders for inference / denoised result(SIDD Val) loader for evaluation.
│         ├─ function.py      # Modules for the proposed methods, APR and Recharger.
│         ├─ metric.py        # PSNR and SSIM calculation module
│         └─ utility.py       # Loss functions, scheduler
│   
├─ config.json     # GPU settings, data and result paths, training hyperparameters, etc.
├─ preparation.py  # Modularization of train, test, and validate.
├─ train.py        # Model initialization, loading training data, training execution.
├─ test.py         # loading trained model, test execution.
├─ validate.py     # Calculation of evaluation results for SIDD Val.
```

## Dataset
Prepare the [SIDD dataset](https://abdokamel.github.io/sidd/)  
-SIDD Medium: Download `sRGB images only` (~12 GB).  
-SIDD Validation: Download `Noisy sRGB data` and `Ground-truth sRGB data` from `SIDD Validation Data and Ground Truth`.  
-SIDD Benchmark: Download `Noisy sRGB data` from `SIDD Benchmark Data`.  
Prepare the [DND dataset](https://noise.visinf.tu-darmstadt.de/downloads/)  
-DND Benchmark: Download `Benchmark data` (12.8 GB).  

For training and evaluation using our code, organize the above prepared datasets as follows.   
```
dataset/
   ├─ SIDD_Medium_Srgb/
   │    ├─ Data/
   │    │    ├─ 0001_001_S6_00100_00060_3200_L/
   │    │    │    ├─ 0001_GT_SRGB_010.png
   │    │    │    ├─ 0001_GT_SRGB_011.png
   │    │    │    ├─ 0001_NOISY_SRGB_010.png
   │    │    │    └─ 0001_NOISY_SRGB_011.png
   │    │    ..
   │    │    └─ 0200_010_GP_01600_03200_5500_N/
   │    │         ├─ 0200_GT_SRGB_010.png
   │    │         ├─ 0200_GT_SRGB_011.png
   │    │         ├─ 0200_NOISY_SRGB_010.png
   │    │         └─ 0200_NOISY_SRGB_011.png
   │    ├─ ReadMe_sRGB.txt
   │    └─ Scene_Instances.txt
   │
   ├─ SIDD_Validation_sRGB/
   │    ├─ ValidationGtBlocksSrgb.mat
   │    └─ ValidationNoisyBlocksSrgb.mat
   │
   ├─ SIDD_Benchmark_sRGB/
   │    └─ BenchmarkNoisyBlocksSrgb.mat
   │
   └─ dnd_2017/
        ├─ images_raw
        │    ├─ 0001.mat
        │    ├─ 0002.mat
        │    ..
        │    └─ 00050.mat
        ├─ images_srgb
        │    ├─ 0001.mat
        │    ├─ 0002.mat
        │    ..
        │    └─ 00050.mat
        ├─ info.mat
        └─ pixelmasks.mat
```

## Train, Test and Validation  
Trainig of APR and RD using SIDD Medium.  
-Each traiend model is stored as `../trained/APR.pth` and `../trained/APR_RD.pth`.  
```shell
python train.py
```

Test on SIDD Validation, Benchmark, and DND Benchmark using the two trained models.  
-Loading the model parameters stored as `../pretrained/APR_pre.pth` and `../pretrained/APR_RD_pre.pth`.  
-Each result is stored in `../result/SIDD_Validation_inf`, `../result/SIDD_Benchmark_inf`, and `../result/DND_Benchmark_inf`, respectively.  
```shell
python test.py
```

Validation  
-Calculation of PSNR and SSIM for the test results of SIDD Val.  
-SIDD Bench and DND Bench require separate processing for submission (refer to their websites).  
```shell
python validate.py
```

## Acknowledgements
This project is based on the following open-source projects. In particular, we benefit from the convenience of using the network of AT-BSN code as it is. We sincerely appreciate their authors for allowing us to use their excellent works as open-source.
- [AT-BSN](https://github.com/hnmizuho/AT-BSN)
- [SIDD](https://abdokamel.github.io/sidd/)
- [DND](https://noise.visinf.tu-darmstadt.de/)
