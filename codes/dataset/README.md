## Dataset
Prepare the [SIDD dataset](https://abdokamel.github.io/sidd/)  
-SIDD Medium: Download 'sRGB images only' (~12 GB).  
-SIDD Validation: Download 'Noisy sRGB data' and 'Ground-truth sRGB data' from 'SIDD Validation Data and Ground Truth'.  
-SIDD Benchmark: Download 'Noisy sRGB data' from 'SIDD Benchmark Data'.  
Prepare the [DND dataset](https://noise.visinf.tu-darmstadt.de/downloads/)  
-DND Benchmark: Download 'Benchmark data' (12.8 GB).  

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
