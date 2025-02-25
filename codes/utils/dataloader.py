import os
import cv2
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import scipy.io as io
import h5py
import numpy as np
import sys

class TrainLoader_SIDD_MD(Dataset):        
    def __init__(self, root_dir, patch_size=256, n_patches=500, transform=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.transform = transform if transform else transforms.ToTensor()
        
        self.images = self._load_noisy_images()

    def _load_noisy_images(self):
        images = []
        subfolders = os.listdir(self.root_dir)
        total_files = 0
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                total_files += sum(1 for file in os.listdir(subfolder_path) if 'NOISY' in file and file.lower().endswith(('png', 'jpg', 'jpeg')))
        
        idx = 0
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if 'NOISY' in file and file.lower().endswith(('png', 'jpg', 'jpeg')):
                        image_path = os.path.join(subfolder_path, file)
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        
                        patch_info = self._patch_info(image)
                        images.append([image,patch_info])      
                        idx += 1
                        sys.stdout.write(f"\r({int(idx/total_files*100)}/100)% training dataset loading..")
                        sys.stdout.flush()
        sys.stdout.write("\n")
        return images
    
    def _patch_info(self, image):
        h, w, _ = image.shape
        top = np.random.randint(0, max(0, h - self.patch_size), (self.n_patches, 1))
        left = np.random.randint(0, max(0, w - self.patch_size), (self.n_patches, 1))
        return np.concatenate((top, left), axis=1)

    def _crop_patches(self, image, patch_info):
        patches = []
        for patch_idx in range(self.n_patches):
            top, left = patch_info[patch_idx]
            patch = image[top:top + self.patch_size, left:left + self.patch_size]
            patch = self.transform(patch)
            patches.append(patch)
        return patches

    def __len__(self):
        return len(self.images) * self.n_patches

    def __getitem__(self, idx):
        image_idx = idx // self.n_patches
        patch_idx = idx % self.n_patches
        
        image, patch_info = self.images[image_idx]
        patches = self._crop_patches(image, patch_info)
        return patches[patch_idx]

    def get_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class TrainLoader_SIDD_MD_old(Dataset):        
    def __init__(self, root_dir, patch_size=256, n_patches=500, transform=None):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.n_patches = n_patches
        self.transform = transform if transform else transforms.ToTensor()
        
        self.images = self._load_noisy_images()

    def _load_noisy_images(self):
        images = []
        subfolders = os.listdir(self.root_dir)
        total_files = 0
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                total_files += sum(1 for file in os.listdir(subfolder_path) if 'NOISY' in file and file.lower().endswith(('png', 'jpg', 'jpeg')))
        
        idx = 0
        for subfolder in subfolders:
            subfolder_path = os.path.join(self.root_dir, subfolder)
            if os.path.isdir(subfolder_path):
                for file in os.listdir(subfolder_path):
                    if 'NOISY' in file and file.lower().endswith(('png', 'jpg', 'jpeg')):
                        image_path = os.path.join(subfolder_path, file)
                        image = cv2.imread(image_path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        images.append(image)
                                      
                        idx += 1
                        sys.stdout.write(f"\r({int(idx/total_files*100)}/100)% training dataset loading..")
                        sys.stdout.flush()
        sys.stdout.write("\n")
        return images

    def _crop_patches(self, image):
        h, w, _ = image.shape
        patches = []
        for _ in range(self.n_patches):
            top = random.randint(0, max(0, h - self.patch_size))
            left = random.randint(0, max(0, w - self.patch_size))
            patch = image[top:top + self.patch_size, left:left + self.patch_size]
            patch = self.transform(patch)
            patches.append(patch)
        return patches

    def __len__(self):
        return len(self.images) * self.n_patches

    def __getitem__(self, idx):
        image_idx = idx // self.n_patches
        patch_idx = idx % self.n_patches
        
        image = self.images[image_idx]
        patches = self._crop_patches(image)
        return patches[patch_idx]

    def get_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class TestLoader_SIDD_VAL_BM(Dataset):
    def __init__(self, root_dir, GT=False, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.GT = GT
        if self.GT == True:
            self.blocks = 'GtBlocksSrgb'
        else:
            self.blocks = 'NoisyBlocksSrgb'
        
        self.test_patches = self._load_noisy_patches()
        self.test_patches_shape = self.test_patches.shape

    def _load_noisy_patches(self):
        mat_files = [f for f in os.listdir(self.root_dir) if self.blocks in f and f.endswith('.mat')]
        if not mat_files:
            raise FileNotFoundError(f"No .mat file containing {self.blocks} found in the directory.")
        mat_file_path = os.path.join(self.root_dir, mat_files[0])
        data = io.loadmat(mat_file_path, variable_names=list(io.whosmat(mat_file_path)[0])[0])
        if self.GT == True:
            print("(100/100)% GT loading..")
        else:
            print("(100/100)% test dataset loading..")
        return data[list(data.keys())[3]]  # (40, 32, 256, 256, 3) 데이터 반환

    def __len__(self):
        return self.test_patches_shape[0] * self.test_patches_shape[1]  # 40 * 32 = 1280

    def __getitem__(self, idx):
        image_idx = idx // self.test_patches_shape[1]
        patch_idx = idx % self.test_patches_shape[1]

        patch = self.test_patches[image_idx, patch_idx]
        return self.transform(patch)

    def get_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)

class TestLoader_DND_BM(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()

        self.mat_files, self.bounding_boxes, self.cumulative_patches = self._load_mat_metadata()
        self.mat_data_list = self._load_noisy_images()

    def _load_mat_metadata(self):
        mat_files = [
            os.path.join(self.root_dir, 'images_srgb', file)
            for file in os.listdir(os.path.join(self.root_dir, 'images_srgb'))
            if file.endswith('.mat')
        ]

        with h5py.File(os.path.join(self.root_dir, 'info.mat'), 'r') as info:
            bb = info['info']['boundingboxes']
            bounding_boxes = [np.array(info[bb[0][img_idx]]).T for img_idx in range(len(mat_files))]
        cumulative_patches = np.cumsum([len(boxes) for boxes in bounding_boxes])
        return mat_files, bounding_boxes, cumulative_patches

    def _load_noisy_images(self):
        mat_data_list = []
        total_files = len(self.mat_files)

        for idx, mat_file in enumerate(self.mat_files, start=1):
            with h5py.File(mat_file, 'r') as dnd_mat:
                mat_data_list.append(np.array(dnd_mat['InoisySRGB']))
            sys.stdout.write(f"\r({int(idx/total_files*100)}/100%) test dataset loading..")
            sys.stdout.flush()
        sys.stdout.write("\n")
        return mat_data_list
    
    def _crop_patches(self, img_idx, patch_idx):
        patch_box = self.bounding_boxes[img_idx][patch_idx]
        x1, x2, y1, y2 = int(patch_box[0] - 1), int(patch_box[2]), int(patch_box[1] - 1), int(patch_box[3])
        return self.mat_data_list[img_idx][:, y1:y2, x1:x2].T

    def __len__(self):
        return self.cumulative_patches[-1]

    def __getitem__(self, idx):
        img_idx = np.searchsorted(self.cumulative_patches, idx, side='right')
        patch_idx = idx - (self.cumulative_patches[img_idx - 1] if img_idx > 0 else 0)

        patch = self._crop_patches(img_idx, patch_idx)
        return self.transform(patch)

    def get_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)  

class ResultLoader(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.ToTensor()
        self.images = self._load_images()

    def _load_images(self):
        images = []
        allowed_extensions = ('png', 'jpg', 'jpeg')
        files = os.listdir(self.root_dir)
        total_files = len(files)
        idx = 0

        for file in files:
            if file.lower().endswith(allowed_extensions):
                image_path = os.path.join(self.root_dir, file)
                image = cv2.imread(image_path)
                if image is not None:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    idx += 1
                    sys.stdout.write(f"\r({int(idx/total_files*100)}/100)% loading images...")
                    sys.stdout.flush()
        sys.stdout.write("\n")
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image
    
    def get_dataloader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)  