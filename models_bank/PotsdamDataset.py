from torch.utils.data import Dataset
import albumentations as A
import os
from tqdm import tqdm
import skimage.io as io
import earthpy.spatial as es
import numpy as np
import torch
import random

class PotsdamDataset_noNVDI(Dataset):
    def __init__(self, folder, amount=None, probs=None) -> None:
        '''
        folder: data path include both rgb img and gt
        gt: 3 channels map with postfix '_gt'
        '''
        super().__init__()
        
        self.probs = probs

        self.imgs = []
        self.gts = []

        self.train_aug = A.Compose([
            A.RandomScale(scale_limit=(0.5, 1.5), p=0.5),
            A.Resize(512, 512, p=1.0),
            A.Rotate(limit=(-90, 90), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])

        filenames = os.listdir(os.path.join(folder, 'rgb'))
        if amount is not None:
            random.shuffle(filenames)
            filenames = filenames[:amount]  # Limit to the same number of images with vaihingen dataset

        for f in tqdm(filenames):
            img = io.imread(os.path.join(folder, 'rgb', f))

            ndsm_path = os.path.join(folder, 'ndsm', f)
            ndsm = io.imread(ndsm_path)
            ndsm = np.expand_dims(ndsm, axis=2)
            
            gt_path = os.path.join(folder, 'gt', f)
            gt = io.imread(gt_path)
            gt = torch.tensor(gt).unsqueeze(0)

            input_4C = np.dstack((img, ndsm))
            input_4C = torch.tensor(input_4C).permute((2, 0, 1))

            self.imgs.append(input_4C)
            self.gts.append(gt)
        

    def __len__(self):
        return len(self.imgs)
    

    def __getitem__(self, index):
        input_4C = self.imgs[index].permute((1, 2, 0)).numpy()
        gt = self.gts[index].permute((1, 2, 0)).numpy()
        
        after_aug = self.train_aug(image=input_4C, mask=gt)
        input_4C = torch.from_numpy(after_aug['image']).permute((2, 0, 1)).float()
        gt = torch.from_numpy(after_aug['mask']).permute((2, 0, 1))

        mask_array = np.array([[True, True], [True, False], [False, True]])
        if self.probs is not None:
            mask_idx = np.random.choice(3, 1, p=self.probs)
        else:
            mask_idx = np.random.choice(3, 1)

        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)

        return input_4C, gt, mask

class PotsdamDataset(Dataset):
    def __init__(self, folder, mode=None, amount=None) -> None:
        '''
        folder: data path include both rgb img and gt
        gt: 3 channels map with postfix '_gt'
        '''
        super().__init__()

        self.imgs = []
        self.gts = []
        self.mode = mode

        self.train_aug = A.Compose([
            A.RandomScale(scale_limit=(0.5, 1.5), p=0.5),
            A.Resize(512, 512, p=1.0),
            A.Rotate(limit=(-90, 90), p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
        ])

        filenames = os.listdir(os.path.join(folder, 'rgb'))
        if amount is not None:
            filenames = random.shuffle(filenames)
            filenames = filenames[:amount]  # Limit to the same number of images with vaihingen dataset

        for f in tqdm(filenames):
            img = io.imread(os.path.join(folder, 'rgb', f))
            
            nir = img[:,:,0]
            red = img[:,:,1]
            ndvi = es.normalized_diff(nir, red)
            ndvi = np.ma.filled(ndvi, ndvi.min())
            ndvi = np.expand_dims(ndvi, axis=2)

            ndsm_path = os.path.join(folder, 'ndsm', f)
            ndsm = io.imread(ndsm_path)
            ndsm = np.expand_dims(ndsm, axis=2)
            
            gt_path = os.path.join(folder, 'gt', f)
            gt = io.imread(gt_path)
            gt = torch.tensor(gt).unsqueeze(0)

            input_5C = np.dstack((img, ndsm, ndvi))
            input_5C = torch.tensor(input_5C).permute((2, 0, 1))

            self.imgs.append(input_5C)
            self.gts.append(gt)
        

    def __len__(self):
        return len(self.imgs)
    

    def __getitem__(self, index):
        input_5C = self.imgs[index].permute((1, 2, 0)).numpy()
        gt = self.gts[index].permute((1, 2, 0)).numpy()
        
        after_aug = self.train_aug(image=input_5C, mask=gt)
        input_5C = torch.from_numpy(after_aug['image']).permute((2, 0, 1)).float()
        gt = torch.from_numpy(after_aug['mask']).permute((2, 0, 1))

        mask_array = np.array([[True, True, True], [False, True, False], [True, False, True]])

        if self.mode == None:
            mask_idx = np.random.choice(3, 1, p=[0.34, 0.33, 0.33])
        elif self.mode == 'rgb':
            mask_idx = 2
        elif self.mode == 'ndsm':
            mask_idx = 1
        elif self.mode == 'gan_ndsm_to_rgb':
            mask_array = np.array([[True, True], [False, True]])
            mask_idx = np.random.choice(2, 1)
        elif self.mode == 'gan_rgb_to_ndsm':
            mask_array = np.array([[True, True], [True, False]])
            mask_idx = np.random.choice(2, 1)
        elif self.mode == 'shaspec':
            mask_array = np.array([[True, True], [True, False], [False, True]])
            mask_idx = np.random.choice(3, 1)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")
            
        mask = torch.squeeze(torch.from_numpy(mask_array[mask_idx]), dim=0)

        return input_5C, gt, mask
