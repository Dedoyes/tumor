import os
from sys import orig_argv
from matplotlib import transforms
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import Dataset
import utilize

class luna16Dataset (Dataset) :
    def __init__ (self, imageDir, annotations, transform=None, patch_size=64) :
        self.imageDir = imageDir
        self.annotations = annotations
        self.transform = transform
        self.patch_size = patch_size
        self.image_map = {}
        for fileName in os.listdir (imageDir) :
            if fileName.endswith (".mhd") :
                seriesuid = os.path.splitext (fileName)[0]
                fullPath = os.path.join (imageDir, fileName)
                if os.path.exists (fullPath) :
                    self.image_map[seriesuid] = fullPath

    def __len__ (self) :
        return len (self.annotations)


    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]
        uid = row["seriesuid"]
        kind = row["class"]
        mhdPath = self.image_map.get(uid)
        if mhdPath is None:
            return  # ⚠️ 可能导致 DataLoader 报错，建议抛异常或跳过

        # 读取图像 + 重采样
        image = sitk.ReadImage(mhdPath)
        #image = self._resample(image, new_spacing=[1.0, 1.0, 1.0])  # ✅ 统一 spacing
        volume = sitk.GetArrayFromImage(image)  # shape: [z, y, x]
        
        #print ("shape of volume : ",volume.shape)
        z, x, y = utilize.getRealxyz(idx, self.annotations, mhdPath)
        r = self.patch_size // 2

        # 确保索引不越界
        z = min(max(z, r), volume.shape[0] - r)
        x = min(max(x, r), volume.shape[1] - r)
        y = min(max(y, r), volume.shape[2] - r)

        #print (mhdPath)
        patch = volume[
            z - r : z + r,
            y - r : y + r,
            x - r : x + r
        ]
        #print ("r = ", r)
        #print (patch.shape)
        #utilize.showSlice (patch)
        patch = patch.astype(np.float32)
        patch = self._pad_if_needed(patch, self.patch_size)
        patch = patch[None, :, :, :]  # shape: [1, D, H, W]

        if self.transform:
            patch = self.transform(patch)

        return torch.tensor(patch), np.float32(kind)
        
    def _pad_if_needed (self, patch, target_size) :
        pad = [(0, max (0, target_size - s)) for s in patch.shape]
        return np.pad (patch, pad, mode='constant', constant_values=0)
    
    def _resample(self, img, new_spacing=[1.0, 1.0, 1.0]):
        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        new_size = [
            int(round(osz * ospc / nspc))
            for osz, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]

        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(img.GetDirection())
        resampler.SetOutputOrigin(img.GetOrigin())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)

        return resampler.Execute(img)

def collect_data_paths (benign_cube, tumor_cube) :
    data = []
    tumor_tot = 0
    benign_tot = 0
    for file in os.listdir (benign_cube) :
        if file.endswith ('.npy') :
            path = os.path.join (benign_cube, file)
            temp = np.load (path)
            if temp.shape == (64, 64, 64) :
                data.append ((path, 0))
                benign_tot += 1
    for file in os.listdir (tumor_cube) :
        if file.endswith ('.npy') :
            path = os.path.join (tumor_cube, file)
            temp = np.load (path)
            if temp.shape == (64, 64, 64) :
                data.append ((path, 1))
                tumor_tot += 1
    print ("tumor_tot = ", tumor_tot, "benign_tot = ", benign_tot)
    return data

class NpyDataset (Dataset) :
    def __init__ (self, data_list, transform=None) :
        self.data_list = data_list
        self.transform = transform

    def __len__ (self) :
        return len (self.data_list)

    def __getitem__ (self, idx) :
        path, label = self.data_list[idx]
        array = np.load (path)
        array = array.astype (np.float32)
        array = array / (np.max (array) + 1e-6)
        tensor = torch.from_numpy (array).unsqueeze (0)
        if self.transform :
            tensor = self.transform (tensor)
        return tensor, torch.tensor (label, dtype=torch.float32) 
