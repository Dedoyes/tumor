from re import sub
import SimpleITK as sitk
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
import csv
from torch import sym_max
import torchio as tio
import torch

from torch.utils import data
from torchio.transforms import augmentation

def finalNum (fileName) :
    length = len (fileName)
    res = ""
    for i in range (length - 1, -1, -1) :
        if (fileName[i] == '.') :
            break
        res = res + fileName[i]
    realName = ''.join (reversed (res))
    return realName

def showPic (path, idx, height, width, radius) :
    if os.path.exists (path) :
        image = sitk.ReadImage (path)
        volume = sitk.GetArrayFromImage (image)
        print ("shape : ", volume.shape)                
        circle = Circle ((height, width), radius, color='red', fill=False)
        plt.gca ().add_patch (circle)
        plt.imshow (volume[idx], cmap='gray')
        plt.show ()
        #input ("press enter to continue...")
    else :
        print ("no such file.")
        exit (0)

def getRealxyz (idx, annotations, path) :
    row = annotations.iloc[idx]
    image = sitk.ReadImage (path)
    origin = np.array (image.GetOrigin ())
    spacing = np.array (image.GetSpacing ())
    #direction = np.array (image.GetDirection ())
    world_coord = np.array ([row["coordX"], row["coordY"], row["coordZ"]])
    voxel_coord = np.round ((world_coord - origin) / spacing).astype (int)
    voxel_coord = [voxel_coord[2], voxel_coord[0], voxel_coord[1]]
    return voxel_coord

def resample(img, new_spacing=[1.0, 1.0, 1.0]):
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


def showSlice (tensor) :
    z_index = 32
    slice_z = tensor[z_index, :, :]
    plt.imshow (slice_z, cmap='gray')
    plt.axis ('off')
    plt.show ()

def extract_patch (idx, path, annotations, size=64) :
    row = annotations.iloc[idx]
    image = sitk.ReadImage (path)
    print (image.GetSize ())
    origin = np.array (image.GetOrigin ())
    spacing = np.array (image.GetSpacing ())
    world_coord = np.array ([row["coordX"], row["coordY"], row["coordZ"]])
    voxel_coord = np.round ((world_coord - origin) / spacing).astype (int)
    voxel_coord = [voxel_coord[0], voxel_coord[1], voxel_coord[2]]
    x, y, z = voxel_coord
    print ("x = ", x, ", y = ", y, ", z = ", z)
    half = size // 2
    array = sitk.GetArrayFromImage (image)
    print (array.shape)
    patch = array[
        z - half: z + half, 
        y - half: y + half,
        x - half: x + half
    ]
    print ("x : ", x, "y : ", y, "z : ", z)
    #exit (0)
    #showSlice (patch)
    return patch

def dataImprovment (annotations, ImagePath, cubePath, patch_size=(64, 64, 64), num=100) :
    print (annotations.shape)
    for i in range (0, annotations.shape[0]) :
        rowi = annotations.iloc[i]
        name = rowi['seriesuid'] + '.mhd'
        dataPath = os.path.join (ImagePath, name)
        print (name)
        image = tio.ScalarImage (dataPath)
        patch = extract_patch (i, dataPath, annotations, 64)
        #continue
        patch_tensor = torch.from_numpy (patch).unsqueeze (0).float ()
        print (patch_tensor.shape)
        subject = tio.Subject (image=tio.ScalarImage (tensor=patch_tensor))
        transform = tio.Compose ([
            tio.RandomAffine (
                scales = (0.9, 1.1),
                degrees = 20,
                translation = 5,
            ),
        ])
        sampler = tio.data.UniformSampler (patch_size)
        transformed = transform (subject)
        #continue
        for j in range (0, num) :
            transformed = transform (subject)
            aug_tensor = transformed['image'][tio.DATA].squeeze ().numpy ()
            np.save (os.path.join(cubePath, f'aug_patch_{i:03d}_{j:03d}.npy'), aug_tensor)
            print(f'Saved: aug_patch_{i:03d}_{j:03d}.npy')
            #showSlice (aug_tensor)
    exit (0)
