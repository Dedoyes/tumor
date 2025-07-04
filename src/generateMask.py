import os
import pandas as pd
from pandas._config.config import is_instance_factory
from pandas.io.pytables import adjoin
import utilize
import SimpleITK as sitk
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
#from scipy.ndimage import rotate, shift, zoom

def center_crop (image, size=512) :
    h, w = image.shape[:2]
    startX = w // 2 - size // 2
    startY = h // 2 - size // 2
    return image[startY:startY + size, startX:startX + size]

def get_affine_matrix (angle_deg, scale, dx, dy) :
    angle_rad = np.deg2rad (angle_deg)
    cos_a = np.cos (angle_rad) * scale
    sin_a = np.sin (angle_rad) * scale
    affine_maxtrix = torch.tensor ([
        [cos_a, -sin_a, dx],
        [sin_a, cos_a, dy],
    ], dtype=torch.float32)
    return affine_maxtrix

def apply_affine_torch(image, angle=0, scale=1.0, dx=0.0, dy=0.0):
    """ image: 2D numpy array (H, W) """
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()

    # Add batch and channel dims
    image = image.unsqueeze(0).unsqueeze(0)  # shape (1, 1, H, W)
    _, _, H, W = image.shape

    # dx, dy 是 pixel 坐标，要换算成 [-1, 1] 归一化坐标
    tx = dx / (W / 2)
    ty = dy / (H / 2)

    affine = get_affine_matrix(angle, scale, tx, ty).unsqueeze(0)  # shape (1, 2, 3)
    grid = F.affine_grid(affine, image.size (), align_corners=False)
    output = F.grid_sample(image, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
    return output.squeeze().numpy()  # 去掉 batch/channel

def random_rotate_scale(image, angle_range=90, scale_range=(0.5, 1.5)):
    angle = np.random.uniform(-angle_range, angle_range)
    scale = np.random.uniform(*scale_range)
    return apply_affine_torch(image, angle=angle, scale=scale)

def random_translate(image, max_shift=20):
    dx = np.random.randint(-max_shift, max_shift)
    dy = np.random.randint(-max_shift, max_shift)
    return apply_affine_torch(image, dx=dx, dy=dy)

def augment_image(first, second):
    angle = np.random.uniform (-90, 90)
    scale = np.random.uniform (*(0.5, 1.5))
    dx = np.random.randint (-30, 30)
    dy = np.random.randint (-30, 30)
    img_first = apply_affine_torch (first, angle=angle, scale=scale)
    img_first = apply_affine_torch (img_first, dx=dx, dy=dy)
    img_second = apply_affine_torch (second, angle=angle, scale=scale)
    img_second = apply_affine_torch (img_second, dx=dx, dy=dy)
    return img_first, img_second


srcPath = os.path.dirname (os.path.abspath (__file__))
basePath = os.path.dirname (srcPath)
dataPath = os.path.join (basePath, "data")
annotationsPath = os.path.join (dataPath, "annotations.csv")
imagePath = os.path.join (dataPath, "subset0")
maskPath = os.path.join (dataPath, "mask")
candidateImagePath = os.path.join (dataPath, "candidate_image")

annotations = pd.read_csv (annotationsPath)
print (annotations.shape)

image_map = {}
mask_map = {}
for i in range (0, annotations.shape[0]) :
    rowi = annotations.iloc[i]
    name = rowi['seriesuid']
    radius = rowi['diameter_mm']
    filePath = os.path.join (imagePath, name + ".mhd")
    if os.path.exists (filePath) :
        print ("find")
        print (filePath)
        image = sitk.ReadImage (filePath)
        volume = sitk.GetArrayFromImage (image)
        z, x, y = utilize.getRealxyz (i, annotations, filePath)
        spacing = np.array (image.GetSpacing ())
        real_radius = radius / spacing
        r = int ((real_radius[0] + real_radius[1] + real_radius[2]) // 3)
        print ("real_radius = ", real_radius)
        #utilize.showPic (filePath, z, x, y, r)
        #useless = input ("suspend")
        value = mask_map.get (name)
        if value is None :
            mask_map[name] = torch.zeros ((512, 512), dtype=torch.float32)
            image_map[name] = volume[z]
            #plt.imshow (volume[z], cmap='grey')
            #plt.show ()
        centerX = int (x)
        centerY = int (y)
        for posX in range (centerX - r, centerX + r + 1) :
            for posY in range (centerY - r, centerY + r + 1) :
                dis = (centerX - posX) ** 2 + (centerY - posY) ** 2
                if (dis <= r ** 2) :
                    mask_map[name][posY, posX] = 1
        #plt.imshow (mask_map[name], cmap='grey')
        #plt.show ()
print (mask_map.__len__ ())
tot = 0
for first, second in mask_map.items () :
    for i in range (0, 100) :
        tot += 1
        #plt.imshow (image_map[first], cmap='grey')
        #plt.show ()
        #plt.imshow (second, cmap='grey')
        #plt.show ()
        aug_first, aug_second = augment_image (image_map[first], second)
        #plt.imshow (aug_first, cmap='grey')
        #plt.show ()
        #plt.imshow (aug_second, cmap='grey')
        #plt.show ()
        generate_path = os.path.join (maskPath, f"{tot}.pt")
        image_path = os.path.join (candidateImagePath, f"{tot}.pt")
        print (f"{tot}.pt save success.")
        torch.save (second, generate_path)
        torch.save (image_map[first], image_path)
