#from pandas import annotations
from torch import utils
import torchio as tio
import numpy as np
import os
from scipy.spatial.transform import Rotation as R
import SimpleITK as sitk
import pandas as pd
import utilize
import random

def create_patch_grid (patch_size=(64, 64, 64)) :
    dz, dy, dx = patch_size
    z = np.arange (-(dz // 2), dz // 2)
    y = np.arange (-(dy // 2), dy // 2)
    x = np.arange (-(dx // 2), dx // 2)
    zz, yy, xx = np.meshgrid (z, y, x, indexing='ij')
    coords = np.stack ([zz, yy, xx], axis=-1)
    return coords

def get_affine_matrix (scale=1.0, angles=(0, 0, 0), translation=(0, 0, 0)) :
    rot = R.from_euler ('zyx', angles, degrees=True).as_matrix ()
    affine = scale * rot
    affine_matrix = np.eye (4)
    affine_matrix[:3, :3] = affine
    affine_matrix[:3, 3] = np.array (translation)
    return affine_matrix

def transform_coords (coords, affine_matrix, center_world_coord) :
    D, H, W, _ = coords.shape
    flat = coords.reshape (-1, 3).T
    ones = np.ones ((1, flat.shape[1]))
    homo = np.vstack ([flat, ones])
    transformed = affine_matrix @ homo
    world_coords = transformed[:3].T + np.array (center_world_coord)
    return world_coords.reshape (D, H, W, 3)

def dataImprovment (annotations, ImagePath, cubePath, patch_size=(64, 64, 64), num=100) :
    print (annotations.shape)
    tot = 0
    for i in range (0, annotations.shape[0]) :
        rowi = annotations.iloc[i]
        name = rowi['seriesuid'] + '.mhd'
        dataPath = os.path.join (ImagePath, name)
        print (name)
        image = sitk.ReadImage (dataPath)
        origin = np.array (image.GetOrigin ())
        spacing = np.array (image.GetSpacing ())
        world_coords = np.array ([rowi['coordX'], rowi['coordY'], rowi['coordZ']])
        grid = create_patch_grid (patch_size)
        scaleSet = [0.9, 1.0, 1.1]
        angleSet = [(0, 0, 0), (0, 0, 15), (0, 15, 0), (0, 15, 15), (15, 0, 0), (15, 0, 15), (15, 15, 0), (15, 15, 15)]
        translationSet = []
        for x in [5, 0, -5] : 
            for y in [5, 0, -5] :
                for z in [5, 0, -5] :
                    translationSet.append ((x, y, z))
        for scale in scaleSet :
            for angles in angleSet :
                for translation in translationSet :
                    randVal = random.randint (0, 10)
                    if randVal != 0 :
                        continue
                    affine = get_affine_matrix (scale, angles, translation)
                    transformed_coords = transform_coords (grid, affine, center_world_coord=world_coords)
                    sampling_voxel_coords = (transformed_coords - origin) / spacing
                    array = sitk.GetArrayFromImage (image)
                    #print (sampling_voxel_coords.shape)
                    sampling_voxel_coords = sampling_voxel_coords.transpose (3, 0, 1, 2)
                    voxel_coords_rounded = np.round (sampling_voxel_coords).astype (int)
                    z = voxel_coords_rounded[0]
                    y = voxel_coords_rounded[1]
                    x = voxel_coords_rounded[2]
                    z[z < 0] = 0 
                    z[z >= array.shape[2]] = array.shape[2] - 1
                    y[y < 0] = 0
                    y[y >= array.shape[1]] = array.shape[1] - 1
                    x[x < 0] = 0
                    x[x >= array.shape[0]] = array.shape[0] - 1
                    print ("scale : ", scale)
                    print ("angles : ", angles)
                    print ("translationSet : ", translation)
                    #print ("z : ", z)
                    #print ("y : ", y)
                    #print ("x : ", x)
                    #print ("array shape : ", array.shape)
                    patch = array[x, y, z]
                    #utilize.showSlice (patch)
                    tot += 1
                    np.save (os.path.join (cubePath, f"{tot:06d}.npy"), patch)
                    print ("tot = ", tot)
        #exit (0)

def tumorExtract (newCandidatePath, tumorPath) :
    if os.path.exists (tumorPath) :
        tumorSet = pd.read_csv (tumorPath)
        return tumorSet
    candidate = pd.read_csv (newCandidatePath)
    tumorSet = candidate[candidate['class'] == 1]
    tumorSet.to_csv (tumorPath, index=False)
    return tumorSet

if __name__ == '__main__' :
    srcPath = os.path.dirname (os.path.abspath (__file__))
    basePath = os.path.dirname (srcPath)
    dataPath = os.path.join (basePath, "data")
    subset0Path = os.path.join (dataPath, "subset0")
    annotationsPath = os.path.join (basePath, "data/annotations.csv")
    #annotations = pd.read_csv (annotationsPath)
    candidatePath = os.path.join (basePath, "data/candidates.csv")
    candidate = pd.read_csv (candidatePath)
    newCandidatePath = os.path.join (dataPath, "newCandidates.csv")
    tumorPath = os.path.join (dataPath, "tumor.csv")
    testPath = os.path.join (dataPath, "test.csv")
    cubePath = os.path.join (dataPath, "cube")

    tumorSet = tumorExtract (newCandidatePath, tumorPath)
    dataImprovment (tumorSet, subset0Path, cubePath)
