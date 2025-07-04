import torch
import numpy as np
from PIL import Image
import os
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.transforms.functional import to_tensor

class UnetDataSet (Dataset) :
    def __init__ (self, mask_dir, image_dir, transform=None) :
        super ().__init__ ()
        self.mask_dir = mask_dir
        self.image_dir = image_dir
        self.image_names = sorted (os.listdir (image_dir))
        self.mask_names = sorted (os.listdir (mask_dir))
        self.transform = transform

    def __len__ (self) :
        return len (self.image_names)

    def __getitem__ (self, idx) :
        image_path = os.path.join (self.image_dir, self.image_names[idx])
        mask_path = os.path.join (self.mask_dir, self.image_names[idx])
        image = torch.load (image_path, weights_only=False)
        mask = torch.load (mask_path, weights_only=False)
        image = image / 255.0
        mask = mask / 255.0
        #print (image.shape)
        image = torch.from_numpy (image)
        image = image.to (torch.float32)
        #mask = torch.from_numpy (mask)
        mask = mask.to (torch.float32)
        if self.transform :
            image = self.transform (image)
            mask = self.transform (mask)
        #print ("image.shape = ", image.shape)
        #print ("mask.shape = ", mask.shape)
        return image.unsqueeze (0), mask.unsqueeze (0)
