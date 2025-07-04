import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Unet
import utilize
import os
import argparse
import pandas as pd
import UnetDataSet
import torch.optim as optim
import matplotlib.pyplot as plt

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
cubePath = os.path.join (dataPath, "tumor_cube")
benignPath = os.path.join (dataPath, "benign_cube")

maskPath = os.path.join (dataPath, "mask")
imagePath = os.path.join (dataPath, "candidate_image")

unetPthPath = os.path.join (srcPath, "unet.pth")

dataset = UnetDataSet.UnetDataSet (maskPath, imagePath)
loader = DataLoader (dataset, batch_size=1, shuffle=False)

unet = Unet (use_dropout=True, features=128).cuda ()
pos_weight = torch.tensor ([10]).cuda ()
criterion = nn.BCEWithLogitsLoss (pos_weight=pos_weight)
optimizer = optim.Adam (unet.parameters (), lr = 1e-4)

start_epoch = 0
num_epoches = 1000

if os.path.exists (unetPthPath) :
    checkpoint = torch.load (unetPthPath)
    unet.load_state_dict (checkpoint['model_state_dict'])
    optimizer.load_state_dict (checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

for epoch in range (start_epoch, num_epoches) :
    unet.train ()
    total_loss = 0
    sum = 0
    for images, masks in loader :
        if True :
            plt.imshow (images.squeeze (0).squeeze (0).detach ().numpy (), 'grey')
            plt.show ()
            plt.imshow (masks.squeeze (0).squeeze (0).detach ().numpy (), 'grey')
            plt.show ()
        sum += 1
        print (sum)
        r = torch.max (images)
        l = torch.min (images)
        images = images / (r - l)
        images -= torch.min (images)
        r = torch.max (masks)
        l = torch.min (masks)
        masks = masks / (r - l)
        masks -= torch.min (masks)
        #print (torch.max (images), torch.min (images))
        images, masks = images.cuda (), masks.cuda ()
        outputs = unet (images)
        if True :
            temp = torch.sigmoid (outputs)
            temp = temp.to ("cpu")
            temp = temp.squeeze (0).squeeze (0).detach ().numpy ()
            plt.imshow (temp, 'grey')
            plt.show ()
        loss = criterion (outputs, masks)
        optimizer.zero_grad ()
        loss.backward ()
        optimizer.step()
        total_loss += loss.item ()
        print (loss)
    #continue
    torch.save ({
        'epoch' : epoch,
        'model_state_dict' : unet.state_dict (),
        'optimizer_state_dict' : optimizer.state_dict (),
        'loss' : loss,
    }, unetPthPath)
    print (f"Epoch[{epoch + 1}], Loss : {total_loss:.4f}")
