import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.cuda import _sleep
from torch.utils import data
import utilize
import pandas as pd
import Luna16Dataset
import model
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import argparse
import time
import random

def getInt (x) :
    if (x >= 0.5) :
        return 1
    else :
        return 0

def tumorExtract (newCandidatePath, tumorPath) :
    if os.path.exists (tumorPath) :
        tumorSet = pd.read_csv (tumorPath)
        return tumorSet
    candidate = pd.read_csv (newCandidatePath)
    tumorSet = candidate[candidate['class'] == 1]
    tumorSet.to_csv (tumorPath, index=False)
    return tumorSet

def generateTest (tumorPath, newCandidatePath, testPath) :
    df_a = pd.read_csv (tumorPath).head (275)
    df_b = pd.read_csv (newCandidatePath).head (275)
    merged = pd.concat ([df_a, df_b], ignore_index=True)
    merged.to_csv (testPath, index=False)
    return merged

def init (candidate, newCandidatePath, basePath) :
    if os.path.exists (newCandidatePath) :
        candidate = pd.read_csv (newCandidatePath)
        return candidate
    print (candidate.shape)
    purgeList = []
    for i in range (0, candidate.shape[0]) :
        print (i)
        rowi = candidate.iloc[i]
        fileName = rowi['seriesuid']
        picPath = basePath + "/data/subset0/" + fileName + ".mhd"
        if not os.path.exists (picPath) :
            purgeList.append (fileName)
    print (purgeList)
    candidate = candidate[~candidate['seriesuid'].isin (purgeList)]
    candidate.reset_index (drop=True, inplace=True)
    candidate.to_csv (newCandidatePath, index=False)
    return candidate

def printLesions (annotations, basePath) :
    print (annotations)
    print (annotations.shape)
    for i in range (0, annotations.shape[0]) :
        rowi = annotations.iloc[i]
        fileName = rowi['seriesuid']
        #print (fileName)
        picPath = basePath + "/data/subset0/" + fileName + ".mhd"
        if os.path.exists (picPath) :
            print (fileName)
            cubeTuple = utilize.getRealxyz (i, annotations, picPath)
            print (cubeTuple)
            utilize.showPic (picPath, cubeTuple[0], cubeTuple[1], cubeTuple[2], 32)

def Extract_benign_cube (annotations, benignPath, subset0Path) :
    tot = 0
    for i in range (0, annotations.shape[0]) :
        rowi = annotations.iloc[i]
        name = rowi['seriesuid'] + '.mhd'
        is_benign = rowi['class']
        if not is_benign :
            tot += 1
            path = os.path.join (subset0Path, name)
            patch = utilize.extract_patch (i, path, annotations)
            np.save (os.path.join (benignPath, f"{tot:06d}.npy"), patch)
            print ("tot = ", tot)
    exit (0)

def main () :
    parser = argparse.ArgumentParser ()
    parser.add_argument('--eval', action='store_true', help='Only run evaluation, not training')
    args = parser.parse_args ()
    print (args)

    isshow = int (input ("need image show :"))
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

    candidate = init (candidate, newCandidatePath, basePath)
    tumorSet = tumorExtract (newCandidatePath, tumorPath)
    test = generateTest (tumorPath, newCandidatePath, testPath)

    if isshow :
        printLesions (tumorSet, basePath)

    #utilize.dataImprovment (tumorSet, subset0Path, cubePath)
    #Extract_benign_cube (candidate, benignPath, subset0Path)
    print (candidate.shape)
    print (tumorSet.shape)
    print (test.shape)
    device = torch.device ("cuda" if torch.cuda.is_available () else "cpu")
    '''
    dataset = Luna16Dataset.luna16Dataset (
        imageDir = subset0Path,
        annotations = test,
        patch_size = 64
    )
    '''
    #loader = DataLoader (dataset, batch_size=1, shuffle=True)
    data_list = Luna16Dataset.collect_data_paths (benignPath, cubePath)
    #random.shuffle (data_list)
    dataset = Luna16Dataset.NpyDataset (data_list)
    loader = DataLoader (dataset, batch_size=1, shuffle=True, num_workers=4)
    print ("loader create success.")
    cnn = model.Tumor3DCNN ().to (device)
    criterion = torch.nn.BCEWithLogitsLoss ()
    c = torch.nn.CrossEntropyLoss ()
    lr = 1e-4
    optimizer = optim.Adam (cnn.parameters (), lr=lr)
    start_epoch = 0
    num_epochs = 10
    '''
    print (subset0Path)
    print (len (dataset))
    for i in range (0, len (dataset)) :
        temp = dataset[i]
        if temp :
            print (temp)
    '''
    checkPointPath = os.path.join (srcPath, "checkpoint.pth")
    if os.path.exists (checkPointPath) :
        checkpoint = torch.load (checkPointPath)
        cnn.load_state_dict (checkpoint['model_state_dict'])
        optimizer.load_state_dict (checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print ("load success.")
    #print (start_epoch)
    correctTot = 0
    wrongTot = 0
    #cnn = cnn.to (device)
    
    '''
    input_tensor, label = dataset[0]
    input_tensor = input_tensor.unsqueeze(0).to(device)
    label = torch.tensor([[1.0]], device=device)
    print ("overfitting test :")
    for i in range (1000) :
        cnn.train ()
        optimizer.zero_grad ()
        output = cnn (input_tensor)
        loss = criterion (output, label)
        loss.backward ()
        optimizer.step ()
        print (loss)
    exit (0)
    '''

    for epoch in range (start_epoch, num_epochs) :
        if args.eval :
            print ("in eval mode")
            cnn.eval ()
            with torch.no_grad () :
                for inputs, labels in loader :
                    labels = labels.view (-1, 1)
                    print ("labels = ", labels)
                    inputs = inputs.to (device)
                    labels = labels.to (device)
                    outputs = cnn (inputs)
                    print ("outputs = ", outputs)
                    if getInt (outputs) == getInt (labels) :
                        correctTot += 1
                    else :
                        wrongTot += 1
                        print ("Error : labels = ", labels, ", cnn = ", outputs)
                    print ("correctTot = ", correctTot, "wrongTot = ", wrongTot)
            continue
        cnn.train ()
        total_loss = 0.0
        tot = 0
        for inputs, labels in loader :
            tot += 1
            print (tot)
            labels = labels.view (-1, 1)
            print ("labels = ", labels)
            inputs, labels = inputs.to (device), labels.to (device)
            optimizer.zero_grad ()
            outputs = cnn (inputs)
            print ("outputs = ", outputs)
            loss = criterion (outputs, labels)
            loss.backward ()
            optimizer.step ()
            total_loss += loss
            if getInt (outputs) == getInt (labels) :
                correctTot += 1
            else :
                wrongTot += 1
            print ("correctTot = ", correctTot, "wrongTot = ", wrongTot)
            if tot < 500 :
                continue
            print ("start to saving (-_-)")
            print (loss)
            time.sleep (1)            
            torch.save ({
                'epoch' : epoch,
                'model_state_dict' : cnn.state_dict (),
                'optimizer_state_dict' : optimizer.state_dict (),
                'loss' : loss,
            }, checkPointPath)
            print ("saving success :)")
            tot = 0
            #print (loss)
        avg_loss = total_loss / len (loader)
        print ("epoch : ", epoch, "loss = ", avg_loss)
        torch.save ({
            'epoch' : epoch,
            'model_state_dict' : cnn.state_dict (),
            'optimizer_state_dict' : optimizer.state_dict (),
            'loss' : loss,
        }, checkPointPath)
    return 0

if __name__ == '__main__' :
    main ()
