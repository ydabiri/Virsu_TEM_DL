%reset
import torch
import matplotlib.pyplot as plt
from torchvision import models,datasets, transforms
import torchvision as tv
from torch import nn
from torch import optim
import torch.nn.functional as F
import json
import shutil
# import matplotlib.image as mpimg
import sys
import numpy as np
from PIL import Image
import os
# from torch.autograd import Variable
from sklearn import metrics
from matplotlib.pyplot import figure
import scikitplot as skplt
import time
import pandas as pd
import matplotlib.patheffects as PathEffects
from sklearn.manifold import TSNE
import random
import seaborn as sns
from sklearn.decomposition import PCA
import gcs


gc.collect()


gc.collect()
torch.manual_seed(60)
torch.cuda.manual_seed(60)
random.seed(10)

rootDir = 'path to files for pca and t-SNE'

for subdir, dirs, files in os.walk(rootDir):
    for file in files:
        filePath = subdir + os.sep + file

        if filePath.endswith('.png'):
          currentImage=Image.open(filePath).convert('L')
          currentImage.save(filePath)

##############################################################
data = 'path to files for pca and t-SNE'
train = data
valid = data
test = data

data_transforms = transforms.Compose([tv.transforms.Grayscale(num_output_channels=1),
                                      transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485],[0.229])
                                                         ])

image_datasets = datasets.ImageFolder(train, transform=data_transforms)
image_test_datasets = datasets.ImageFolder(test, transform=data_transforms)

dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=1, shuffle=False)
datatestloaders = torch.utils.data.DataLoader(image_test_datasets, batch_size=1, shuffle=False)
#########################################################################
imageDimesion = 224*224
numberOfImages = 1600

imagesData = torch.Tensor(numberOfImages,imageDimesion+1)
imageInOneRow = torch.Tensor(1,imageDimesion+1)

counter1 = 0
for data, label in dataloaders:

    data = data.view(1,-1)
    imageInOneRow[0,0] = label
    imageInOneRow[0,1:] = data
    imagesData[counter1,:] = imageInOneRow
    counter1 = counter1 + 1

# change lables so that they match with exact labels
classToLabel = image_datasets.class_to_idx
labeltoClass = {labelsImageData:realLabel for realLabel,labelsImageData in classToLabel.items()}
###########################################################################
imagesDataArray = imagesData.numpy()
pcaData = imagesDataArray[:,1:]
labelsData = imagesDataArray[:,0]
##########################################################################
random.seed(10)

time_start = time.time()
pca = PCA(n_components=5)
pcaResult = pca.fit_transform(pcaData)

#############################################################################
pcaDf = pd.DataFrame(columns = ['pca1','pca2','pca3','pca4'])

pcaDf['pca1'] = pcaResult[:,0]
pcaDf['pca2'] = pcaResult[:,1]
pcaDf['pca3'] = pcaResult[:,2]
pcaDf['pca4'] = pcaResult[:,3]
pcaDf['pca5'] = pcaResult[:,4]

print('Variance per principal component: {}'.format(pca.explained_variance_ratio_))
###########################################################################
numberToName = {"1": "Adenovirus", "2": "Astrovirus",
  "3": "CCHF", "4": "Cowpox",
"5": "Dengue", "6": "Ebola",
"7": "Influenza", "8": "Lassa",
"9": "Marburg", "10": "Norovirus",
"11": "Orf", "12": "Papilloma",
"13": "Rift Valley", "14": "Rotavirus",
"15": "WestNile","16":"SARS-CoV-2"}

def fScatter(valuesOfData, ColorsFromLabels):
    numClasses = len(np.unique(ColorsFromLabels))
    palette = np.array(sns.color_palette("hls", numClasses))

    fig = plt.figure(figsize=(20, 20))
    axisAx = plt.subplot(aspect='equal')
    figPlot = axisAx.scatter(valuesOfData[:,0], valuesOfData[:,1], lw=0, s=40, c=palette[ColorsFromLabels.astype(np.int)])


    axisAx.axis('off')
    axisAx.axis('tight')

    names = []

    for counter in range(0,numClasses):


        xMean = valuesOfData[ColorsFromLabels == counter, :]
        xtext, ytext = xMean.mean(axis=0)
        realClassNumber = labeltoClass[counter]
        name = axisAx.text(xtext, ytext, numberToName[realClassNumber], fontsize=20)
        name.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()])
        names.append(name)

    return fig, axisAx, figPlot, names
##############################################################################
topTwoComps = pcaDf[['pca1','pca2']] # first and second principal components

fScatter(topTwoComps.values,labelsData)
###############################################################################
timeStart = time.time()


tsne = TSNE(n_components=2, init='pca',random_state=0,perplexity=50,n_iter=10000,learning_rate=100)
t0 = time.time()
results_tsne = tsne.fit_transform(pcaDf)
#############################################################################
fScatter(results_tsne, labelsData)
