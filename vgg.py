%reset
import matplotlib.pyplot as plt
import torch
from torchvision import models,datasets, transforms
from torch import nn
from torch import optim
import torch.nn.functional as F
import json
import shutil
# import matplotlib.image as mpimg
# import sys
import numpy as np
from PIL import Image
import os
# from torch.autograd import Variable
from sklearn import metrics
from matplotlib.pyplot import figure
import scikitplot as skplt
import gc
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd

gc.collect()
torch.manual_seed(60)
torch.cuda.manual_seed(60)
numberOfClasses = 16 #number of viruse classes
####################################################################
rootDir = 'path to files'
for subdir, dirs, files in os.walk(rootDir):
    for file in files:
        filePath = subdir + os.sep + file

        if filePath.endswith('.png'):
          currentImage=Image.open(filePath).convert('RGB')
          currentImage.save(filePath)
####################################################################
data = 'path to files'
train = data + '/train'
valid = data + '/valid'
test = data + '/test'

data_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

image_datasets = datasets.ImageFolder(train, transform=data_transforms)
image_test_datasets = datasets.ImageFolder(test, transform=data_transforms)

dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=100, shuffle=True)
datatestloaders = torch.utils.data.DataLoader(image_test_datasets, batch_size=320, shuffle=True)

with open('path to catToName.json', 'r') as f:
    cat_to_name = json.load(f)
####################################################################
vgg = models.vgg16(pretrained=True)

vgg.class_to_idx = image_datasets.class_to_idx
#####################################################################
numberFeatures = 25088
outFeaturesS = 528

for param in vgg.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(numberFeatures, outFeaturesS)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(outFeaturesS, numberOfClasses)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))

vgg.classifier = classifier

modelBeforeTraining = vgg
######################################################################
model = modelBeforeTraining

epochs = 2000
steps = 0
running_loss = 0
print_every = 1
testLossMin = np.inf
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr = 0.00001)
model.to(device)

testLoss = []
trainLoss = []
epochNumber = []
accuracyArray = []

for epoch in range(epochs):
    for inputs, labels in dataloaders:

        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model.forward(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    if steps % print_every == 0:
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in datatestloaders:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)

                equals = top_class == labels.view(*top_class.shape)

                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


        print(f"Epoch {epoch+1}/{epochs}.. "
              f"Train loss: {running_loss:.3f}.. "
              f"Test loss: {test_loss:.3f}.. "
              f"Test accuracy: {accuracy:.3f}")

        trainLoss.append(running_loss)
        testLoss.append(test_loss)
        epochNumber.append(epoch+1)
        accuracyArray.append(accuracy)

        running_loss = 0
        model.train()

        potentialLossMin = testLoss[-1]

        if potentialLossMin < testLossMin:

          model.epochs = epoch

          checkpoint = {'epoch': model.epochs,
                          'state_dict': model.state_dict,
                        'class_to_idx': model.class_to_idx,
                          'classifier': model.classifier}

          torch.save(checkpoint, 'checkpoint.pth')
          testLossMin = potentialLossMin

trainLoss = np.array(trainLoss)
testLoss = np.array(testLoss)
epochNumber = np.array(epochNumber)
accuracyArray = np.array(accuracyArray)

fig, ax1 = plt.subplots()

ax2 = ax1.twinx()
ax1.plot(epochNumber, trainLoss, 'b-',linewidth=5.0)
ax2.plot(epochNumber, testLoss, 'r-',linewidth=5.0)

ax1.set_xlabel('Epoch Number')
ax1.set_ylabel('Train NLLLoss')
ax2.set_ylabel('Test NLLoss')

plt.show()

print('max accuracy is: ',np.max(accuracyArray))

model.eval()
###############################################
model = modelBeforeTraining

checkpoint = torch.load('checkpointVGG.pth')
model.state_dict = checkpoint['state_dict']
model.class_to_idx = checkpoint['class_to_idx']
model.epochs = checkpoint['epoch']
##############################################

device = torch.device("cpu")
model.to(device)

numberOfBatch = 320
imageDatasetsVal = datasets.ImageFolder('path to test files', transform=data_transforms)
dataloadersVal = torch.utils.data.DataLoader(imageDatasetsVal, batch_size=numberOfBatch, shuffle=True)
for inputs, labels in dataloadersVal:
    inputs, labels = inputs.to(device), labels.to(device)

    label_to_classNumber = {label: classNumber for classNumber, label in model.class_to_idx.items()}

    logps = model.forward(inputs)
    ps = torch.exp(logps)

    top_ps, top_classes = ps.topk(1, dim=1)

    lablesArray = labels.numpy()

    top_classesArray = top_classes.numpy()
    top_classesArray = top_classesArray.reshape(1,numberOfBatch)
    top_classesArray = top_classesArray[0]

    lablesArrayList = lablesArray.tolist()

    labelsMatrixLabelsPredict = top_classesArray
    labelsMatrixClassNumberPredict = [label_to_classNumber[x] for x in labelsMatrixLabelsPredict]
    labelsMatrixPredict = [cat_to_name[x] for x in labelsMatrixClassNumberPredict]


    labelsMatrixLabels = lablesArray
    labelsMatrixClassNumber = [label_to_classNumber[x] for x in labelsMatrixLabels]
    labelsMatrix = [cat_to_name[x] for x in labelsMatrixClassNumber]

    cmLabels = list(set(labelsMatrix))
    confusionMatrix = confusion_matrix(labelsMatrix, labelsMatrixPredict,labels=cmLabels)
    ############################################################
df = pd.DataFrame(confusionMatrix,index=cmLabels, columns=cmLabels)

plt.figure(figsize = (10,7))
sn.heatmap(df, annot=True)
#################################################################
probabilities = ps
probabilities = probabilities.detach().numpy()
skplt.metrics.plot_roc(lablesArray,probabilities,title='',figsize=(12,12),text_fontsize='large')
##################################################################
gc.collect()
torch.manual_seed(60)
torch.cuda.manual_seed(60)

imageDatasetsVal = datasets.ImageFolder('path to test files', transform=data_transforms)
dataloadersVal = torch.utils.data.DataLoader(imageDatasetsVal, batch_size=1, shuffle=True)
class_to_idx = model.class_to_idx
index_to_class = {label: classNumber for classNumber, label in model.class_to_idx.items()}

device = torch.device("cpu")
inputs, labels = next(iter(dataloadersVal))

image = inputs[0,:,:,:]
trueLabel = labels.item()
trueIndex = index_to_class[trueLabel]
trueClass = cat_to_name[trueIndex]
print('True class is = ',trueClass)

inputs, labels = inputs.to(device), labels.to(device)
model = model.to(device)
logps = model.forward(inputs)
probabilities = torch.exp(logps)
top_probabilities, top_indexs = probabilities.topk(15,dim=1)
topProbsList1 = top_probabilities.cpu()
topProbsList = topProbsList1.detach().numpy()
top_indexs = top_indexs.numpy()

class_to_idx = model.class_to_idx
idx_to_class = {folderNumber: virusClassNumber for virusClassNumber, folderNumber in class_to_idx.items()}

closestVirusIndex = []
for x in top_indexs[0]:
  closestVirusIndex.append(x)

closestVirusName = []
for x in closestVirusIndex:
  virusIndex = idx_to_class[x]
  closestVirusName.append(cat_to_name[virusIndex])

imshow(image)
fig, ax = plt.subplots()
ax.barh(closestVirusName,topProbsList[0])
